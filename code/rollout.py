"""LIBERO rollouts for OpenVLA with activation logging.

Per episode we store (npz): last-token residual-stream activations at selected language-model layers
(prefill pass of every control step), the 7x256 action-bin logits, executed actions, EE state,
the truth value of the goal predicate of EVERY task in the suite at every step, and the
policy input images. Preprocessing mirrors openvla/experiments/robot (JPEG round-trip,
lanczos3 resize, 0.9-area center crop) so success rates match the published numbers.
"""
import argparse, json, math, os, sys, time
import numpy as np

import tensorflow as tf  # preprocessing parity with OpenVLA's RLDS pipeline (CPU only)
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
import torch
torch.set_num_threads(4)
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import robosuite_parse_problem

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_WAIT = 10
DUMMY = [0, 0, 0, 0, 0, 0, -1]


# ----------------------------------------------------------------------------- preprocessing
def resize_image(img, size=224):
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def center_crop(img, crop_scale=0.9):
    x = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)[None]
    s = math.sqrt(crop_scale)
    o = (1 - s) / 2
    x = tf.image.crop_and_resize(x, [[o, o, o + s, o + s]], [0], (224, 224))[0]
    x = tf.image.convert_image_dtype(tf.clip_by_value(x, 0, 1), tf.uint8, saturate=True)
    return x.numpy()


def obs_image(obs):
    return resize_image(obs["agentview_image"][::-1, ::-1])  # 180-degree rotation as in training


def quat2axisangle(q):
    q = q.copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


# ----------------------------------------------------------------------------- policy
class OpenVLAPolicy:
    def __init__(self, ckpt, unnorm_key, layers, device="cuda:0"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
            attn_implementation="sdpa").to(device).eval()
        with open(os.path.join(ckpt, "dataset_statistics.json")) as f:
            self.vla.norm_stats = json.load(f)
        if unnorm_key not in self.vla.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        self.unnorm_key = unnorm_key
        self.layers = layers
        self._buf = {}
        self.hooks = []  # steering/patching hooks may be appended by callers
        blocks = self.vla.language_model.model.layers
        for li in layers:
            blocks[li - 1].register_forward_hook(self._make_hook(li))  # output of block li-1 == hidden_states[li]

    def set_steer(self, layer=None, vec=None, mode=None, span_from_end=None):
        """Activation addition: h <- h + vec at the output of block `layer` (decision token + decode steps).
        mode='posvec': vec[0] at the decision token (prefill), vec[j] at the j-th decoded action token.
        mode='span':   vec (span x d) added in prefill at positions [S-a, S-b) counted from the sequence end."""
        for hd in self.hooks:
            hd.remove()
        self.hooks = []
        if layer is None:
            return
        v = torch.tensor(vec, dtype=torch.bfloat16, device=self.device)
        if mode in ("posvec", "span"):
            state = {"j": 0}

            def hook2(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if mode == "posvec":
                    if h.shape[1] > 1:
                        state["j"] = 0; h[:, -1] += v[0]
                    else:
                        state["j"] += 1
                        if state["j"] < v.shape[0]:
                            h += v[state["j"]]
                elif h.shape[1] > 1:
                    S = h.shape[1]; a, b = span_from_end
                    h[:, S - a: S - b] += v
                return out
            self.hooks.append(self.vla.language_model.model.layers[layer - 1].register_forward_hook(hook2))
            return

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if v.dim() == 2:  # per-position vectors for the last n prefill positions; last row at decode steps
                if h.shape[1] > 1:
                    h[:, -v.shape[0]:] += v
                else:
                    h += v[-1]
            elif h.shape[1] > 1:
                h[:, -1] += v
            else:
                h += v
            return out
        self.hooks.append(self.vla.language_model.model.layers[layer - 1].register_forward_hook(hook))

    def _make_hook(self, li):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:  # prefill pass only (not KV-cached decode steps)
                self._buf[li] = h[0, -1].float().cpu().numpy().astype(np.float16)
        return hook

    def _encode(self, instruction, pil):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, pil).to(self.device, dtype=torch.bfloat16)
        ids = inputs["input_ids"]
        if not torch.all(ids[:, -1] == 29871):
            ids = torch.cat((ids, torch.tensor([[29871]], device=ids.device)), dim=1)
        return inputs, ids

    def _patch_hook(self, patch, pil, ids):
        """Closed-loop activation patching: overwrite the residual stream after block `layer` at a position
        group (instr / dec / text) with its value under the counterfactual instruction on the same frame."""
        _, ids_d = self._encode(patch["src"], pil)
        assert ids_d.shape == ids.shape, "patching needs equal-length instructions"
        hs = self.vla(input_ids=ids_d, pixel_values=self._pv, output_hidden_states=True).hidden_states
        L = patch["layer"]
        n_img = hs[0].shape[1] - ids.shape[1]
        diff = (ids[0] != ids_d[0]).nonzero().flatten().cpu().numpy()
        last = hs[0].shape[1] - 1
        span = np.arange(diff.min(), diff.max() + 1) + n_img
        pos = {"instr": span, "dec": np.array([last]), "text": np.arange(span.min(), last + 1)}[patch["group"]]
        pos_t, src = torch.tensor(pos, device=self.device), hs[L]

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                h[:, pos_t] = src[:, pos_t].to(h.dtype)
            return out
        return self.vla.language_model.model.layers[L - 1].register_forward_hook(hook)

    @torch.no_grad()
    def act_stream(self, img, instruction, sp):
        """Closed-loop decode-stream patching: manual greedy decoding of the 7 action tokens; at every step the
        layer-L residual at the decision token and at all generated action tokens is overwritten with its value
        from the counterfactual instruction run fed the same action prefix."""
        img = center_crop(img)
        pil = Image.fromarray(img).convert("RGB")
        inputs, ids = self._encode(instruction, pil)
        _, ids_d = self._encode(sp["src"], pil)
        assert ids_d.shape == ids.shape
        pv, L = inputs["pixel_values"], sp["layer"]
        blk = self.vla.language_model.model.layers[L - 1]
        act0 = self.vla.vocab_size - 256
        toks, lgs, H = [], [], None
        for j in range(7):
            ext = torch.tensor([toks], device=self.device, dtype=ids.dtype) if toks else None
            xc = torch.cat([ids, ext], 1) if toks else ids
            xd = torch.cat([ids_d, ext], 1) if toks else ids_d
            src = self.vla(input_ids=xd, pixel_values=pv, output_hidden_states=True).hidden_states[L]
            pos = torch.arange(src.shape[1] - 1 - j, src.shape[1], device=self.device)

            def hook(mod, inp, out, pos=pos, src=src):
                h = out[0] if isinstance(out, tuple) else out
                h[:, pos] = src[:, pos].to(h.dtype)
                return out
            hd = blk.register_forward_hook(hook)
            self._buf = {}
            lg = self.vla(input_ids=xc, pixel_values=pv).logits[0, -1].float()
            hd.remove()
            if j == 0:
                H = np.stack([self._buf[li] for li in self.layers], 0)
            toks.append(int(lg[act0:].argmax()) + act0); lgs.append(lg[-256:].cpu().numpy())
        tok = np.array(toks)
        disc = np.clip(self.vla.vocab_size - tok - 1, 0, self.vla.bin_centers.shape[0] - 1)
        norm = self.vla.bin_centers[disc]
        st = self.vla.norm_stats[self.unnorm_key]["action"]
        mask = np.array(st.get("mask", np.ones_like(st["q01"], dtype=bool)))
        hi, lo = np.array(st["q99"]), np.array(st["q01"])
        a = np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)
        a[-1] = -np.sign(2 * a[-1] - 1)
        return a, np.stack(lgs).astype(np.float16), H

    @torch.no_grad()
    def act(self, img, instruction, patch=None):
        img = center_crop(img)
        pil = Image.fromarray(img).convert("RGB")
        inputs, ids = self._encode(instruction, pil)
        self._pv = inputs["pixel_values"]
        ph = self._patch_hook(patch, pil, ids) if patch else None
        self._buf = {}
        out = self.vla.generate(ids, pixel_values=inputs["pixel_values"], attention_mask=inputs["attention_mask"],
                                max_new_tokens=7, do_sample=False, output_scores=True, return_dict_in_generate=True)
        if ph is not None:
            ph.remove()
        tok = out.sequences[0, -7:].cpu().numpy()
        logits = torch.stack(out.scores, 0)[:, 0, -256:].float().cpu().numpy()  # 7 x 256 action-bin logits
        disc = np.clip(self.vla.vocab_size - tok - 1, 0, self.vla.bin_centers.shape[0] - 1)
        norm = self.vla.bin_centers[disc]
        st = self.vla.norm_stats[self.unnorm_key]["action"]
        mask = np.array(st.get("mask", np.ones_like(st["q01"], dtype=bool)))
        hi, lo = np.array(st["q99"]), np.array(st["q01"])
        a = np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)
        a[-1] = -np.sign(2 * a[-1] - 1)  # normalize+binarize gripper, then invert (OpenVLA convention)
        H = np.stack([self._buf[li] for li in self.layers], 0)
        return a, logits.astype(np.float16), H


# ----------------------------------------------------------------------------- env / predicates
def all_goal_states(suite):
    root = get_libero_path("bddl_files")
    out = []
    for k in range(suite.n_tasks):
        t = suite.get_task(k)
        out.append(robosuite_parse_problem(os.path.join(root, t.problem_folder, t.bddl_file))["goal_state"])
    return out


def eval_goals(env, goals):
    base = env.env
    res = []
    for gs in goals:
        ok = True
        for s in gs:
            try:
                ok = base._eval_predicate(s) and ok
            except KeyError:
                ok = False
        res.append(ok)
    return np.array(res, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--plan", required=True, help="jsonl of episodes: {id, task, init, instruction, cond}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--layers", default="4,8,12,16,20,24,28,32")
    ap.add_argument("--save_images", type=int, default=1)
    ap.add_argument("--stop_on_any", type=int, default=0, help="also stop when any other task goal holds for 20 steps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    plan = [json.loads(l) for l in open(args.plan)]
    plan = [p for i, p in enumerate(plan) if i % args.nworkers == args.worker]
    done_ids = set()
    meta_path = os.path.join(args.out, f"meta_w{args.worker}.jsonl")
    for mf in os.listdir(args.out):
        if mf.startswith("meta_w") and mf.endswith(".jsonl"):
            done_ids |= {json.loads(l)["id"] for l in open(os.path.join(args.out, mf))}
    plan = [p for p in plan if p["id"] not in done_ids]
    print(f"[w{args.worker}] {len(plan)} episodes to run", flush=True)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    goals = all_goal_states(suite)
    layers = [int(x) for x in args.layers.split(",")]
    pol = OpenVLAPolicy(args.ckpt, args.suite, layers)
    max_steps = MAX_STEPS[args.suite]

    env_cache = {}
    for ep in plan:
        t0 = time.time()
        task = suite.get_task(ep["task"])
        if ep["task"] not in env_cache:
            for e in env_cache.values():
                e.close()
            env_cache.clear()
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env_cache[ep["task"]] = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env = env_cache[ep["task"]]
        st = ep.get("steer")
        pol.set_steer(st["layer"], np.load(st["vec"]) * st["beta"], st.get("mode"), st.get("span_from_end")) if st else pol.set_steer()
        env.seed(0)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(ep["task"])[ep["init"]])
        for _ in range(NUM_WAIT):
            obs, _, _, _ = env.step(DUMMY)
        okeys = sorted(k for k in obs if k.endswith("_pos") and not k.startswith("robot0"))
        Hs, Ls, As, Ss, Gs, Is, Os = [], [], [], [], [], [], []
        xg = [[x] for x in ep.get("extra_goals", [])]  # novel goal predicates (evaluated like BDDL goals)
        XGs = []
        success, other_run = False, np.zeros(len(goals) + len(xg), int)
        tm = dict(pre=0.0, model=0.0, env=0.0)
        for t in range(max_steps):
            t1 = time.time()
            img = obs_image(obs)
            t2 = time.time(); tm["pre"] += t2 - t1
            instr = ep["instruction"] if t < ep.get("switch_t", 10**9) else ep["instruction2"]
            if ep.get("stream_patch"):
                a, lg, H = pol.act_stream(img, instr, ep["stream_patch"])
            else:
                a, lg, H = pol.act(img, instr, ep.get("patch"))
            t3 = time.time(); tm["model"] += t3 - t2
            state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            Os.append(np.concatenate([obs[k] for k in okeys]))
            obs, _, done, _ = env.step(a.tolist())
            g = eval_goals(env, goals)
            gx = eval_goals(env, xg) if xg else np.zeros(0, bool)
            XGs.append(gx)
            tm["env"] += time.time() - t3
            Hs.append(H); Ls.append(lg); As.append(a); Ss.append(state); Gs.append(g)
            if args.save_images:
                Is.append(img)
            if done:
                success = True
                break
            other_run = np.where(np.concatenate([g, gx]), other_run + 1, 0)
            if args.stop_on_any and (other_run >= 20).any():
                break
        rec = dict(ep, success=bool(success), steps=len(As), secs=round(time.time() - t0, 1),
                   final_goals=Gs[-1].astype(int).tolist(),
                   first_hit=[int(np.argmax(col)) if col.any() else -1 for col in np.array(Gs).T],
                   first_hit_x=[int(np.argmax(col)) if col.any() else -1 for col in np.array(XGs).T] if xg else [])
        np.savez(os.path.join(args.out, f"ep{ep['id']:05d}.npz"), H=np.stack(Hs), logits=np.stack(Ls),
                 actions=np.stack(As).astype(np.float32), state=np.stack(Ss).astype(np.float32), goals=np.stack(Gs), objpos=np.stack(Os).astype(np.float32), objkeys=np.array(okeys), xgoals=np.array(XGs))
        if args.save_images:
            np.savez_compressed(os.path.join(args.out, f"img{ep['id']:05d}.npz"), img=np.stack(Is))
        with open(meta_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[w{args.worker}] ep{ep['id']} task{ep['task']} cond={ep['cond']} succ={success} "
              f"steps={len(As)} hits={rec['first_hit']} {rec['secs']}s tm={ {k: round(v / len(As), 3) for k, v in tm.items()} }", flush=True)


if __name__ == "__main__":
    main()
