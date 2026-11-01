"""Per-action-dimension transfer (tests the 're-reading' mechanism behind the steering null).
For frames of task c and an equal-length instruction d, greedy-decode all 7 action tokens under:
  patch dec@L16, patch instr@L8, patch text@L16 (prefill only, values from the d-run on the same frame),
  class-mean steering at the decision token L12 (+decode) and at post-instruction positions L16 (+decode),
and report per-dimension transfer (a_p - a_c)_j / (a_d - a_c)_j for dims with |a_d - a_c|_j > 0.1."""
import json, os, sys
import numpy as np
os.environ.setdefault("USE_FLAX", "0")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image
from rollout import OpenVLAPolicy, center_crop
from make_plan import TASKS
from alib import load_meta, load_run, ROOT, LAYERS

PAIRS = [(0, 1), (1, 0), (2, 5), (5, 2), (2, 6), (6, 2), (4, 9), (9, 4), (5, 6), (6, 5)]


def main(n_eps=4, steps="5,20,40"):
    n_eps, steps = int(n_eps), [int(s) for s in steps.split(",")]
    pol = OpenVLAPolicy("data/ckpt/openvla-7b-finetuned-libero-goal", "libero_goal", [32])
    vla, dev = pol.vla, pol.device
    blocks = vla.language_model.model.layers
    meta = [m for m in load_meta("c0") if m["success"]]
    # class means for steering vectors
    c0 = load_run("c0", keys=("H",), layer_idx=LAYERS.index(12))
    mu12 = {k: np.concatenate([e["H"][:40] for e in c0 if e["task"] == k and e["y_exec"] == k]).mean(0) for k in range(10)}
    post16 = np.load("data/steer/postmeans_L16.npy")

    def act(ids, pv):
        out = vla.generate(ids, pixel_values=pv, max_new_tokens=7, do_sample=False)
        tok = out[0, -7:].cpu().numpy()
        return vla.bin_centers[np.clip(vla.vocab_size - tok - 1, 0, vla.bin_centers.shape[0] - 1)][:6]

    def with_hook(L, fn, ids, pv):
        hd = blocks[L - 1].register_forward_hook(fn)
        a = act(ids, pv); hd.remove(); return a

    rows = []
    for c, d in PAIRS:
        for m in [m for m in meta if m["task"] == c][:n_eps]:
            imgs = np.load(os.path.join(ROOT, "c0", f"img{m['id']:05d}.npz"))["img"]
            for t in steps:
                if t >= len(imgs):
                    continue
                pil = Image.fromarray(center_crop(imgs[t])).convert("RGB")
                inp_c, ids_c = pol._encode(TASKS[c], pil)
                _, ids_d = pol._encode(TASKS[d], pil)
                pv = inp_c["pixel_values"]
                with torch.no_grad():
                    hs = vla(input_ids=ids_d, pixel_values=pv, output_hidden_states=True).hidden_states
                    a_c, a_d = act(ids_c, pv), act(ids_d, pv)
                    n_img = hs[0].shape[1] - ids_c.shape[1]
                    diff = (ids_c[0] != ids_d[0]).nonzero().flatten().cpu().numpy()
                    span = np.arange(diff.min(), diff.max() + 1) + n_img
                    last = hs[0].shape[1] - 1
                    conds = {}
                    for name, L, pos in (("patch_dec_L16", 16, [last]), ("patch_instr_L8", 8, span),
                                         ("patch_text_L16", 16, np.arange(span.min(), last + 1))):
                        pos_t, src = torch.tensor(pos, device=dev), hs[L]

                        def fn(mod, inp, out, pos_t=pos_t, src=src):
                            h = out[0] if isinstance(out, tuple) else out
                            if h.shape[1] > 1:
                                h[:, pos_t] = src[:, pos_t].to(h.dtype)
                            return out
                        conds[name] = with_hook(L, fn, ids_c, pv)
                    for name, L, v in (("steer_dec_L12", 12, torch.tensor(mu12[d] - mu12[c], device=dev, dtype=torch.bfloat16)),
                                       ("steer_post_L16", 16, torch.tensor(post16[d] - post16[c], device=dev, dtype=torch.bfloat16))):
                        def fn(mod, inp, out, v=v):
                            h = out[0] if isinstance(out, tuple) else out
                            if v.dim() == 2:
                                if h.shape[1] > 1: h[:, -v.shape[0]:] += v
                                else: h += v[-1]
                            else:
                                if h.shape[1] > 1: h[:, -1] += v
                                else: h += v
                            return out
                        conds[name] = with_hook(L, fn, ids_c, pv)
                rec = dict(c=c, d=d, ep=m["id"], t=t, a_c=a_c.tolist(), a_d=a_d.tolist(), **{k: v.tolist() for k, v in conds.items()})
                rows.append(rec)
                gap = a_d - a_c
                print(c, d, t, {k: np.round(np.where(np.abs(gap) > 0.1, (np.array(v) - a_c) / np.where(gap == 0, 1, gap), np.nan), 2).tolist() for k, v in conds.items()}, flush=True)
    json.dump(rows, open("data/results/patching4.json", "w"))
    print("done", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
