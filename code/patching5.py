"""Direct test of the re-reading mechanism + a practical downstream write.
(1) Decode-stream patching: greedy-decode the 7 action tokens manually; at every step overwrite the layer-L
    residual at the decision token AND at every already-generated action token with its value from the
    d-run fed the SAME action prefix (teacher-forced counterfactual). If downstream writes fail only because
    later tokens re-read the instruction, this should transfer all dimensions.
(2) Counterfactual-mean steering: v = mean over frames of other episodes of (h_d - h_c) at the decision token
    (layer 16, same frames, leave-one-episode-out), added at the decision token and every decoded token.
Per-dimension transfer (a - a_c)_j / (a_d - a_c)_j is reported for dims with |a_d - a_c|_j > 0.1."""
import json, os, sys
import numpy as np
os.environ.setdefault("USE_FLAX", "0")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image
from rollout import OpenVLAPolicy, center_crop
from make_plan import TASKS
from alib import load_meta, ROOT

PAIRS = [(0, 1), (1, 0), (2, 5), (5, 2), (2, 6), (6, 2), (4, 9), (9, 4), (5, 6), (6, 5)]


def main(n_eps=4, steps="5,20,40"):
    n_eps, steps = int(n_eps), [int(s) for s in steps.split(",")]
    pol = OpenVLAPolicy("data/ckpt/openvla-7b-finetuned-libero-goal", "libero_goal", [32])
    vla, dev = pol.vla, pol.device
    blocks = vla.language_model.model.layers
    meta = [m for m in load_meta("c0") if m["success"]]
    ACT0 = vla.vocab_size - 256  # action tokens occupy the last 256 ids

    def to_act(tok):
        return vla.bin_centers[np.clip(vla.vocab_size - np.array(tok) - 1, 0, vla.bin_centers.shape[0] - 1)][:6]

    @torch.no_grad()
    def fwd(ids, pv, hook=None, L=None, hs=False):
        hd = blocks[L - 1].register_forward_hook(hook) if hook else None
        out = vla(input_ids=ids, pixel_values=pv, output_hidden_states=hs)
        if hd: hd.remove()
        return out

    def greedy(ids, pv, step_hook=None, L=None):
        """manual greedy decoding of 7 action tokens; step_hook(j, prefix) -> forward hook or None"""
        toks = []
        for j in range(7):
            x = torch.cat([ids, torch.tensor([toks], device=dev, dtype=ids.dtype)], 1) if toks else ids
            hk = step_hook(j, toks) if step_hook else None
            lg = fwd(x, pv, hk, L).logits[0, -1]
            toks.append(int(lg[ACT0:].argmax()) + ACT0)
        return toks

    frames = []
    for c, d in PAIRS:
        for m in [m for m in meta if m["task"] == c][:n_eps]:
            imgs = np.load(os.path.join(ROOT, "c0", f"img{m['id']:05d}.npz"))["img"]
            frames += [(c, d, m["id"], t, imgs[t]) for t in steps if t < len(imgs)]
    # pass 1: clean actions and decision-token differences at L16 (for counterfactual means)
    cache = []
    for c, d, ep, t, img in frames:
        pil = Image.fromarray(center_crop(img)).convert("RGB")
        inp, ids_c = pol._encode(TASKS[c], pil); _, ids_d = pol._encode(TASKS[d], pil); pv = inp["pixel_values"]
        hc = fwd(ids_c, pv, hs=True).hidden_states; hd_ = fwd(ids_d, pv, hs=True).hidden_states
        n_img = hc[0].shape[1] - ids_c.shape[1]
        diff = (ids_c[0] != ids_d[0]).nonzero().flatten().cpu().numpy()
        span0, dec = int(diff.min()) + n_img, hc[0].shape[1] - 1
        cache.append(dict(c=c, d=d, ep=ep, t=t, pil=pil, dec=dec, span0=span0,
                          dv16=(hd_[16][0, dec] - hc[16][0, dec]).float().cpu().numpy(),
                          a_c=to_act(greedy(ids_c, pv)), a_d=to_act(greedy(ids_d, pv))))
    rows = []
    for fr in cache:
        c, d = fr["c"], fr["d"]
        inp, ids_c = pol._encode(TASKS[c], fr["pil"]); _, ids_d = pol._encode(TASKS[d], fr["pil"]); pv = inp["pixel_values"]
        res = {}
        for name, L, first in (("stream_dec_L16", 16, fr["dec"]), ("stream_dec_L24", 24, fr["dec"]), ("stream_text_L16", 16, fr["span0"])):
            def step_hook(j, toks, L=L, first=first):
                xd = torch.cat([ids_d, torch.tensor([toks], device=dev, dtype=ids_d.dtype)], 1) if toks else ids_d
                src = fwd(xd, pv, hs=True).hidden_states[L]
                pos = torch.arange(first if first != fr["dec"] else fr["dec"], src.shape[1], device=dev)
                if first == fr["dec"]:
                    pos = torch.arange(fr["dec"], src.shape[1], device=dev)  # decision token + generated tokens

                def hook(mod, inp_, out, pos=pos, src=src):
                    h = out[0] if isinstance(out, tuple) else out
                    h[:, pos] = src[:, pos].to(h.dtype)
                    return out
                return hook
            res[name] = to_act(greedy(ids_c, pv, step_hook, L)).tolist()
        # counterfactual-mean steering (leave-one-episode-out), decision token + every decoded token, L16
        others = [x["dv16"] for x in cache if x["c"] == c and x["d"] == d and x["ep"] != fr["ep"]]
        v = torch.tensor(np.mean(others, 0), device=dev, dtype=torch.bfloat16)
        for beta in (1.0, 2.0):
            def step_hook(j, toks, beta=beta):
                def hook(mod, inp_, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h[:, fr["dec"]:] += beta * v
                    return out
                return hook
            res[f"cfsteer_L16_b{beta:g}"] = to_act(greedy(ids_c, pv, step_hook, 16)).tolist()
        rows.append(dict(c=c, d=d, ep=fr["ep"], t=fr["t"], a_c=fr["a_c"].tolist(), a_d=fr["a_d"].tolist(),
                         vnorm=float(np.linalg.norm(np.mean(others, 0))), **res))
        gap = fr["a_d"] - fr["a_c"]
        print(c, d, fr["t"], {k: np.round(np.where(np.abs(gap) > 0.1, (np.array(x) - fr["a_c"]) / np.where(gap == 0, 1, gap), np.nan), 2).tolist() for k, x in res.items()}, flush=True)
    json.dump(rows, open("data/results/patching5.json", "w"))
    print("done", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
