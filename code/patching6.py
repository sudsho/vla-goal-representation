"""Fixed-vector controls for the writing claims (offline, per-dimension transfer on the patching frames):
(i)  position-specific counterfactual vectors at layer 16: v_j = mean over OTHER episodes of (h_d - h_c) at the
     decision token (j=0) and at action position j=1..6 under a teacher-forced prefix (c's greedy tokens);
     v_j is added at position j during greedy decoding (beta in {1,2}).
(ii) upstream fixed vectors at layer 8: per-position means of (h_d - h_c) over the instruction span, added in
     every forward pass (beta=1).
If (i) fails while decode-stream patching succeeds, the write must be prefix-dependent, not merely position-specific."""
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
    ACT0 = vla.vocab_size - 256
    to_act = lambda tok: vla.bin_centers[np.clip(vla.vocab_size - np.array(tok) - 1, 0, vla.bin_centers.shape[0] - 1)][:6]

    @torch.no_grad()
    def fwd(x, pv, hooks=(), hs=False):
        hd = [blocks[L - 1].register_forward_hook(h) for L, h in hooks]
        out = vla(input_ids=x, pixel_values=pv, output_hidden_states=hs)
        for h in hd:
            h.remove()
        return out

    def greedy(ids, pv, hooks_fn=None):
        toks = []
        for j in range(7):
            x = torch.cat([ids, torch.tensor([toks], device=dev, dtype=ids.dtype)], 1) if toks else ids
            lg = fwd(x, pv, hooks_fn(j) if hooks_fn else ()).logits[0, -1]
            toks.append(int(lg[ACT0:].argmax()) + ACT0)
        return toks

    frames = []
    for c, d in PAIRS:
        for m in [m for m in meta if m["task"] == c][:n_eps]:
            imgs = np.load(os.path.join(ROOT, "c0", f"img{m['id']:05d}.npz"))["img"]
            frames += [(c, d, m["id"], t, imgs[t]) for t in steps if t < len(imgs)]
    cache = []
    for c, d, ep, t, img in frames:
        pil = Image.fromarray(center_crop(img)).convert("RGB")
        inp, ids_c = pol._encode(TASKS[c], pil); _, ids_d = pol._encode(TASKS[d], pil); pv = inp["pixel_values"]
        tc = greedy(ids_c, pv); td = greedy(ids_d, pv)
        pre = torch.tensor([tc[:6]], device=dev, dtype=ids_c.dtype)  # teacher-forced prefix: c's tokens
        hc = fwd(torch.cat([ids_c, pre], 1), pv, hs=True).hidden_states
        hd = fwd(torch.cat([ids_d, pre], 1), pv, hs=True).hidden_states
        n_img = hc[0].shape[1] - (ids_c.shape[1] + 6)
        dec = n_img + ids_c.shape[1] - 1
        diff = (ids_c[0] != ids_d[0]).nonzero().flatten().cpu().numpy()
        span = np.arange(diff.min(), diff.max() + 1) + n_img
        cache.append(dict(c=c, d=d, ep=ep, t=t, pil=pil, dec=int(dec), span=span, a_c=to_act(tc), a_d=to_act(td),
                          v16=(hd[16][0, dec:dec + 7] - hc[16][0, dec:dec + 7]).float().cpu().numpy(),
                          v8=(hd[8][0, span] - hc[8][0, span]).float().cpu().numpy()))
    rows = []
    for fr in cache:
        c, d = fr["c"], fr["d"]
        oth = [x for x in cache if x["c"] == c and x["d"] == d and x["ep"] != fr["ep"]]
        V16 = torch.tensor(np.mean([x["v16"] for x in oth], 0), device=dev, dtype=torch.bfloat16)  # 7 x 4096
        V8 = torch.tensor(np.mean([x["v8"] for x in oth], 0), device=dev, dtype=torch.bfloat16)    # span x 4096
        inp, ids_c = pol._encode(TASKS[c], fr["pil"]); pv = inp["pixel_values"]
        res = {}
        for beta in (1.0, 2.0):
            def hooks_fn(j, beta=beta):
                def hk(mod, i_, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h[:, fr["dec"]: fr["dec"] + j + 1] += beta * V16[: j + 1]
                    return out
                return ((16, hk),)
            res[f"posvec_L16_b{beta:g}"] = to_act(greedy(ids_c, pv, hooks_fn)).tolist()

        def hooks8(j):
            def hk(mod, i_, out):
                h = out[0] if isinstance(out, tuple) else out
                h[:, torch.tensor(fr["span"], device=dev)] += V8
                return out
            return ((8, hk),)
        res["instrvec_L8"] = to_act(greedy(ids_c, pv, hooks8)).tolist()
        rows.append(dict(c=c, d=d, ep=fr["ep"], t=fr["t"], a_c=fr["a_c"].tolist(), a_d=fr["a_d"].tolist(), **res))
        gap = fr["a_d"] - fr["a_c"]
        print(c, d, fr["t"], {k: np.round(np.where(np.abs(gap) > 0.1, (np.array(v) - fr["a_c"]) / np.where(gap == 0, 1, gap), np.nan), 2).tolist() for k, v in res.items()}, flush=True)
    json.dump(rows, open("data/results/patching6.json", "w"))
    print("done", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
