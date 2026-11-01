"""Activation patching with a clean metric: the first action token is fully determined by the prefill,
so we patch the residual stream after block l at a position group (values from the instruction-d run,
same image) and report the normalized logit difference of the first action token,
  T = (LD_patch - LD_c) / (LD_d - LD_c),  LD = logit[k_d] - logit[k_c],  k_* = argmax under c / d,
over frames with k_c != k_d. Groups: instruction tokens, post-instruction template tokens (excl. last),
decision token, all text."""
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
LAYERS = [1] + list(range(2, 33, 2))


def main(n_eps=6, steps="5,20,40"):
    n_eps, steps = int(n_eps), [int(s) for s in steps.split(",")]
    pol = OpenVLAPolicy("data/ckpt/openvla-7b-finetuned-libero-goal", "libero_goal", [32])
    vla, proc, dev = pol.vla, pol.processor, pol.device
    blocks = vla.language_model.model.layers
    embed = vla.language_model.model.embed_tokens
    meta = [m for m in load_meta("c0") if m["success"]]

    def encode(instr, img):
        x = proc(f"In: What action should the robot take to {instr}?\nOut:", Image.fromarray(img).convert("RGB")).to(dev, dtype=torch.bfloat16)
        return torch.cat((x["input_ids"], torch.tensor([[29871]], device=dev)), 1), x["pixel_values"]

    @torch.no_grad()
    def first_logits(ids, pv, hook_mod=None, hook=None):
        hd = hook_mod.register_forward_hook(hook) if hook_mod is not None else None
        out = vla(input_ids=ids, pixel_values=pv, output_hidden_states=hook_mod is None)
        if hd is not None:
            hd.remove()
        return out.logits[0, -1].float(), out.hidden_states

    rows = []
    for c, d in PAIRS:
        for m in [m for m in meta if m["task"] == c][:n_eps]:
            imgs = np.load(os.path.join(ROOT, "c0", f"img{m['id']:05d}.npz"))["img"]
            for t in steps:
                if t >= len(imgs):
                    continue
                img = center_crop(imgs[t])
                ids_c, pv = encode(TASKS[c], img)
                ids_d, _ = encode(TASKS[d], img)
                lc, _ = first_logits(ids_c, pv)
                ld, hs = first_logits(ids_d, pv)
                kc, kd = int(lc.argmax()), int(ld.argmax())
                if kc == kd:
                    continue
                LD = lambda z: float(z[kd] - z[kc])
                LDc, LDd = LD(lc), LD(ld)
                n_img = hs[0].shape[1] - ids_c.shape[1]
                diff = (ids_c[0] != ids_d[0]).nonzero().flatten().cpu().numpy()
                span = np.arange(diff.min(), diff.max() + 1) + n_img
                last = hs[0].shape[1] - 1
                P = {"instr": span, "tmpl": np.arange(span.max() + 1, last), "dec": np.array([last]),
                     "text": np.arange(span.min(), last + 1)}
                rec = dict(c=c, d=d, ep=m["id"], t=t, gap=LDd - LDc)
                for L in LAYERS:
                    src = hs[L]
                    mod = blocks[L - 1]
                    for g, pos in P.items():
                        if len(pos) == 0:
                            continue
                        pos_t = torch.tensor(pos, device=dev)

                        def hook(mod_, inp, out, pos_t=pos_t, src=src):
                            h = out[0] if isinstance(out, tuple) else out
                            if h.shape[1] > 1:
                                h[:, pos_t] = src[:, pos_t].to(h.dtype)
                            return out
                        lp, _ = first_logits(ids_c, pv, mod, hook)
                        rec[f"{g}_L{L}"] = (LD(lp) - LDc) / (LDd - LDc)
                rows.append(rec)
                print(c, d, m["id"], t, {g: [round(rec.get(f"{g}_L{L}", np.nan), 2) for L in LAYERS[::2]] for g in P}, flush=True)
    json.dump(rows, open("data/results/patching3.json", "w"))
    print("done", len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
