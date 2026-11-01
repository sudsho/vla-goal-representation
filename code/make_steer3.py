"""Closed-loop fixed-vector steering with the two vector types that transfer offline (patching6):
 posvec (layer 16): v_j = mean (h_d - h_c) at the decision token (j=0) and at action token j (teacher-forced c prefix)
 span   (layer 8):  per-position mean (h_d - h_c) over the instruction span (prefill)
Vectors come from nominal episodes with init < 40 (test inits 45-47). Writes data/steer/*.npy, plans/steer3.jsonl."""
import json, os, sys
import numpy as np
os.environ.setdefault("USE_FLAX", "0")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image
from rollout import OpenVLAPolicy, center_crop
from make_plan import TASKS
from alib import load_meta, ROOT

PAIRS = [(0, 1), (1, 0), (2, 5), (5, 2), (4, 9), (9, 4)]


def main(n_eps=6):
    n_eps = int(n_eps)
    pol = OpenVLAPolicy("data/ckpt/openvla-7b-finetuned-libero-goal", "libero_goal", [32])
    vla, dev = pol.vla, pol.device
    meta = [m for m in load_meta("c0") if m["success"] and m["init"] < 40]
    plan, i = [], 9600
    for c, d in PAIRS:
        P, S8, off = [], [], None
        for m in [m for m in meta if m["task"] == c][:n_eps]:
            imgs = np.load(os.path.join(ROOT, "c0", f"img{m['id']:05d}.npz"))["img"]
            for t in (5, 20, 40, 60):
                if t >= len(imgs):
                    continue
                pil = Image.fromarray(center_crop(imgs[t])).convert("RGB")
                inp, ids_c = pol._encode(TASKS[c], pil); _, ids_d = pol._encode(TASKS[d], pil); pv = inp["pixel_values"]
                with torch.no_grad():
                    tc = vla.generate(ids_c, pixel_values=pv, max_new_tokens=7, do_sample=False)[0, -7:-1]
                    pre = tc[None]
                    hc = vla(input_ids=torch.cat([ids_c, pre], 1), pixel_values=pv, output_hidden_states=True).hidden_states
                    hd = vla(input_ids=torch.cat([ids_d, pre], 1), pixel_values=pv, output_hidden_states=True).hidden_states
                n_img = hc[0].shape[1] - (ids_c.shape[1] + 6)
                dec = n_img + ids_c.shape[1] - 1
                P.append((hd[16][0, dec:dec + 7] - hc[16][0, dec:dec + 7]).float().cpu().numpy())
                diff = (ids_c[0] != ids_d[0]).nonzero().flatten().cpu().numpy()
                s0, s1 = n_img + diff.min(), n_img + diff.max() + 1
                S8.append((hd[8][0, s0:s1] - hc[8][0, s0:s1]).float().cpu().numpy())
                Sp = n_img + ids_c.shape[1]  # prefill length (no action tokens)
                off = [int(Sp - s0), int(Sp - s1)]
        vp, vs = np.mean(P, 0).astype(np.float32), np.mean(S8, 0).astype(np.float32)
        fp, fs = f"data/steer/pos_L16_{c}to{d}.npy", f"data/steer/span_L8_{c}to{d}.npy"
        np.save(fp, vp); np.save(fs, vs)
        print(c, d, len(P), "|v_pos|", np.round(np.linalg.norm(vp, axis=1), 1), "|v_span|", round(float(np.linalg.norm(vs)), 1), "off", off, flush=True)
        for r0 in range(3):
            plan.append(dict(id=i, task=c, init=45 + r0, instruction=TASKS[c], cond="posvec_L16_b2", task2=d,
                             steer=dict(layer=16, vec=fp, beta=2.0, mode="posvec"))); i += 1
            plan.append(dict(id=i, task=c, init=45 + r0, instruction=TASKS[c], cond="span_L8_b1", task2=d,
                             steer=dict(layer=8, vec=fs, beta=1.0, mode="span", span_from_end=off))); i += 1
    with open("plans/steer3.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in plan)
    print("plan", len(plan))


if __name__ == "__main__":
    main(*sys.argv[1:])
