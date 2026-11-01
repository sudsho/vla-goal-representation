"""Counterfactual-mean steering vectors v(c->d) = mean over nominal frames of task c of
h_d - h_c at the decision token, layer L (same frame, instruction swapped). Frames come from episodes
disjoint from the closed-loop test inits (C0 inits 0-39 vs test inits 45-47). Writes data/steer/cf_L{L}_{c}to{d}.npy
and plans/cfsteer.jsonl (closed loop, decision token + every decoded token)."""
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


def main(L=16, betas="2", n_eps=6):
    L, n_eps = int(L), int(n_eps)
    pol = OpenVLAPolicy("data/ckpt/openvla-7b-finetuned-libero-goal", "libero_goal", [32])
    meta = [m for m in load_meta("c0") if m["success"] and m["init"] < 40]
    os.makedirs("data/steer", exist_ok=True)
    plan, i = [], 9700
    for c, d in PAIRS:
        diffs = []
        for m in [m for m in meta if m["task"] == c][:n_eps]:
            imgs = np.load(os.path.join(ROOT, "c0", f"img{m['id']:05d}.npz"))["img"]
            for t in range(0, min(len(imgs), 80), 10):
                pil = Image.fromarray(center_crop(imgs[t])).convert("RGB")
                inp, ids_c = pol._encode(TASKS[c], pil); _, ids_d = pol._encode(TASKS[d], pil)
                with torch.no_grad():
                    hc = pol.vla(input_ids=ids_c, pixel_values=inp["pixel_values"], output_hidden_states=True).hidden_states[L][0, -1]
                    hd = pol.vla(input_ids=ids_d, pixel_values=inp["pixel_values"], output_hidden_states=True).hidden_states[L][0, -1]
                diffs.append((hd - hc).float().cpu().numpy())
        v = np.mean(diffs, 0).astype(np.float32)
        path = f"data/steer/cf_L{L}_{c}to{d}.npy"; np.save(path, v)
        print(c, d, len(diffs), "|v|", round(float(np.linalg.norm(v)), 1), flush=True)
        for beta in [float(b) for b in betas.split(",")]:
            for r0 in range(3):
                plan.append(dict(id=i, task=c, init=45 + r0, instruction=TASKS[c], cond=f"cf_L{L}_b{beta:g}", task2=d,
                                 steer=dict(layer=L, vec=path, beta=beta))); i += 1
    with open("plans/cfsteer.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in plan)
    print("plan", len(plan))


if __name__ == "__main__":
    main(*sys.argv[1:])
