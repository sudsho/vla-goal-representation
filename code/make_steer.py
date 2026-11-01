"""Steering plan: diff-in-means task directions v_{c->d} = mu_d - mu_c at one layer (decision token,
matched C0 episodes, first `tmax` steps), plus norm-matched random directions as controls."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from alib import load_run, LAYERS
from make_plan import TASKS

PAIRS = [(1, 8), (8, 1), (4, 8), (8, 4), (2, 9), (9, 2), (0, 3), (3, 0), (7, 1), (1, 7)]


def main(layer=12, tmax=40, betas="1,2", n_init=5):
    layer, tmax, n_init = int(layer), int(tmax), int(n_init)
    eps = load_run("c0", keys=("H",), layer_idx=LAYERS.index(layer))
    mu = {}
    for k in range(10):
        X = [e["H"][:tmax] for e in eps if e["task"] == k and e["y_exec"] == k]
        mu[k] = np.concatenate(X).mean(0)
        print(k, len(X), "matched eps")
    vdir = os.path.join("data", "steer"); os.makedirs(vdir, exist_ok=True)
    rng = np.random.RandomState(0)
    plan, i = [], 5000
    for beta in [float(b) for b in betas.split(",")]:
        for c, d in PAIRS:
            v = (mu[d] - mu[c]).astype(np.float32)
            r = rng.randn(len(v)).astype(np.float32); r *= np.linalg.norm(v) / np.linalg.norm(r)
            np.save(f"{vdir}/L{layer}_{c}to{d}.npy", v); np.save(f"{vdir}/L{layer}_{c}to{d}_rand.npy", r)
            for kind in ("probe", "rand"):
                vec = f"{vdir}/L{layer}_{c}to{d}" + ("_rand" if kind == "rand" else "") + ".npy"
                for r0 in range(n_init):
                    plan.append(dict(id=i, task=c, init=45 + r0, instruction=TASKS[c], cond=f"st_{kind}_b{beta:g}",
                                     task2=d, steer=dict(layer=layer, vec=vec, beta=beta))); i += 1
    os.makedirs("plans", exist_ok=True)
    with open("plans/steer.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in plan)
    print("steer plan", len(plan), "| |v| examples:", [round(float(np.linalg.norm(mu[d] - mu[c])), 1) for c, d in PAIRS[:4]],
          "| |mu|:", round(float(np.linalg.norm(mu[1])), 1))


if __name__ == "__main__":
    main(*sys.argv[1:])
