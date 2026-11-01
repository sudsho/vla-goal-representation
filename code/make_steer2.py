"""Steering v2: per-position class-mean differences over the post-instruction positions
(M_d - M_c, shape n_post x 4096) at one layer, with per-position norm-matched random controls."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from make_plan import TASKS

PAIRS = [(1, 8), (8, 1), (4, 8), (8, 4), (2, 9), (9, 2), (0, 3), (3, 0), (7, 1), (1, 7), (6, 8), (8, 6)]


def main(layer=16, betas="1", n_init=3, pairs="all", kinds="probe,rand", out="steer2", init0=45):
    layer, n_init, init0 = int(layer), int(n_init), int(init0)
    M = np.load(f"data/steer/postmeans_L{layer}.npy")  # 10 x n_post x 4096
    rng = np.random.RandomState(layer)
    prs = PAIRS if pairs == "all" else [tuple(int(x) for x in p.split(">")) for p in pairs.split(",")]
    plan, i = [], {"steer2": 8000}.get(out, 9000)
    for beta in [float(b) for b in betas.split(",")]:
        for c, d in prs:
            v = (M[d] - M[c]).astype(np.float32)
            r = rng.randn(*v.shape).astype(np.float32)
            r *= (np.linalg.norm(v, axis=1) / np.linalg.norm(r, axis=1))[:, None]
            base = f"data/steer/v2_L{layer}_{c}to{d}"
            np.save(base + ".npy", v); np.save(base + "_rand.npy", r)
            for kind in kinds.split(","):
                for r0 in range(n_init):
                    plan.append(dict(id=i, task=c, init=init0 + r0, instruction=TASKS[c], cond=f"s2_{kind}_L{layer}_b{beta:g}",
                                     task2=d, steer=dict(layer=layer, vec=base + ("_rand" if kind == "rand" else "") + ".npy", beta=beta)))
                    i += 1
    with open(f"plans/{out}.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in plan)
    print(out, len(plan), "| per-pos |v| (first pair):", np.round(np.linalg.norm(M[prs[0][1]] - M[prs[0][0]], axis=1), 1))


if __name__ == "__main__":
    main(*sys.argv[1:])
