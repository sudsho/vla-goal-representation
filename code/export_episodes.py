"""Write a compact per-episode summary, data/episodes/<run>.jsonl, with the executed goal of every rollout
(y_exec, t_exec) and the episode length T. make_numbers4.py and paired_within.py read these files when the raw
rollouts in data/runs are not present (or when USE_EPISODE_SUMMARY=1 is set)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from alib import load_meta, exec_goal, ROOT

RUNS = ["c0", "negation", "negation2", "switch", "switch_late", "under", "novel", "steer_pilot", "steer2_pilot",
        "patch_cl", "cfsteer", "steer3", "stream16", "stream24"]
OUT = os.path.join(ROOT, "..", "episodes")


def main():
    os.makedirs(OUT, exist_ok=True)
    for run in RUNS:
        rows = []
        for m in load_meta(run):
            G = np.load(os.path.join(ROOT, run, f"ep{m['id']:05d}.npz"))["goals"]
            y, t = exec_goal(G)
            r = dict(m)
            r.update(y_exec=int(y), t_exec=int(t), T=int(len(G)))
            rows.append(r)
        with open(os.path.join(OUT, f"{run}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(run, len(rows), flush=True)


if __name__ == "__main__":
    main()
