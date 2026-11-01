"""Paired within-pair AUROC differences reported in the paper (CPU only).
Uses the split-0 monitor scores saved by analyze_v3.py (data/results/v3_raw.pkl) and the same pair-cluster
bootstrap (resample the 12 ordered pairs, n=1000, seed 0) that produced within_diff_ci in v3.json.
Redirect times t_r are rebuilt from the run metadata with the filters of analyze_v3.load_all.
Writes data/results/paired_within.json"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from alib import load_run, auroc, NONE, ROOT

W = 10
OUT = os.path.join(ROOT, "..", "results")


def within_pair_auc(scores, labels, pairs):
    scores, labels = np.asarray(scores), np.asarray(labels); vals, ws = [], []
    for p in set(pairs):
        idx = np.array([i for i, q in enumerate(pairs) if q == p])
        if labels[idx].any() and (~labels[idx]).any():
            vals.append(auroc(scores[idx][labels[idx]], scores[idx][~labels[idx]])); ws.append(len(idx))
    return float(np.average(vals, weights=ws)) if vals else np.nan


def load_eps(run):
    """Episode metadata with y_exec, t_exec and T, from data/runs or from the summary in data/episodes."""
    if not os.environ.get("USE_EPISODE_SUMMARY") and os.path.isdir(os.path.join(ROOT, run)):
        return load_run(run, keys=())
    return [json.loads(l) for l in open(os.path.join(ROOT, "..", "episodes", f"{run}.jsonl"))]


def redirect_times():
    eps = load_eps("switch") + load_eps("switch_late")
    for e in eps:
        if e["y_exec"] != NONE:
            e["T"] = min(e["T"], e["t_exec"] + 1)
    red_all = [e for e in eps if not (e["y_exec"] == e["task"] and e["t_exec"] < e["switch_t"]) and e["switch_t"] < e["T"]]
    red = [e for e in red_all if e["T"] - e["switch_t"] >= W]
    return red


def main():
    P = pickle.load(open(os.path.join(OUT, "v3_raw.pkl"), "rb"))
    keep, viol, pairs = P["keep"], np.asarray(P["viol"]), [tuple(p) for p in P["pairs"]]
    red = redirect_times()
    assert len(red) == len(viol), (len(red), len(viol))
    assert [(e["task"], e["task2"]) for e in red] == pairs
    tr_ = np.array([e["switch_t"] for e in red], float)

    def boot(fn, n=1000, seed=0):
        up = sorted(set(pairs), key=str); rng2 = np.random.RandomState(seed); vals = []
        for _ in range(n):
            pick = [up[i] for i in rng2.randint(0, len(up), len(up))]
            idx = np.concatenate([[j for j in range(len(red)) if pairs[j] == q] for q in pick]).astype(int)
            v = fn(idx)
            if v == v:
                vals.append(v)
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    def wp(sc):
        return lambda idx: within_pair_auc(np.asarray(sc)[idx], viol[idx], [pairs[j] for j in idx])

    a = keep["goal-L32"]["f"]
    out = {"timing_within": within_pair_auc(tr_, viol, pairs)}
    for name, b in (("time", tr_), ("vt", keep["vision+text"]["f"]), ("sp", keep["success-probe"]["f"]), ("ent", keep["entropy"]["f"])):
        d = within_pair_auc(a, viol, pairs) - within_pair_auc(b, viol, pairs)
        out[f"L32_minus_{name}"] = d
        out[f"L32_minus_{name}_ci"] = boot(lambda idx: wp(a)(idx) - wp(b)(idx))
        out[f"L32_minus_{name}_ci_10k"] = boot(lambda idx: wp(a)(idx) - wp(b)(idx), n=10000)
        print(name, round(d, 3), [round(x, 3) for x in out[f"L32_minus_{name}_ci"]], [round(x, 3) for x in out[f"L32_minus_{name}_ci_10k"]], flush=True)
    json.dump(out, open(os.path.join(OUT, "paired_within.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
