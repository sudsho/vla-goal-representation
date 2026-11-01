"""C0 analysis: layer-wise intent decoding, early decodability vs controls, conformal mismatch monitor.
Writes data/results/c0.json (+ cached out-of-fold log-probs for figures)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from alib import *

OUT = os.path.join(ROOT, "..", "results"); os.makedirs(OUT, exist_ok=True)


def kin_feats(e):
    S, O = e["state"], e["objpos"]
    return np.concatenate([S, O, S - S[:1], O - O[:1]], 1).astype(np.float32)


def logit_feats(e):
    z = e["logits"].astype(np.float32)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z); p /= p.sum(-1, keepdims=True)
    return (p @ np.linspace(-1, 1, 256)).astype(np.float32)  # expected normalized action per dim (7)


def main(run="c0", folds=5):
    folds = int(folds)
    eps = load_run(run, keys=("H", "state", "objpos", "logits"))
    n = len(eps)
    y = np.array([e["y_exec"] for e in eps]); c = np.array([e["task"] for e in eps])
    succ = np.array([e["success"] for e in eps])
    mis = y != c
    res = dict(n=n, success_rate=float(succ.mean()), n_mismatch=int(mis.sum()),
               n_wrong_goal=int((mis & (y != NONE)).sum()), n_none=int((y == NONE).sum()),
               exec_confusion=np.array([[int(((c == i) & (y == j)).sum()) for j in range(NCLS)] for i in range(10)]).tolist())
    print({k: v for k, v in res.items() if k != "exec_confusion"}, flush=True)

    # --- features per episode
    for e in eps:
        e["kin"] = kin_feats(e); e["act"] = logit_feats(e)
        e["Hall"] = e.pop("H").astype(np.float32)  # T x 8 x 4096

    lp = {}
    for li, L in enumerate(LAYERS):
        for e in eps:
            e["h"] = e["Hall"][:, li]
        lp[f"L{L}"] = crossfit_logp(eps, "h", folds)
        print("layer", L, "done", flush=True)
    lp["kin"] = crossfit_logp(eps, "kin", folds, k=64, C=1.0)
    lp["act"] = crossfit_logp(eps, "act", folds, k=7, C=1.0)

    # --- decodability vs normalized time (fraction of the way to t_exec or episode end)
    bins = np.linspace(0, 1, 11)
    acc = {}
    for name, L in lp.items():
        rows = []
        for sub in ("all", "mismatch"):
            idx = np.flatnonzero(mis) if sub == "mismatch" else np.arange(n)
            curve = []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                hits = []
                for i in idx:
                    tend = eps[i]["t_exec"] if eps[i]["y_exec"] != NONE else eps[i]["T"] - 1
                    tt = np.arange(eps[i]["T"]) / max(tend, 1)
                    m = (tt >= b0) & (tt < b1)
                    if m.any():
                        hits.append((L[i][m].argmax(1) == y[i]).mean())
                curve.append(float(np.mean(hits)) if hits else np.nan)
            rows.append(curve)
        acc[name] = dict(all=rows[0], mismatch=rows[1])
    res["acc_vs_time"] = acc
    # accuracy in the first 10 control steps (before any contact in LIBERO-Goal)
    res["acc_first10"] = {k: float(np.mean([(L[i][:10].argmax(1) == y[i]).mean() for i in range(n)])) for k, L in lp.items()}

    # --- conformal monitor: calibrate on matched episodes, positives = mismatches
    best = max((k for k in lp if k.startswith("L")), key=lambda k: res["acc_first10"][k])
    res["best_layer"] = best
    scores = {"intent": [mismatch_score(lp[best][i], c[i]) for i in range(n)],
              "kin-intent": [mismatch_score(lp["kin"][i], c[i]) for i in range(n)],
              "entropy": [entropy_score(eps[i]["logits"]) for i in range(n)]}
    # success-probe baseline (SAFE-style): same features, binary target = instructed success
    for e, s in zip(eps, succ):
        e["h"] = e["Hall"][:, LAYERS.index(int(best[1:]))]
        e["y_exec_bak"] = e["y_exec"]; e["y_exec"] = int(not s)
    sp = crossfit_logp(eps, "h", folds)
    for e in eps:
        e["y_exec"] = e.pop("y_exec_bak")
    scores["success-probe"] = [np.maximum.accumulate(sp[i][:, 1] - sp[i][:, 0]) for i in range(n)]

    rng = np.random.RandomState(0)
    matched = np.flatnonzero(~mis)
    mon = {}
    for name, S in scores.items():
        final = np.array([s[-1] for s in S])
        out = {"auroc_all": float(auroc(final[mis], final[~mis])),
               "auroc_wrong_goal": float(auroc(final[mis & (y != NONE)], final[~mis]))}
        for alpha in (0.05, 0.1, 0.2):
            fars, tprs, tprw, leads = [], [], [], []
            for rep in range(200):
                perm = rng.permutation(matched)
                cal, te = perm[: len(perm) // 2], perm[len(perm) // 2:]
                tau = conformal_tau(final[cal], alpha)
                fars.append(np.mean(final[te] > tau))
                tprs.append(np.mean(final[mis] > tau))
                wg = np.flatnonzero(mis & (y != NONE))
                tprw.append(np.mean(final[wg] > tau) if len(wg) else np.nan)
                for i in wg:
                    ta = first_alarm(S[i], tau)
                    if ta >= 0:
                        leads.append(eps[i]["t_exec"] - ta)
            out[f"a{alpha}"] = dict(far=float(np.mean(fars)), tpr=float(np.mean(tprs)), tpr_wrong_goal=float(np.nanmean(tprw)),
                                   lead_median=float(np.median(leads)) if leads else None,
                                   lead_frac_pos=float(np.mean(np.array(leads) > 0)) if leads else None)
        mon[name] = out
    res["monitor"] = mon
    json.dump(res, open(os.path.join(OUT, f"{run}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{run}_lp.npz"), **{k: np.array(v, dtype=object) for k, v in lp.items()},
                        y=y, c=c, succ=succ, t_exec=np.array([e["t_exec"] for e in eps]), T=np.array([e["T"] for e in eps]))
    print(json.dumps({k: res[k] for k in ("acc_first10", "best_layer", "monitor")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
