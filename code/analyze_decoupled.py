"""Decoupled conditions (instruction != behaviour): mid-episode switches and novel instructions.
Probes are trained on C0 only (where prompt == intent), then applied zero-shot. The conformal
threshold is calibrated on C0 matched episodes; the switch run supplies follow/persist episodes."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from alib import *
from analyze_c0 import kin_feats

OUT = os.path.join(ROOT, "..", "results")


def train_full(eps, feat, **pk):
    X, y, g, _ = stack_steps(eps, feat)
    return Probe(**pk).fit(X, y, ep_weights(g))


def running_from(s, k):
    """running max of s restricted to t >= k (compliance with the instruction active since k)."""
    out = np.full(len(s), -np.inf)
    if k < len(s):
        out[k:] = np.maximum.accumulate(s[k:])
    return out


def main(folds=5, layer=32):
    c0 = load_run("c0", keys=("H", "state", "objpos", "logits"))
    sw = load_run("switch", keys=("H", "state", "objpos", "logits"))
    nv = load_run("novel", keys=("H", "state", "objpos", "logits")) if os.path.isdir(os.path.join(ROOT, "novel")) else []
    for e in c0 + sw + nv:
        e["kin"] = kin_feats(e); e["Hall"] = e.pop("H").astype(np.float32)
    res = {}

    # ---------------- behaviour
    for e in sw:
        y, a, b = e["y_exec"], e["task"], e["task2"]
        e["outcome"] = "follow" if y == b else "persist" if y == a else "none" if y == NONE else "other"
    ks = sorted({e["switch_t"] for e in sw})
    res["switch_outcomes"] = {str(k): {o: int(sum(1 for e in sw if e["switch_t"] == k and e["outcome"] == o))
                                       for o in ("follow", "persist", "none", "other")} for k in ks}
    res["novel_outcomes"] = {}
    for e in nv:
        fx = e.get("first_hit_x", [-1])
        key = e["instruction"]
        d = res["novel_outcomes"].setdefault(key, {"novel_goal": 0, "trained_goal": {}, "none": 0})
        if fx and fx[0] >= 0 and (e["y_exec"] == NONE or fx[0] <= e["t_exec"]):
            d["novel_goal"] += 1
        elif e["y_exec"] != NONE:
            d["trained_goal"][TASKS[e["y_exec"]]] = d["trained_goal"].get(TASKS[e["y_exec"]], 0) + 1
        else:
            d["none"] += 1
    print(json.dumps({k: res[k] for k in ("switch_outcomes", "novel_outcomes")}, indent=1), flush=True)

    # ---------------- zero-shot probes (trained on all C0) applied after the switch
    post = {}
    probes = {}
    for li, L in enumerate(LAYERS):
        for e in c0 + sw + nv:
            e["h"] = e["Hall"][:, li]
        pr = probes[L] = train_full(c0, "h")
        rows = {"persist": [], "follow": [], "none": []}
        for e in sw:
            if e["outcome"] not in rows:
                continue
            k = e["switch_t"]
            lp_ = pr.logp(e["h"][k:])
            if len(lp_) == 0:
                continue
            am = lp_.argmax(1)
            rows[e["outcome"]].append((np.mean(am == e["task"]), np.mean(am == e["task2"])))
        post[f"L{L}"] = {o: dict(p_old=float(np.mean([r[0] for r in v])) if v else None,
                                 p_new=float(np.mean([r[1] for r in v])) if v else None, n=len(v))
                         for o, v in rows.items()}
        # novel: early prediction (first half before t_exec) of the executed trained goal
        acc = []
        for e in nv:
            if e["y_exec"] != NONE and e["t_exec"] > 4:
                h = e["h"][: e["t_exec"] // 2]
                acc.append(np.mean(pr.logp(h).argmax(1) == e["y_exec"]))
        post[f"L{L}"]["novel_early_acc"] = float(np.mean(acc)) if acc else None
        print("layer", L, post[f"L{L}"], flush=True)
    kin_pr = train_full(c0, "kin", k=64, C=1.0)
    rows = {"persist": [], "follow": []}
    for e in sw:
        if e["outcome"] in rows:
            am = kin_pr.logp(e["kin"][e["switch_t"]:]).argmax(1)
            rows[e["outcome"]].append((np.mean(am == e["task"]), np.mean(am == e["task2"])))
    post["kin"] = {o: dict(p_old=float(np.mean([r[0] for r in v])) if v else None,
                           p_new=float(np.mean([r[1] for r in v])) if v else None) for o, v in rows.items()}
    res["post_switch_decoding"] = post

    # ---------------- monitor on switch episodes (calibrated on C0 matched, cross-fitted)
    c0_best = f"L{layer}"  # selected on C0 (highest nominal mismatch AUROC in the layer sweep)
    res["monitor_layer"] = c0_best
    li = LAYERS.index(int(c0_best[1:]))
    for e in c0 + sw:
        e["h"] = e["Hall"][:, li]
    c0_lp = crossfit_logp(c0, "h", folds)
    c0_kin = crossfit_logp(c0, "kin", folds, k=64, C=1.0)
    pr, kpr = probes[int(c0_best[1:])], kin_pr
    y0 = np.array([e["y_exec"] for e in c0]); c0c = np.array([e["task"] for e in c0])
    matched = np.flatnonzero(y0 == c0c)

    def sw_score(e, lp_):
        return running_from(np.delete(lp_, e["task2"], 1).max(1) - lp_[:, e["task2"]], e["switch_t"])

    sc = {"intent": ([mismatch_score(c0_lp[i], c0c[i]) for i in range(len(c0))],
                     [sw_score(e, pr.logp(e["h"])) for e in sw]),
          "kin-intent": ([mismatch_score(c0_kin[i], c0c[i]) for i in range(len(c0))],
                         [sw_score(e, kpr.logp(e["kin"])) for e in sw]),
          "entropy": ([entropy_score(e["logits"]) for e in c0],
                      [running_from(_ent(e["logits"]), e["switch_t"]) for e in sw])}
    # success probe (instructed-task success) trained on C0
    for e in c0:
        e["y_bak"] = e["y_exec"]; e["y_exec"] = int(not e["success"])
    sp_cf = crossfit_logp(c0, "h", folds)
    sp = train_full(c0, "h")
    for e in c0:
        e["y_exec"] = e.pop("y_bak")
    sc["success-probe"] = ([np.maximum.accumulate(l[:, 1] - l[:, 0]) for l in sp_cf],
                           [running_from((lambda l: l[:, 1] - l[:, 0])(sp.logp(e["h"])), e["switch_t"]) for e in sw])

    pos = [i for i, e in enumerate(sw) if e["outcome"] in ("persist", "other", "none")]
    posw = [i for i, e in enumerate(sw) if e["outcome"] in ("persist", "other")]
    neg = [i for i, e in enumerate(sw) if e["outcome"] == "follow"]
    rng = np.random.RandomState(0)
    mon = {}
    for name, (S0, S1) in sc.items():
        f0 = np.array([s[-1] for s in S0]); f1 = np.array([s[-1] for s in S1])
        out = {"auroc_persist_vs_follow": float(auroc(f1[posw], f1[neg])) if posw and neg else None}
        for alpha in (0.05, 0.1):
            fa, fa_sw, tp, tpw, lead = [], [], [], [], []
            for rep in range(200):
                perm = rng.permutation(matched)
                cal, te = perm[: len(perm) // 2], perm[len(perm) // 2:]
                tau = conformal_tau(f0[cal], alpha)
                fa.append(np.mean(f0[te] > tau))
                if neg: fa_sw.append(np.mean(f1[neg] > tau))
                if pos: tp.append(np.mean(f1[pos] > tau))
                if posw: tpw.append(np.mean(f1[posw] > tau))
                for i in posw:
                    ta = first_alarm(S1[i], tau)
                    if ta >= 0:
                        lead.append(sw[i]["t_exec"] - ta)
            out[f"a{alpha}"] = dict(far_c0=float(np.mean(fa)), far_follow=float(np.mean(fa_sw)) if fa_sw else None,
                                   tpr=float(np.mean(tp)) if tp else None, tpr_wrong_goal=float(np.mean(tpw)) if tpw else None,
                                   lead_median=float(np.median(lead)) if lead else None,
                                   lead_frac_pos=float(np.mean(np.array(lead) > 0)) if lead else None)
        mon[name] = out
    res["switch_monitor"] = mon
    sweep = {}
    for L in LAYERS:
        li2 = LAYERS.index(L)
        for e in c0 + sw:
            e["h"] = e["Hall"][:, li2]
        cf = crossfit_logp(c0, "h", folds)
        f0 = np.array([mismatch_score(cf[i], c0c[i])[-1] for i in range(len(c0))])
        S1 = [sw_score(e, probes[L].logp(e["h"])) for e in sw]
        f1 = np.array([s_[-1] for s_ in S1])
        mis0 = y0 != c0c
        tp, tpw, fa = [], [], []
        for rep in range(200):
            perm = rng.permutation(matched); cal, te = perm[: len(perm) // 2], perm[len(perm) // 2:]
            tau = conformal_tau(f0[cal], 0.1)
            fa.append(np.mean(f0[te] > tau)); tp.append(np.mean(f1[pos] > tau)); tpw.append(np.mean(f1[posw] > tau))
        sweep[f"L{L}"] = dict(c0_auroc=float(auroc(f0[mis0], f0[~mis0])), sw_auroc_persist=float(auroc(f1[posw], f1[neg])),
                              sw_auroc_allmis=float(auroc(f1[pos], f1[neg])), far_c0=float(np.mean(fa)),
                              tpr=float(np.mean(tp)), tpr_wrong_goal=float(np.mean(tpw)))
        print("sweep", L, {k: round(v, 3) for k, v in sweep[f"L{L}"].items()}, flush=True)
    res["layer_sweep"] = sweep
    json.dump(res, open(os.path.join(OUT, "decoupled.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("post_switch_decoding", "switch_monitor")}, indent=1))


def _ent(logits):
    z = logits.astype(np.float64); z = z - z.max(-1, keepdims=True)
    p = np.exp(z); p /= p.sum(-1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(-1).mean(-1)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
