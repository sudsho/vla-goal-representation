"""Monitor analysis used in the paper (Table 1 and Figure 2b).
- protocol match: non-nominal rollouts are truncated at t_exec+1 like nominal ones (which end at `done`)
- fixed horizon: episode statistic S taken at min(T-1, start+H) with H=60 (removes length confound)
- debounce uses full windows only: S_t = max_{t'<=t} min_{t'-w<u<=t'} s_u, undefined (-inf) until w steps
- redirect: AUROC (all / wrong-goal vs followed), WITHIN-PAIR AUROC, instantaneous AUROC at the switch,
  pair-prior baseline (leave-one-out non-follow rate of the ordered pair; no observation)
- negation: AUROC executed vs not, spec-aware success probe (alarm on predicted success of X)
- cluster bootstrap CIs (redirect: by ordered pair; negation: by task); point estimates averaged over splits
- conformal TPR/FA at alpha=0.1 on held-out nominal, followed redirects, compliant negations
Writes data/results/v3.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from alib import *
from analyze_c0 import kin_feats
from analyze_decoupled import _ent

OUT = os.path.join(ROOT, "..", "results")
N_SHORT = 0
H, W = 60, 10
GL = [4, 32]


def bow_fn(vocab):
    return lambda s: np.array([w in s.lower().replace(chr(39), " ").split() for w in vocab], np.float32)


def load_all():
    D = {k: (load_run(k, keys=("H", "state", "objpos", "logits")) if os.path.isdir(os.path.join(ROOT, k)) else [])
         for k in ("c0", "switch", "switch_late", "negation", "negation2")}
    vocab = sorted({w for eps in D.values() for e in eps for w in (e["instruction"] + " " + e.get("instruction2", "")).lower().replace(chr(39), " ").split()})
    bow = bow_fn(vocab)
    for run, eps in D.items():
        for e in eps:
            T = e["T"]
            if run != "c0" and e["y_exec"] != NONE:
                T = min(T, e["t_exec"] + 1)  # match nominal protocol (episode ends when a goal first holds)
            e["T"] = T
            e["kin"] = kin_feats(e)[:T]; e["ent"] = _ent(e["logits"])[:T]
            Hh = e.pop("H")
            for L in GL:
                e[f"h{L}"] = Hh[:T, LAYERS.index(L)].astype(np.float32)
            fp = os.path.join(ROOT, "..", "feats", run, f"ep{e['id']:05d}.npy")
            V = np.load(fp).astype(np.float32)[:T] if os.path.exists(fp) else None
            if V is not None and len(V) == T:
                k = e.get("switch_t", 10 ** 9)
                e["vt"] = np.concatenate([V, 4.0 * np.stack([bow(e["instruction"] if t < k else e["instruction2"]) for t in range(T)])], 1)
            else:
                e["vt"] = None
    red_all = [e for e in D["switch"] + D["switch_late"] if not (e["y_exec"] == e["task"] and e["t_exec"] < e["switch_t"]) and e["switch_t"] < e["T"]]
    red = [e for e in red_all if e["T"] - e["switch_t"] >= W]  # a debounced monitor needs W post-switch steps
    global N_SHORT
    N_SHORT = len(red_all) - len(red)
    for e in red:
        y = e["y_exec"]
        e["outcome"] = "follow" if y == e["task2"] else "persist" if y == e["task"] else "stall" if y == NONE else "other"
        e["pair"] = (e["task"], e["task2"])
    neg = D["negation"] + D["negation2"]
    for e in neg:
        e["outcome"] = "forbidden" if e["y_exec"] == e["task"] else "stall" if e["y_exec"] == NONE else "other"
    return D["c0"], red, neg


def dstat(s, start, w=W):
    """debounced running max with full windows only"""
    out = np.full(len(s), -np.inf)
    for t in range(start + w - 1, len(s)):
        out[t] = s[t - w + 1: t + 1].min()
    return np.maximum.accumulate(out)


def at_h(S, start):
    return S[min(len(S) - 1, start + H)]


def cluster_boot_auc(scores, labels, clusters, n=1000, seed=0):
    scores, labels = np.asarray(scores), np.asarray(labels); clusters = np.array([str(c) for c in clusters])
    uc = np.unique(clusters); rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        pick = rng.choice(uc, len(uc)); idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
        if labels[idx].all() or (~labels[idx]).all():
            continue
        vals.append(auroc(scores[idx][labels[idx]], scores[idx][~labels[idx]]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def within_pair_auc(scores, labels, pairs):
    scores, labels = np.asarray(scores), np.asarray(labels); vals, ws = [], []
    for p in set(pairs):
        idx = np.array([i for i, q in enumerate(pairs) if q == p])
        if labels[idx].any() and (~labels[idx]).any():
            vals.append(auroc(scores[idx][labels[idx]], scores[idx][~labels[idx]])); ws.append(len(idx))
    return float(np.average(vals, weights=ws)) if vals else np.nan, len(vals)


def main(n_splits=10):
    c0, red, neg = load_all()
    y0 = np.array([e["y_exec"] for e in c0]); cc = np.array([e["task"] for e in c0])
    has_vt = all(e["vt"] is not None for e in c0 + red + neg)
    print("n", len(c0), len(red), len(neg), "vt", has_vt, flush=True)
    viol = np.array([e["outcome"] != "follow" for e in red]); wrong = np.array([e["outcome"] in ("persist", "other") for e in red])
    pairs = [e["pair"] for e in red]; nexec = np.array([e["outcome"] == "forbidden" for e in neg])
    # pair prior: leave-one-out non-follow rate of the ordered pair (uses no observation)
    prior = np.array([np.mean([viol[j] for j in range(len(red)) if pairs[j] == pairs[i] and j != i]) for i in range(len(red))])
    names = ["goal-L32", "goal-L4"] + (["vision+text"] if has_vt else []) + ["entropy", "success-probe", "privileged-state"]
    R = {m: dict(red_auc=[], red_wrong=[], red_within=[], red_d0=[], red_d0_within=[], neg_auc=[], far_nom=[], far_fol=[], tpr_red=[],
                 far_negc=[], tpr_neg=[]) for m in names}
    keep = {}
    rng = np.random.RandomState(0)
    for sp in range(n_splits):
        tr, ca = [], []
        for k in range(10):
            idx = [i for i in range(len(c0)) if cc[i] == k]; rng.shuffle(idx); m = int(round(0.6 * len(idx)))
            tr += idx[:m]; ca += idx[m:]
        TR = [c0[i] for i in tr]; CA = [c0[i] for i in ca if y0[i] == cc[i]]
        pr = {}
        for L in GL:
            X, y, g, _ = stack_steps(TR, f"h{L}"); pr[f"goal-L{L}"] = (Probe().fit(X, y, ep_weights(g)), f"h{L}")
        X, y, g, _ = stack_steps(TR, "kin"); pr["privileged-state"] = (Probe(k=64, C=1.0).fit(X, y, ep_weights(g)), "kin")
        if has_vt:
            X, y, g, _ = stack_steps(TR, "vt"); pr["vision+text"] = (Probe(k=128, C=0.5).fit(X, y, ep_weights(g)), "vt")
        for e in TR:
            e["yb"] = e["y_exec"]; e["y_exec"] = int(not e["success"])
        X, y, g, _ = stack_steps(TR, "h32"); succ = Probe().fit(X, y, ep_weights(g))
        for e in TR:
            e["y_exec"] = e.pop("yb")
        grng = np.random.RandomState(100 + sp); gcal = [grng.choice([k for k in range(10) if k != e["task"]]) for e in CA]

        def per_step(e, m, allowed, spec_neg=None):
            if m == "entropy":
                return e["ent"]
            if m == "success-probe":
                lp = succ.logp(e["h32"]); s = lp[:, 1] - lp[:, 0]  # evidence of FAILURE of the instructed task
                return -s if spec_neg else s  # spec-aware: under 'do not X', success of X is the violation
            p, f = pr[m]; lp = p.logp(e[f]); A = np.zeros(NCLS, bool)
            out = np.empty(len(lp))
            for t in range(len(lp)):
                A[:] = False; A[list(allowed(t))] = True
                out[t] = lp[t, ~A].max() - lp[t, A].max()
            return out

        for m in names:
            cal = np.array([at_h(dstat(per_step(e, m, lambda t, e=e: {e["task"]}), 0), 0) for e in CA])
            hh = np.random.RandomState(7 + sp).permutation(len(cal)); h1, h2 = hh[: len(hh) // 2], hh[len(hh) // 2:]
            R[m]["far_nom"].append(float(np.mean(cal[h2] > conformal_tau(cal[h1], 0.1))))
            tau = conformal_tau(cal, 0.1)
            raw = [per_step(e, m, lambda t, e=e: {e["task"]} if t < e["switch_t"] else {e["task2"]}) for e in red]
            f = np.array([at_h(dstat(s, e["switch_t"]), e["switch_t"]) for s, e in zip(raw, red)])
            d0 = np.array([s[e["switch_t"]] for s, e in zip(raw, red)])
            R[m]["red_auc"].append(auroc(f[viol], f[~viol])); R[m]["red_wrong"].append(auroc(f[wrong], f[~viol]))
            R[m]["red_within"].append(within_pair_auc(f, viol, pairs)[0]); R[m]["red_d0"].append(auroc(d0[viol], d0[~viol]))
            R[m]["red_d0_within"].append(within_pair_auc(d0, viol, pairs)[0])
            R[m]["tpr_red"].append(float(np.mean(f[viol] > tau))); R[m]["far_fol"].append(float(np.mean(f[~viol] > tau)))
            rawn = [per_step(e, m, lambda t, e=e: set(range(NCLS)) - {e["task"]}, spec_neg=True) for e in neg]
            fn = np.array([at_h(dstat(s, 0), 0) for s in rawn])
            R[m]["neg_auc"].append(auroc(fn[nexec], fn[~nexec]))
            caln = np.array([at_h(dstat(per_step(e, m, lambda t, g=g: set(range(NCLS)) - {g}, spec_neg=True), 0), 0) for e, g in zip(CA, gcal)])
            taun = conformal_tau(caln, 0.1)
            R[m]["tpr_neg"].append(float(np.mean(fn[nexec] > taun))); R[m]["far_negc"].append(float(np.mean(fn[~nexec] > taun)))
            if sp == 0:
                keep[m] = dict(f=f, d0=d0, fn=fn)
        print("split", sp, {m: round(float(np.mean(R[m]["red_within"])), 3) for m in names}, flush=True)

    import pickle
    pickle.dump(dict(R=R, keep=keep, prior=prior, viol=viol, wrong=wrong, pairs=pairs, nexec=nexec,
                     tasks_neg=[e["task"] for e in neg]), open(os.path.join(OUT, "v3_raw.pkl"), "wb"))
    rate_pair = {p: np.mean([viol[j] for j in range(len(red)) if pairs[j] == p]) for p in set(pairs)}
    prior_tie = np.array([rate_pair[p] for p in pairs])  # ties within a pair (no LOO anti-ranking)
    cells = [(e["pair"], e["cond"]) for e in red]
    rate_cell = {c: np.mean([viol[j] for j in range(len(red)) if cells[j] == c]) for c in set(cells)}
    prior_time = np.array([rate_cell[c] for c in cells]); tr_ = np.array([e["switch_t"] for e in red], float)

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
        return lambda idx: within_pair_auc(np.asarray(sc)[idx], viol[idx], [(pairs[j], k) for k, j in enumerate(idx)] if False else [pairs[j] for j in idx])[0]
    extra = {"n_short_excluded": N_SHORT,
             "pair_prior_tie_auc": float(auroc(prior_tie[viol], prior_tie[~viol])),
             "pair_time_prior_auc": float(auroc(prior_time[viol], prior_time[~viol])),
             "pair_time_prior_within": within_pair_auc(tr_, viol, pairs)[0],
             "pair_time_prior_within_ci": boot(wp(tr_))}
    for m in names:
        extra[f"{m}_within_ci"] = boot(wp(keep[m]["f"]))
        sub = wrong | ~viol  # wrong-goal completions vs followed (stalls excluded)
        extra[f"{m}_wrong_ci"] = cluster_boot_auc(keep[m]["f"][sub], wrong[sub], [pairs[j] for j in np.flatnonzero(sub)])
    if "vision+text" in names:
        a, b = keep["goal-L32"]["f"], keep["vision+text"]["f"]
        extra["within_diff_L32_minus_vt"] = within_pair_auc(a, viol, pairs)[0] - within_pair_auc(b, viol, pairs)[0]
        extra["within_diff_ci"] = boot(lambda idx: wp(a)(idx) - wp(b)(idx))
    print("extra", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in extra.items()}, flush=True)
    out = {"extra": extra, "n": dict(nominal=len(c0), redirect=len(red), viol=int(viol.sum()), wrong=int(wrong.sum()), follow=int((~viol).sum()),
                     negation=len(neg), neg_exec=int(nexec.sum()), n_mixed_pairs=within_pair_auc(prior, viol, pairs)[1]),
           "pair_prior": dict(red_auc=float(auroc(prior[viol], prior[~viol])), red_within=0.5,
                              red_ci=cluster_boot_auc(prior, viol, pairs))}
    tasks_neg = [e["task"] for e in neg]
    for m in names:
        s = {k: float(np.nanmean(v)) for k, v in R[m].items()}
        s["red_ci"] = cluster_boot_auc(keep[m]["f"], viol, pairs)
        s["red_d0_ci"] = cluster_boot_auc(keep[m]["d0"], viol, pairs)
        s["neg_ci"] = cluster_boot_auc(keep[m]["fn"], nexec, tasks_neg)
        out[m] = s
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    json.dump(out, open(os.path.join(OUT, "v3.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
