"""Analysis utilities: episode loading, executed-goal labels, probes, conformal monitor."""
import glob, json, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "runs")
NONE = 10          # "no goal achieved" class
NCLS = 11
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
TASKS = ["open middle drawer", "bowl on stove", "wine on cabinet", "top drawer + bowl", "bowl on cabinet",
         "plate to stove front", "cheese in bowl", "turn on stove", "bowl on plate", "wine on rack"]


def load_meta(run):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, run, "meta_w*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return sorted(rows, key=lambda r: r["id"])


def exec_goal(G, hold=5):
    """Executed goal = goal whose predicate first becomes true and stays true for `hold` steps
    (or until the episode ends). Returns (label, time)."""
    T = len(G)
    best = (NONE, T)
    for j in range(G.shape[1]):
        col = G[:, j]
        for t in np.flatnonzero(col):
            if col[t:min(t + hold, T)].all():
                if t < best[1]:
                    best = (j, int(t))
                break
    return best


def load_run(run, keys=("H",), layer_idx=None):
    """Returns list of episode dicts with meta + requested arrays (+ y_exec, t_exec)."""
    eps = []
    for m in load_meta(run):
        d = np.load(os.path.join(ROOT, run, f"ep{m['id']:05d}.npz"))
        e = dict(m)
        G = d["goals"]
        e["y_exec"], e["t_exec"] = exec_goal(G)
        e["T"] = len(G)
        for k in keys:
            if k == "H" and layer_idx is not None:
                e["H"] = d["H"][:, layer_idx].astype(np.float32)
            elif k in d.files:
                e[k] = d[k]
        eps.append(e)
    return eps


def stack_steps(eps, feat="H", stride=1, tmax=None):
    X, y, g, t = [], [], [], []
    for i, e in enumerate(eps):
        F = e[feat]
        idx = np.arange(0, len(F) if tmax is None else min(len(F), tmax), stride)
        X.append(F[idx]); y.append(np.full(len(idx), e["y_exec"])); g.append(np.full(len(idx), i)); t.append(idx)
    return np.concatenate(X), np.concatenate(y), np.concatenate(g), np.concatenate(t)


class Probe:
    """Standardize -> PCA(k) -> multinomial logistic regression over NCLS classes."""

    def __init__(self, k=256, C=0.05):
        self.k, self.C = k, C

    def fit(self, X, y, w=None):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.pca = PCA(min(self.k, Z.shape[1], Z.shape[0] - 1), random_state=0).fit(Z)
        Z = self.pca.transform(Z)
        self.lr = LogisticRegression(C=self.C, max_iter=500).fit(Z, y, sample_weight=w)
        return self

    def logp(self, X):
        lp = np.full((len(X), NCLS), -30.0)
        lp[:, self.lr.classes_] = self.lr.predict_log_proba(self.pca.transform(self.sc.transform(X)))
        return lp


def ep_weights(g):
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def crossfit_logp(eps, feat="H", folds=5, seed=0, train_extra=None, **pk):
    """Out-of-fold per-step log-probs for every episode (grouped by episode)."""
    rng = np.random.RandomState(seed)
    fold = rng.permutation(len(eps)) % folds
    out = [None] * len(eps)
    for f in range(folds):
        tr = [e for e, fo in zip(eps, fold) if fo != f] + (train_extra or [])
        X, y, g, _ = stack_steps(tr, feat)
        pr = Probe(**pk).fit(X, y, ep_weights(g))
        for i in np.flatnonzero(fold == f):
            out[i] = pr.logp(eps[i][feat])
    return out


# ----------------------------------------------------------------------------- conformal monitor
def mismatch_score(logp, instr):
    """s_t = max_{j != i} log p_j - log p_i ; returns running max S_t (time-uniform statistic)."""
    other = np.delete(logp, instr, axis=1).max(1)
    s = other - logp[:, instr]
    return np.maximum.accumulate(s)


def conformal_tau(cal_scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest calibration score."""
    s = np.sort(np.asarray(cal_scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def first_alarm(S_run, tau):
    idx = np.flatnonzero(S_run > tau)
    return int(idx[0]) if len(idx) else -1


def entropy_score(logits):
    """Mean over the 7 action dims of the entropy of the action-bin distribution; running max."""
    z = logits.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z); p /= p.sum(-1, keepdims=True)
    H = -(p * np.log(p + 1e-12)).sum(-1).mean(-1)
    return np.maximum.accumulate(H)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bootstrap_ci(x, fn=np.mean, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.RandomState(seed)
    vals = [fn(x[rng.randint(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])
