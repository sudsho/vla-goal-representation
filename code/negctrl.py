"""Negation vs non-negating insertions on identical frames: prompts 'X', 'not X', 'please X', 'now X'.
Per layer: (i) a linear detector 'not X' vs {'please X','now X'} (grouped CV by episode) - is negation encoded
as negation rather than as 'an extra word'? (ii) nominal goal-probe p(X) under each prompt (out-of-sample probe:
trained on nominal rollouts of OTHER episodes than the frames' episodes). (iii) first-action-token agreement with 'X'.
Writes data/results/negctrl.json"""
import json, os, sys
import numpy as np
os.environ.setdefault("USE_FLAX", "0")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image
from rollout import OpenVLAPolicy, center_crop
from make_plan import TASKS
from alib import load_run, load_meta, ROOT, LAYERS, Probe, stack_steps, ep_weights
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROMPTS = {"X": "{}", "not X": "not {}", "please X": "please {}", "now X": "now {}", "Y": None}  # Y: another task's instruction


def main(n_eps=4, steps="5,20,40"):
    n_eps, steps = int(n_eps), [int(s) for s in steps.split(",")]
    pol = OpenVLAPolicy("data/ckpt/openvla-7b-finetuned-libero-goal", "libero_goal", [32])
    vla = pol.vla
    meta = [m for m in load_meta("c0") if m["success"]]
    used = set()
    feats = {p: [] for p in PROMPTS}; first = {p: [] for p in PROMPTS}; lab = []; grp = []
    for k in range(10):
        for m in [m for m in meta if m["task"] == k][:n_eps]:
            used.add(m["id"])
            imgs = np.load(os.path.join(ROOT, "c0", f"img{m['id']:05d}.npz"))["img"]
            for t in steps:
                if t >= len(imgs):
                    continue
                pil = Image.fromarray(center_crop(imgs[t])).convert("RGB")
                for p, tmpl in PROMPTS.items():
                    inp, ids = pol._encode(tmpl.format(TASKS[k]) if tmpl else TASKS[(k + 5) % 10], pil)
                    with torch.no_grad():
                        out = vla(input_ids=ids, pixel_values=inp["pixel_values"], output_hidden_states=True)
                    feats[p].append(np.stack([out.hidden_states[L][0, -1].float().cpu().numpy() for L in LAYERS]))
                    first[p].append(int(out.logits[0, -1, -256:].argmax()))
                lab.append(k); grp.append(m["id"])
    lab, grp = np.array(lab), np.array(grp)
    F = {p: np.stack(v) for p, v in feats.items()}  # N x 8 x 4096
    res = {"n_frames": len(lab), "first_token_agree_with_X": {p: float(np.mean(np.array(first[p]) == np.array(first["X"]))) for p in PROMPTS}}
    c0 = [e for e in load_run("c0", keys=("H",)) if e["id"] not in used]  # probe never sees these episodes
    per = {}
    for li, L in enumerate(LAYERS):
        # (i) negation vs other insertions
        X = np.concatenate([F["not X"][:, li], F["please X"][:, li], F["now X"][:, li]])
        y = np.concatenate([np.ones(len(lab)), np.zeros(2 * len(lab))]); g = np.concatenate([grp, grp, grp])
        ug = np.unique(grp); fold = {u: i % 5 for i, u in enumerate(np.random.RandomState(0).permutation(ug))}
        fo = np.array([fold[u] for u in g]); accs = []
        for f in range(5):
            tr, te = fo != f, fo == f
            sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(C=0.01, max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            pr = clf.predict(sc.transform(X[te])); accs.append(0.5 * (np.mean(pr[y[te] == 1] == 1) + np.mean(pr[y[te] == 0] == 0)))
        # (ii) goal read-out, out-of-sample
        for e in c0:
            e["h"] = e["H"][:, li].astype(np.float32)
        Xg, yg, gg, _ = stack_steps(c0, "h")
        pb = Probe().fit(Xg, yg, ep_weights(gg))
        pX = {p: float(np.mean(np.exp(pb.logp(F[p][:, li].astype(np.float32))[np.arange(len(lab)), lab]))) for p in PROMPTS}
        per[f"L{L}"] = dict(neg_vs_insert_bal_acc=float(np.mean(accs)), p_goal=pX)
        print(L, per[f"L{L}"], flush=True)
    res["per_layer"] = per
    print(res["first_token_agree_with_X"])
    json.dump(res, open("data/results/negctrl.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
