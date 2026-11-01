"""Numbers and Table 1 of the paper -> paper/numbers.tex, paper/tab_monitor.tex.
Reads the episode summaries (data/episodes) or the raw rollouts (data/runs), the analysis outputs in
data/results, and the paired within-pair AUROC differences from code/paired_within.py. Rates use Wilson
intervals, and the matched negation comparison uses a Newcombe interval."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from alib import exec_goal, NONE

B = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(B, "data", "results")


def meta_goal(run):
    """Rollout metadata with the executed goal (y) and its time (ty). Uses the raw rollouts in data/runs/<run>
    when present and the compact summary data/episodes/<run>.jsonl (code/export_episodes.py) otherwise."""
    if os.environ.get("USE_EPISODE_SUMMARY") or not os.path.isdir(os.path.join(B, "data", "runs", run)):
        rows = [json.loads(l) for l in open(os.path.join(B, "data", "episodes", f"{run}.jsonl"))]
        for r in rows:
            r["y"], r["ty"] = r["y_exec"], r["t_exec"]
        return rows
    rows = []
    for f in glob.glob(os.path.join(B, "data", "runs", run, "meta_w*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            r["y"], r["ty"] = exec_goal(np.load(os.path.join(B, "data", "runs", run, f"ep{r['id']:05d}.npz"))["goals"])
            rows.append(r)
    return rows


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{max(0.0, 100 * (c - h)):.0f}, {min(100.0, 100 * (c + h)):.0f}"


pct = lambda a, b: f"{100 * a / max(b, 1):.0f}"
f2 = lambda x: f"{x:.2f}"
M = {}
c0 = meta_goal("c0"); n = len(c0); s = sum(r["y"] == r["task"] for r in c0)
M.update(nC=n, succC=pct(s, n), succCci=wilson(s, n), stallC=sum(r["y"] == NONE for r in c0), wrongC=sum(r["y"] not in (r["task"], NONE) for r in c0))
rate = {k: np.mean([r["y"] == k for r in c0 if r["task"] == k]) for k in range(10)}
neg = meta_goal("negation") + meta_goal("negation2"); nx = sum(r["y"] == r["task"] for r in neg)
M.update(nNG=len(neg), forbNG=nx, forbNGpct=pct(nx, len(neg)), forbNGci=wilson(nx, len(neg)),
         forbExp=f"{sum(rate[r['task']] for r in neg):.0f}", stallNG=sum(r["y"] == NONE for r in neg),
         otherNG=sum(r["y"] not in (r["task"], NONE) for r in neg))
for i, cond in enumerate(["neg0", "neg1", "neg2", "neg3", "neg4"]):
    rr = [r for r in neg if r["cond"] == cond]; M[f"negT{i}"] = f"{sum(r['y'] == r['task'] for r in rr)}/{len(rr)}"
red = [r for r in meta_goal("switch") + meta_goal("switch_late") if not (r["y"] == r["task"] and r["ty"] < r["switch_t"])]
o = lambda r: "follow" if r["y"] == r["task2"] else "persist" if r["y"] == r["task"] else "stall" if r["y"] == NONE else "other"
for k in ("follow", "persist", "stall", "other"):
    M[f"R{k}"] = sum(o(r) == k for r in red)
M.update(nR=len(red), Rfollowpct=pct(M["Rfollow"], len(red)), Rstallpct=pct(M["Rstall"], len(red)), Rwrong=M["Rpersist"] + M["Rother"])
pf = {}
for r in red:
    pf.setdefault((r["task"], r["task2"]), []).append(o(r) == "follow")
M.update(RpairMax=f"{sum(pf[(8, 4)])}/{len(pf[(8, 4)])}", RpairMin=f"{sum(pf[(0, 3)])}/{len(pf[(0, 3)])}")
un = meta_goal("under"); M.update(nTR=len(un), noneTR=sum(r["y"] == NONE for r in un))
nv = meta_goal("novel")
isnov = lambda r: r.get("first_hit_x") and r["first_hit_x"][0] >= 0 and (r["y"] == NONE or r["first_hit_x"][0] <= r["ty"])
M.update(nNV=len(nv), novelNV=sum(bool(isnov(r)) for r in nv), trainedNV=sum(r["y"] != NONE and not isnov(r) for r in nv))
st = meta_goal("steer_pilot") + meta_goal("steer2_pilot")
M.update(nST=len(st), switchST=sum(r["y"] == r["task2"] for r in st), origST=pct(sum(r["y"] == r["task"] for r in st), len(st)))
# class-mean runs by layer: steer_pilot = layer 12 (decision-token class mean), steer2_pilot = layers 16 and 20 (per-position)
cm16 = [r for r in st if (r.get("steer") or {}).get("layer") == 16]
M.update(cmLn=len(cm16), cmLtgt=sum(r["y"] == r["task2"] for r in cm16),
         cmTwelven=sum((r.get("steer") or {}).get("layer") == 12 for r in st),
         cmLaten=sum((r.get("steer") or {}).get("layer") in (16, 20) for r in st))
USE_EXT = False  # optional extension run (data/runs/ext); not used in the paper
EXT = meta_goal("ext") if USE_EXT and os.path.isdir(os.path.join(B, "data", "runs", "ext")) else []
ext = lambda cond: [r for r in EXT if r["cond"] == cond]
pc = meta_goal("patch_cl") + ext("pt_instr_L8")
for g, key in (("pt_instr_L8", "PCi"), ("pt_dec_L16", "PCd"), ("pt_text_L16", "PCt")):
    rr = [r for r in pc if r["cond"] == g]
    M[f"{key}n"] = len(rr); M[f"{key}tgt"] = sum(r["y"] == r["task2"] for r in rr); M[f"{key}stall"] = sum(r["y"] == NONE for r in rr)
M["nWrite"] = M["PCdn"] + M["PCtn"] + M["nST"]  # prefill patches after the handoff + class-mean steering (cf-steer added below)
M["PCiNomTgt"] =pct(np.mean([rate[r["task2"]] for r in pc if r["cond"] == "pt_instr_L8"]) * 100, 100)
cf = meta_goal("cfsteer")
for b in ("2", "3"):
    rr = [r for r in cf if r["cond"] == f"cf_L16_b{b}"]
    M[f"CF{b}n"] = len(rr); M[f"CF{b}tgt"] = sum(r["y"] == r["task2"] for r in rr); M[f"CF{b}orig"] = sum(r["y"] == r["task"] for r in rr)
    M[f"CF{b}stall"] = sum(r["y"] == NONE for r in rr)
st3 = meta_goal("steer3") + ext("span_L8_b1")
for cond, key in (("posvec_L16_b2", "PV"), ("span_L8_b1", "SV")):
    rr = [r for r in st3 if r["cond"] == cond]
    M[f"{key}n"] = len(rr); M[f"{key}tgt"] = sum(r["y"] == r["task2"] for r in rr); M[f"{key}stall"] = sum(r["y"] == NONE for r in rr)
sm = meta_goal("stream16") + ext("stream_L16")
M.update(STn=len(sm), STtgt=sum(r["y"] == r["task2"] for r in sm), STstall=sum(r["y"] == NONE for r in sm))
sm24 = meta_goal("stream24") + ext("stream_L24")
M.update(STfn=len(sm24), STftgt=sum(r["y"] == r["task2"] for r in sm24))
for key in ("SV", "STf", "ST", "PCi"):
    M[f"{key}pct"] = pct(M[f"{key}tgt"], M[f"{key}n"]); M[f"{key}ci"] = wilson(M[f"{key}tgt"], M[f"{key}n"])
M["zeroUp"] = wilson(0, M["PVn"]).split(", ")[1]  # upper Wilson bound (%) for a 0/18 count
M["clInits"] = "40 to 49" if all(len(ext(c)) == 42 for c in ("span_L8_b1", "stream_L24", "stream_L16", "pt_instr_L8")) else "45, 46 and 47"
print("ext rows", {c: len(ext(c)) for c in ("span_L8_b1", "stream_L24", "stream_L16", "pt_instr_L8")})
M["nWrite"] += M["CF2n"] + M["CF3n"] + M["PVn"]
nomtwin = {(r["task"], r["init"]): r["y"] == r["task"] for r in c0}
tw = [(r["y"] == r["task"], nomtwin[(r["task"], r["init"])]) for r in neg if (r["task"], r["init"]) in nomtwin]
from scipy.stats import binomtest
b_ = sum(1 for a, b in tw if a and not b); c_ = sum(1 for a, b in tw if b and not a)
M.update(twN=len(tw), twNeg=sum(a for a, _ in tw), twNom=sum(b for _, b in tw),
         twP=f"{binomtest(min(b_, c_), b_ + c_, 0.5).pvalue:.2f}" if b_ + c_ else "1.00")


def newcombe_paired(a, b, c, d, z=1.96):
    """Newcombe (1998) method 10 CI for p1 - p2 with paired data; p1 = (a+b)/n, p2 = (a+c)/n."""
    n = a + b + c + d; p1, p2 = (a + b) / n, (a + c) / n

    def wl(k):
        p = k / n; dd = 1 + z * z / n; cc = (p + z * z / (2 * n)) / dd; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
        return cc - h, cc + h
    (l1, u1), (l2, u2) = wl(a + b), wl(a + c)
    num, den = a * d - b * c, np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = 0.0 if den == 0 else (max(num - n / 2, 0) if num > 0 else num) / den
    th = p1 - p2
    lo = th - np.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = th + np.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return th, lo, hi


a_ = sum(1 for x, y in tw if x and y); d_ = sum(1 for x, y in tw if not x and not y)
th_, lo_, hi_ = newcombe_paired(a_, b_, c_, d_)
M.update(twDiff=f"{100 * th_:+.0f}", twDiffCi=f"{100 * lo_:+.0f}, {100 * hi_:+.0f}")
print("newcombe", a_, b_, c_, d_, round(100 * th_, 1), round(100 * lo_, 1), round(100 * hi_, 1))

P = json.load(open(os.path.join(R, "patching3.json"))); M["nPatch"] = len(P)
med = lambda g, l: np.median([r[f"{g}_L{l}"] for r in P])
for l in (8, 12, 16):
    M[f"pI{l}"] = f2(med("instr", l)); M[f"pD{l}"] = f2(med("dec", l))
M["pTmax"] = f2(max(med("tmpl", l) for l in [1] + list(range(2, 33, 2))))


def perdim(rows, c):
    per = [[] for _ in range(6)]
    for r in rows:
        ac, ad, ap = np.array(r["a_c"]), np.array(r["a_d"]), np.array(r[c])
        for j in range(6):
            g = ad[j] - ac[j]
            if abs(g) > 0.1:
                per[j].append(np.clip((ap[j] - ac[j]) / g, -1, 2))
    return [float(np.median(p)) for p in per]


P4 = json.load(open(os.path.join(R, "patching4.json"))); P5 = json.load(open(os.path.join(R, "patching5.json")))
PD = {"pre_dec16": perdim(P4, "patch_dec_L16"), "instr8": perdim(P4, "patch_instr_L8"), "steer_class": perdim(P4, "steer_post_L16"),
      "stream_dec16": perdim(P5, "stream_dec_L16"), "stream_dec24": perdim(P5, "stream_dec_L24"), "cf_b2": perdim(P5, "cfsteer_L16_b2")}
json.dump(PD, open(os.path.join(R, "perdim_final.json"), "w"))
M.update(nPfour=len(P4), preDecFirst=f2(PD["pre_dec16"][0]), preDecRest=f2(max(PD["pre_dec16"][1:])),
         streamSixMin=f2(min(PD["stream_dec16"])), streamTwentyfourMin=f2(min(PD["stream_dec24"])), instrMin=f2(min(PD["instr8"])),
         steerMax=f2(max(PD["steer_class"])), cfFirst=f2(PD["cf_b2"][0]), cfRestMax=f2(max(PD["cf_b2"][1:])))

NC = json.load(open(os.path.join(R, "negctrl.json")))
pl = NC["per_layer"]
M.update(ncDetEarly=f2(pl["L4"]["neg_vs_insert_bal_acc"]), ncDetMid=f2(pl["L16"]["neg_vs_insert_bal_acc"]),
         ncDetLate=f2(pl["L28"]["neg_vs_insert_bal_acc"]),
         ncX=f2(pl["L16"]["p_goal"]["X"]), ncNot=f2(pl["L16"]["p_goal"]["not X"]), ncPlease=f2(pl["L16"]["p_goal"]["please X"]),
         ncNow=f2(pl["L16"]["p_goal"]["now X"]), ncFrames=NC["n_frames"],
         ftNot=pct(NC["first_token_agree_with_X"]["not X"] * 100, 100), ftPlease=pct(NC["first_token_agree_with_X"]["please X"] * 100, 100),
         ncYmax=f2(max(v["p_goal"]["Y"] for v in pl.values())), ncNotMin=f2(min(v["p_goal"]["not X"] for k, v in pl.items() if k != "L32")),
         ncAllMin=f2(min(v["p_goal"][p] for k, v in pl.items() if k != "L32" for p in ("X", "not X", "please X"))))
P6 = json.load(open(os.path.join(R, "patching6.json")))
PD.update(posvec_b2=perdim(P6, "posvec_L16_b2"), span8=perdim(P6, "instrvec_L8"))
json.dump(PD, open(os.path.join(R, "perdim_final.json"), "w"))
M.update(pvMin=f2(min(PD["posvec_b2"])), pvMax=f2(max(PD["posvec_b2"])), svMin=f2(min(PD["span8"])))

V = json.load(open(os.path.join(R, "v3.json")))
g = V["goal-L32"]
M.update(vViol=V["n"]["viol"], vFol=V["n"]["follow"], vWrong=V["n"]["wrong"], vPairs=V["n"]["n_mixed_pairs"],
         prior=f2(V["pair_prior"]["red_auc"]), gAuc=f2(g["red_auc"]), gWithin=f2(g["red_within"]), gWrong=f2(g["red_wrong"]),
         gTpr=f2(g["tpr_red"]), gFarFol=f2(g["far_fol"]), gFarNom=f2(g["far_nom"]), gNegTpr=f2(g["tpr_neg"]),
         vtWithin=f2(V["vision+text"]["red_within"]), privWithin=f2(V["privileged-state"]["red_within"]),
         entWithin=f2(V["entropy"]["red_within"]), spWithin=f2(V["success-probe"]["red_within"]), sWithin=f2(V["goal-L4"]["red_within"]),
         entNegTpr=f2(V["entropy"]["tpr_neg"]), spNegTprRaw="--")
X = V["extra"]
M.update(nShort=X["n_short_excluded"], priorTie=f2(X["pair_prior_tie_auc"]), priorTime=f2(X["pair_time_prior_auc"]),
         priorTimeWithin=f2(X["pair_time_prior_within"]), priorTimeWithinCi=f"{X['pair_time_prior_within_ci'][0]:.2f}, {X['pair_time_prior_within_ci'][1]:.2f}",
         gWithinCi=f"{X['goal-L32_within_ci'][0]:.2f}, {X['goal-L32_within_ci'][1]:.2f}",
         dWithin=f"{X['within_diff_L32_minus_vt']:+.2f}", dWithinCi=f"{X['within_diff_ci'][0]:+.2f}, {X['within_diff_ci'][1]:+.2f}",
         gNegFa=f2(g["far_negc"]), entWrong=f2(V["entropy"]["red_wrong"]), spWrong=f2(V["success-probe"]["red_wrong"]),
         vtWrong=f2(V["vision+text"]["red_wrong"]), privWrong=f2(V["privileged-state"]["red_wrong"]))
# paired within-pair differences of the layer-32 read-out (split-0 scores, pair-cluster bootstrap, 10,000 resamples)
PW = json.load(open(os.path.join(R, "paired_within.json")))
for nm, key in (("time", "dTime"), ("sp", "dSp"), ("ent", "dEnt"), ("vt", "dWithin")):
    lo, hi = PW[f"L32_minus_{nm}_ci_10k"]
    M[key] = f"{PW[f'L32_minus_{nm}']:+.2f}"; M[key + "Ci"] = f"{lo:+.2f}, {hi:+.2f}"


def roman(mo):
    k, out = int(mo.group()), ""
    for v, r in ((50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while k >= v:
            out += r; k -= v
    return out


with open(os.path.join(B, "paper", "numbers.tex"), "w") as f:
    for k, v in M.items():
        f.write("\\newcommand{\\n%s}{%s}\n" % (re.sub(r"\d+", roman, k), v))

BS = chr(92)
ci = lambda x: "{" + BS + "scriptsize[%.2f,%.2f]}" % (x[0], x[1])
EOL = " " + BS + BS
rows = [("Pair+time prior", "PT"), ("Goal probe, layer 32", "goal-L32"), ("Vision+text (DINOv2)", "vision+text"),
        ("Object poses$^" + BS + "ast$ (linear)", "privileged-state"), ("Success probe", "success-probe"), ("Goal probe, layer 4", "goal-L4"), ("Action entropy", "entropy")]
t = [BS + "begin{tabular}{lccccccc}", BS + "toprule",
     " & " + BS + "multicolumn{5}{c}{Redirect not followed ($n{=}%d$) vs." % V["n"]["viol"] + BS + " followed ($n{=}%d$)} & " % V["n"]["follow"] + BS + "multicolumn{2}{c}{Negation}" + EOL,
     BS + "cmidrule(lr){2-6}" + BS + "cmidrule(lr){7-8}",
     "Monitor & AUROC & within-pair AUROC & wrong-goal AUROC & TPR & FA & TPR & FA" + EOL,
     " & [95" + BS + "% CI] & [95" + BS + "%% CI] & ($n{=}%d$) & & & ($n{=}%d$) & ($n{=}%d$)" % (V["n"]["wrong"], V["n"]["neg_exec"], V["n"]["negation"] - V["n"]["neg_exec"]) + EOL,
     BS + "midrule"]
for name, m in rows:
    if m == "PP":
        t.append("%s & %.2f & 0.50 & -- & -- & -- & -- & --" % (name, X["pair_prior_tie_auc"]) + EOL); continue
    if m == "PT":
        t.append("%s & %.2f & %.2f %s & -- & -- & -- & -- & --" % (name, X["pair_time_prior_auc"], X["pair_time_prior_within"], ci(X["pair_time_prior_within_ci"])) + EOL)
        t.append(BS + "midrule"); continue
    x = V[m]
    t.append("%s & %.2f %s & %.2f %s & %.2f & %.2f & %.2f & %.2f & %.2f" % (name, x["red_auc"], ci(x["red_ci"]), x["red_within"], ci(X[m + "_within_ci"]),
                                                                        x["red_wrong"], x["tpr_red"], x["far_fol"], x["tpr_neg"], x["far_negc"]) + EOL)
t += [BS + "bottomrule", BS + "end{tabular}"]
open(os.path.join(B, "paper", "tab_monitor.tex"), "w").write("\n".join(t))
print(json.dumps(M)[:3000])
