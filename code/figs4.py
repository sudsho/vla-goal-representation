"""Figures 1 and 2 of the paper
fig_read.pdf:   (a) goal localization by patching   (b) negation and the goal read-out
fig_write.pdf:  (a) per-dimension effect of edits    (b) detection of redirect non-compliance"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(B, "data", "results")
OUT = os.path.join(B, "paper", "figs")
LQ, RQ = chr(0x201C), chr(0x201D)  # typographic quotes in legends
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "lines.linewidth": 1.6, "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42})
num = {}
for l in open(os.path.join(B, "paper", "numbers.tex")):
    num[l.split("{\\n")[1].split("}{")[0]] = l.split("}{")[1].rstrip("}\n")


def grid(ax):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6); ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)


# Figure 1: reading the goal
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.46), gridspec_kw=dict(wspace=0.32))
# layer 32 is omitted in both panels: the model returns the post-norm final state as hidden_states[32],
# while rollouts recorded the pre-norm block-32 output that the probes were trained on
P = json.load(open(os.path.join(R, "patching3.json"))); L = [1] + list(range(2, 31, 2)); ax = axs[0]
for g, col, lab, mk in (("instr", C1, "instruction tokens", "o"), ("dec", C2, "decision token", "s"), ("tmpl", C3, "template tokens", "^")):
    Mx = np.array([[r[f"{g}_L{l}"] for l in L] for r in P])
    ax.fill_between(L, np.percentile(Mx, 25, 0), np.percentile(Mx, 75, 0), color=col, alpha=0.15, linewidth=0)
    ax.plot(L, np.median(Mx, 0), color=col, label=lab, marker=mk, markersize=2.8)
ax.set_xlabel("layer patched"); ax.set_ylabel(r"logit difference $\rho$")
ax.set_title("(a) Goal localization by patching", loc="left"); ax.set_ylim(-0.08, 1.1); ax.set_xticks([0, 8, 16, 24, 30]); grid(ax)
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.04, 0.5), handlelength=1.5)

NC = json.load(open(os.path.join(R, "negctrl.json")))["per_layer"]; LL = [4, 8, 12, 16, 20, 24, 28]; ax = axs[1]
ax.plot(LL, [NC[f"L{l}"]["neg_vs_insert_bal_acc"] for l in LL], color=C3, marker="^", markersize=3, label=LQ + 'not' + RQ + ' detector')
for p, col, mk, ls in (("X", C1, "o", "-"), ("not X", C2, "s", "-"), ("please X", C4, "D", "--"), ("Y", C5, "v", ":")):
    ax.plot(LL, [NC[f"L{l}"]["p_goal"][p] for l in LL], color=col, marker=mk, markersize=2.8, linestyle=ls,
            label='$p(X)$ given ' + LQ + p + RQ + (" (other task)" if p == "Y" else ""))
ax.set_xlabel("layer"); ax.set_ylabel("probability or accuracy")
ax.set_title("(b) Negation and the goal read-out", loc="left"); ax.set_ylim(-0.03, 1.05); ax.set_xticks([4, 8, 16, 24, 28]); grid(ax)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(-0.01, 0.38), labelspacing=0.22, handlelength=1.6)
save(fig, "fig_read")

# Figure 2: writing the goal and monitoring it
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.56), gridspec_kw=dict(wspace=0.3))
PD = json.load(open(os.path.join(R, "perdim_final.json"))); ax = axs[0]; x = np.arange(6)
for k, col, mk, ls, lab in (("span8", C3, "^", "-", f"instruction-span vectors, L8 ({num['SVtgt']}/{num['SVn']})"),
                            ("stream_dec24", C1, "o", "-", f"decode-stream patch, L24 ({num['STftgt']}/{num['STfn']})"),
                            ("pre_dec16", C2, "s", "-", f"prefill patch, L16 ({num['PCdtgt']}/{num['PCdn']})"),
                            ("posvec_b2", C5, "v", "--", f"position-wise vectors, L16 ({num['PVtgt']}/{num['PVn']})"),
                            ("steer_class", C4, "D", "--", f"class-mean vectors, L16 ({num['cmLtgt']}/{num['cmLn']})")):
    ax.plot(x, PD[k], color=col, marker=mk, markersize=3, linestyle=ls, label=lab)
ax.set_xticks(x); ax.set_xticklabels(["x", "y", "z", "roll", "pitch", "yaw"])
ax.set_xlabel("action dimension (decoding order)"); ax.set_ylabel("transfer to other goal")
ax.set_title("(a) Per-dimension effect of edits", loc="left"); ax.set_ylim(-0.1, 2.3); ax.set_yticks([0, 0.5, 1]); grid(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)

V = json.load(open(os.path.join(R, "v3.json"))); ax = axs[1]
ms = [("goal-L32", "goal\nL32"), ("vision+text", "vision\n+text"), ("privileged-state", "object\nposes"),
      ("success-probe", "success\nprobe"), ("goal-L4", "goal\nL4"), ("entropy", "action\nentropy")]
xx = np.arange(len(ms)); w = 0.38
ov = [V[m]["red_auc"] for m, _ in ms]; lo = [V[m]["red_auc"] - V[m]["red_ci"][0] for m, _ in ms]; hi = [V[m]["red_ci"][1] - V[m]["red_auc"] for m, _ in ms]
wi = [V[m]["red_within"] for m, _ in ms]
ax.bar(xx - w / 2 - 0.01, ov, w, color=C1, label="pooled", yerr=[lo, hi], error_kw=dict(lw=0.7, capsize=1.5, ecolor=MUTED))
wlo = [V[m]["red_within"] - V["extra"][m + "_within_ci"][0] for m, _ in ms]; whi = [V["extra"][m + "_within_ci"][1] - V[m]["red_within"] for m, _ in ms]
ax.bar(xx + w / 2 + 0.01, wi, w, color=C2, label="within pair", yerr=[wlo, whi], error_kw=dict(lw=0.7, capsize=1.5, ecolor=MUTED))
X = V["extra"]
ax.axhline(X["pair_time_prior_auc"], color=C1, linestyle="--", linewidth=0.9, label="pair+time")
ax.axhline(X["pair_time_prior_within"], color=C2, linestyle="--", linewidth=0.9, label="pair+time, within")
ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=0.8)
ax.set_xticks(xx); ax.set_xticklabels([n for _, n in ms], fontsize=6.8)
ax.set_ylabel("non-compliance AUROC"); ax.set_ylim(0.4, 1.27); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_title("(b) Detection of non-compliance", loc="left"); grid(ax)
ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6, bbox_to_anchor=(0.02, 1.05), columnspacing=1.0, labelspacing=0.2, handletextpad=0.5)
save(fig, "fig_write")
print("saved")
