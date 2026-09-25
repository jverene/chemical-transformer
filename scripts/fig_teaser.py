"""Teaser figure (Figure 1): the paper's three findings in one row.

(a) Endogenous allocation signals fail (testbed gates by difficulty bin).
    Entropy gate and supervised (tag) means are the realized per-bin values
    quoted in Sec. 4.1 / Table 1 (entropy 0.232/0.233/0.233; tag mean 0.51,
    spread +0.21). Tag-init is shown as a schematic of the measured
    inversion (per-problem gate-difficulty r = -0.59); unsupervised is flat
    with a seed-instability band (vacuity). Legend sits BELOW the axes so
    it never collides with the curves.
(b) The value field's life cycle: ordering value (random - waterfill
    held-out loss, zero-shot on frozen checkpoint bodies) from
    results-headroom/trajectory_seed{0,1}.json.
(c) The causal test: chasing the measured field online loses to random
    allocation of the same budget at both scales tested; a constant gate
    collapses. Bars from the causal (11M) and starved-150M tables. Legend
    in the free upper-left zone (bars there top out at 51.6; the dense
    reference line runs above it).

Writes paper/teaser.pdf. Run from repo root:
  .venv/bin/python scripts/fig_teaser.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_GRAY = "#999999"
plt.rcParams.update({
    "font.size": 6.8, "axes.titlesize": 7.4, "axes.labelsize": 6.6,
    "legend.fontsize": 5.9, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200})


def panel_a(ax):
    bins = np.array([0, 1, 2])
    labels = ["E", "M", "H"]
    # realized gates, Table 1 / Sec 4.1: entropy 0.232/0.233/0.233 (collapse);
    # tag targets: mean 0.51, E/M/H spread +0.21 (ordered).
    ax.plot(bins, [0.232, 0.233, 0.233], "o-", color=C_GRAY, lw=1.4, ms=3.4,
            label="entropy gate (collapses)")
    tag = [0.51 - 0.21 / 2, 0.51, 0.51 + 0.21 / 2]
    ax.plot(bins, tag, "o-", color=C_GREEN, lw=1.8, ms=3.4,
            label="tag-supervised (orders)")
    # inversion, schematic of r = -0.59 (tag-init, end-to-end)
    ax.plot(bins, [0.56, 0.50, 0.44], "v--", color=C_VERM, lw=1.4, ms=3.4,
            alpha=0.9, label="tag-init end-to-end (inverts)")
    # vacuity: flat, seed-unstable
    ax.fill_between(bins, 0.42, 0.58, color=C_BLUE, alpha=0.15, lw=0)
    ax.plot(bins, [0.50] * 3, "s:", color=C_BLUE, lw=1.3, ms=3.4,
            label="unsupervised head (vacuous)")
    # direct labels at the right edge, colored to match; no legend block
    ax.annotate("tag (orders)", (2.06, 0.615), fontsize=5.6, color=C_GREEN,
                va="center")
    ax.annotate("tag-init (inverts)", (2.06, 0.44), fontsize=5.6,
                color=C_VERM, va="center")
    ax.annotate("vacuous", (2.06, 0.50), fontsize=5.6, color=C_BLUE,
                va="center")
    ax.annotate("entropy", (2.06, 0.233), fontsize=5.6, color="0.4",
                va="center")
    ax.set_xticks(bins, labels)
    ax.set_xlabel("difficulty bin")
    ax.set_ylabel("mean FFN gate")
    ax.set_ylim(0.15, 0.72)
    ax.set_xlim(-0.15, 2.95)
    ax.set_title("(a) Endogenous signals fail", loc="left")


def panel_b(ax):
    curves = []
    for s in (0, 1):
        d = json.load(open(f"results-headroom/trajectory_seed{s}.json"))
        tr = d["trajectory"]
        curves.append([t["random"] - t["waterfill"] for t in tr])
    steps = [t["step"] for t in
             json.load(open("results-headroom/trajectory_seed0.json"))["trajectory"]]
    mean = [(a + b) / 2 for a, b in zip(*curves)]
    lo = [min(a, b) for a, b in zip(*curves)]
    hi = [max(a, b) for a, b in zip(*curves)]
    # one line only: bold seed mean, with the seed spread as a soft envelope
    ax.fill_between(steps, lo, hi, color="0.6", alpha=0.22, lw=0, zorder=2)
    ax.plot(steps, mean, "o-", ms=3.2, lw=2.2, color="black", zorder=6,
            markerfacecolor="white", markeredgecolor="black")
    # shaded: the plastic window (mean ordering value > 0), decaying to
    # zero as the body converges; dashed: the entire dense-vs-dropout gap
    ax.axvspan(100, 1600, color=C_GREEN, alpha=0.10, lw=0)
    ax.axhline(0.107, color="0.45", ls="--", lw=0.9, zorder=1)
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.21, 0.34)
    ax.set_xticks(steps)
    show = {"50", "200", "800", "3200"}
    ax.set_xticklabels([s if s in show else "" for s in map(str, steps)],
                       fontsize=6.0)
    ax.set_xlabel("training step (checkpoint body)")
    ax.set_ylabel("ordering value\n(random $-$ waterfill loss)")
    ax.set_title("(b) The value field's life cycle", loc="left")


def panel_c(ax):
    groups = ["Random\nwindows", "Online\nchasing", "Static\n$g{=}0.5$"]
    acc_11m = [51.6, 36.7, 31.8]
    acc_150 = [44.2, 41.7, 22.4]
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w / 2, acc_11m, w, color=C_BLUE, label="11M")
    b2 = ax.bar(x + w / 2, acc_150, w, color=C_VERM, label="starved-150M")
    ax.axhline(64.2, color=C_GREEN, lw=1.1, ls="--")
    # above the line, not on it
    ax.annotate("dense (11M)", (2.42, 64.2), textcoords="offset points",
                xytext=(0, 2.5), fontsize=5.6, color=C_GREEN, ha="right",
                va="bottom")
    ax.set_xticks(x, groups)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 88)
    # single-row legend in the clear zone above the dense line (64.2)
    ax.legend(loc="upper left", ncol=2, frameon=False, borderaxespad=0.1,
              columnspacing=0.8, handlelength=1.2, handletextpad=0.4)
    ax.set_title("(c) Chasing loses to random", loc="left", fontsize=6.6)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.72), layout="constrained")
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.savefig("paper/teaser.pdf")
    print("wrote paper/teaser.pdf")


if __name__ == "__main__":
    main()
