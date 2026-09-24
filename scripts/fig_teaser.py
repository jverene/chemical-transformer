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
    ax.set_xticks(bins, labels)
    ax.set_xlabel("difficulty bin")
    ax.set_ylabel("mean FFN gate")
    ax.set_ylim(0.15, 0.72)
    ax.set_title("(a) Endogenous signals fail", loc="left")


def panel_b(ax):
    curves = []
    for s, c in [(0, C_BLUE), (1, C_VERM)]:
        d = json.load(open(f"results-headroom/trajectory_seed{s}.json"))
        tr = d["trajectory"]
        steps = [t["step"] for t in tr]
        v = [t["random"] - t["waterfill"] for t in tr]
        curves.append(v)
        ax.plot(steps, v, "o-", ms=3.2, lw=1.2, color=c, alpha=0.8,
                label=f"seed {s}")
    mean = [(a + b) / 2 for a, b in zip(*curves)]
    ax.plot(steps, mean, "s--", ms=3.8, lw=1.7, color="black",
            label="seed mean", zorder=5)
    # plastic window: seed-mean ordering value is positive here
    ax.axvspan(100, 1600, color=C_GREEN, alpha=0.09, lw=0)
    ax.annotate("plastic window:\nordering value $> 0$", (850, 0.255),
                fontsize=5.8, color=C_GREEN, ha="center")
    ax.annotate("converged: flat\n(frac score$\\leq 0 \\approx 0.51$)",
                (2900, -0.145), fontsize=5.8, color="0.35", ha="center")
    # reference height: the entire dense-vs-dropout gap the recipe pays,
    # drawn under the peak so the peak's ~2x magnitude is visible
    ax.axhline(0.107, color="0.45", ls="--", lw=0.9, zorder=1)
    ax.annotate("dense$-$dropout gap", (3300, 0.128), fontsize=5.4,
                color="0.35", ha="right")
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.185, 0.315)
    ax.set_xticks(steps)
    show = {"50", "200", "800", "3200"}
    ax.set_xticklabels([s if s in show else "" for s in map(str, steps)],
                       fontsize=6.0)
    ax.set_xlabel("training step (checkpoint body)")
    ax.set_ylabel("ordering value\n(random $-$ waterfill loss)")
    ax.legend(loc="upper right", frameon=False, borderaxespad=0.1,
              labelspacing=0.25)
    ax.set_title("(b) The value field's life cycle", loc="left")


def panel_c(ax):
    groups = ["Random\nwindows", "Online\nfield-chasing", "Static $g{=}0.5$"]
    acc_11m = [51.6, 36.7, 31.8]
    acc_150 = [44.2, 41.7, 22.4]
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w / 2, acc_11m, w, color=C_BLUE, label="11M (3000 steps)")
    b2 = ax.bar(x + w / 2, acc_150, w, color=C_VERM,
                label="starved-150M (2500 steps)")
    ax.axhline(64.2, color=C_GREEN, lw=1.1, ls="--")
    ax.annotate("dense, 11M: 64.2", (2.42, 64.2), fontsize=5.8,
                color=C_GREEN, va="center")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", fontsize=5.6, padding=1.2)
    ax.set_xticks(x, groups)
    ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 72)
    # free zone: above the left bars (top 51.6), below the dense line (64.2)
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.1,
              labelspacing=0.25)
    ax.set_title("(c) Chasing loses to random", loc="left")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.86), layout="constrained")
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    # panel (a)'s legend lives OUTSIDE the axes (below), reserved properly
    # by constrained layout via the figure-level 'outside' legend location
    hnd, lbl = axes[0].get_legend_handles_labels()
    fig.legend(hnd, lbl, loc="outside lower left", ncol=2, frameon=False,
               handlelength=1.3, labelspacing=0.25, columnspacing=0.8,
               fontsize=5.6)
    fig.savefig("paper/teaser.pdf")
    print("wrote paper/teaser.pdf")


if __name__ == "__main__":
    main()
