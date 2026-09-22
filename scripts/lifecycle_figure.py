"""Life-cycle figure: ordering value of the compute-value field vs training.

Ordering value = (random - waterfill) held-out loss under zero-shot
allocation on frozen checkpoint bodies (permutation-controlled readout,
see results-headroom/trajectory_seed{0,1}.json). Positive = the field's
structure is real; the value decays to ~0 as bodies converge. The step-5000
sign flips are seed-inconsistent and pre-registered as noise, not claimed.

Writes paper/lifecycle.pdf. Run from repo root:
  .venv/bin/python scripts/lifecycle_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [0, 1]
# Okabe-Ito colorblind-safe palette
COLORS = {0: "#0072B2", 1: "#D55E00"}
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150})


def main():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    means_x = means_y = None
    for s in SEEDS:
        d = json.load(open(f"results-headroom/trajectory_seed{s}.json"))
        tr = d["trajectory"]
        steps = [t["step"] for t in tr]
        val = [t["random"] - t["waterfill"] for t in tr]
        ax.plot(steps, val, "o-", ms=4, lw=1.2, color=COLORS[s],
                label=f"seed {s}", alpha=0.85)
    # seed mean
    curves = []
    for s in SEEDS:
        d = json.load(open(f"results-headroom/trajectory_seed{s}.json"))
        curves.append([t["random"] - t["waterfill"] for t in d["trajectory"]])
    steps = [t["step"] for t in json.load(
        open("results-headroom/trajectory_seed0.json"))["trajectory"]]
    mean = [(a + b) / 2 for a, b in zip(*curves)]
    ax.plot(steps, mean, "s--", ms=5, lw=1.6, color="black",
            label="seed mean", alpha=0.9)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps], fontsize=8)
    ax.set_xlabel("training step (checkpoint body)")
    ax.set_ylabel("ordering value\n(random $-$ waterfill held loss)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("The value field has a life cycle", fontsize=10)
    fig.tight_layout()
    out = os.path.join("paper", "lifecycle.pdf")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
