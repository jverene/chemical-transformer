"""Frontier figure: accuracy vs billed training compute, per scale.

One panel, three scales. x-axis = billed training FLOPs per token
RELATIVE to the dense arm at the same scale (dense = 1.0x), so scales
share an axis honestly; y = held-out accuracy. Points regenerate from
committed artifacts only:
  11M  : results-headroom/single_*_seed0.json (causal singles, 3000 steps)
  150M : results-p3b/tagged-v2/*_seed{0,1}.json (probe grid, 5000 steps)
  400M : results-p1/tagged-v2/baseline_seed0.json (calibration probe —
         collapsed; untrainable at any LR swept, per DECISIONS.md)

Writes paper/frontier.pdf. Run from repo root:
  .venv/bin/python scripts/fig_frontier.py
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
C_11 = "#56B4E9"    # sky blue
C_150 = "#D55E00"   # vermillion
C_400 = "#000000"   # black
C_DENSE = "#009E73" # bluish green
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150})

ARM_STYLE = {
    "dense": ("o", "Dense"),
    "dropout": ("^", "FFN dropout"),
    "shuffled": ("x", "Random windows"),
    "online": ("D", "Field-chasing"),
    "static": ("s", "Static g=0.5"),
}


def fig11m():
    d = json.load(open("results-headroom/single_dense_seed0.json"))
    ref = d["history"]["cum_flops"][-1]
    out = {"dense": (1.0, 100 * d["history"]["heldout_acc"][-1])}
    for arm in ["dropout", "shuffled", "online", "static"]:
        r = json.load(open(f"results-headroom/single_{arm}_seed0.json"))
        out[arm] = (r["history"]["cum_flops"][-1] / ref,
                    100 * r["history"]["heldout_acc"][-1])
    return out


def fig150m():
    arms = [("baseline", "dense"), ("fixed-schedule", "static"),
            ("fixed2stage_150m", "static2stage"),
            ("predictor-supervised_150m", "ours"), ("shuffled_150m", "shuffled"),
            ("mod", "mod")]
    ref = None
    pts = {}
    for pat, name in arms:
        accs, ratios = [], []
        for p in sorted(glob.glob(f"results-p3b/tagged-v2/{pat}_seed*.json")):
            r = json.load(open(p))
            fpt = r["final"]["flops_per_token"]
            if name == "dense":
                ref = fpt
            accs.append(100 * r["final"]["acc"])
            ratios.append(fpt)
        if not accs:
            continue
        if ref is None:
            ref = ratios[0]
        pts[name] = ([x / ref for x in ratios], accs)
    return pts


def main():
    fig, ax = plt.subplots(figsize=(4.0, 2.7))

    m11 = fig11m()
    xs = [m11[a][0] for a in m11]
    ys = [m11[a][1] for a in m11]
    ax.scatter(xs, ys, s=46, facecolors="none", edgecolors=C_11, linewidths=1.6,
               label="11M (3000 steps)", marker="s")
    for a, (x, y) in m11.items():
        mk, lab = ARM_STYLE.get(a, ("o", a))
        ax.annotate(lab if a != "dense" else "dense", (x, y),
                    textcoords="offset points", xytext=(6, -3), fontsize=6.5,
                    color=C_11)

    p150 = fig150m()
    for name, (xs, ys) in p150.items():
        mk, lab = {"dense": ("o", None), "static": ("s", None),
                   "static2stage": ("s", None), "ours": ("*", "Ours (tag)"),
                   "shuffled": ("x", "Random 2-stage"), "mod": ("D", "MoD")}.get(
            name, ("o", name))
        ax.scatter(xs, ys, s=54, color=C_150, marker=mk,
                   label="150M grid (5000 steps)" if name == "baseline" else None,
                   zorder=3)
        if name == "ours":
            ax.annotate("Ours (tag)", (xs[0], ys[0]), textcoords="offset points",
                        xytext=(6, 3), fontsize=6.5, color=C_150)

    w = json.load(open("results-p1/tagged-v2/baseline_seed0.json"))
    acc = 100 * w["history"]["heldout_acc"][-1] if w["history"].get("heldout_acc") \
        else 100 * w["history"]["bin_acc"][0][-1]
    ax.scatter([1.0], [acc], s=46, color=C_400, marker="^",
               label="400M probe (collapsed)")
    ax.annotate("400M: untrainable\n(any LR swept)", (1.0, acc),
                textcoords="offset points", xytext=(8, -4), fontsize=6.5,
                color=C_400)

    ax.set_xscale("log")
    ax.set_xlabel("billed training FLOPs/token, relative to dense (=1.0)")
    ax.set_ylabel("held-out accuracy (%)")
    ax.axvline(1.0, color="gray", lw=0.7, ls=":")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Accuracy vs.\\ billed training compute, by scale", fontsize=9)
    fig.tight_layout()
    fig.savefig("paper/frontier.pdf")
    print("wrote paper/frontier.pdf")


if __name__ == "__main__":
    main()
