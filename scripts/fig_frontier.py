"""Frontier figure: accuracy vs billed training compute, per scale.

Redesign for legibility (the original had thin 'x' markers, five arms
stacked at nearly identical x, and 6.5pt annotations that collided):
- shared marker vocabulary: dense = circle, dropout/windows = triangle-up,
  field-chasing = diamond, static = square, ours-tag = star, MoD = pentagon
- filled markers with dark edges (no thin crosses); scale = color
- the 11M gated arms share x ~ 0.5 (same budget): they are drawn as a
  vertical stem with one label group; dropout/windows are a tie and are
  labeled together
- 150M arms (2 seeds, seed-invariant FLOPs): vertical seed-range line +
  mean marker, direct arm labels to the right of the cluster
- legend outside the data area (top strip): markers = arms, colors =
  scales

Points regenerate from committed artifacts only:
  11M  : results-headroom/single_*_seed0.json (causal singles, 3000 steps)
  150M : results-p3b/tagged-v2/*_seed{0,1}.json (probe grid, 5000 steps)
  400M : results-p1/tagged-v2/baseline_seed0.json (collapsed calibration
         probe, untrainable at any LR swept, per DECISIONS.md)

Writes paper/frontier.pdf. Run from repo root:
  .venv/bin/python scripts/fig_frontier.py
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito
C_11 = "#0072B2"    # blue (11M)
C_150 = "#D55E00"   # vermillion (150M)
C_400 = "#000000"   # black (400M)
C_DENSE = "#009E73"
plt.rcParams.update({
    "font.size": 7.2, "axes.titlesize": 8, "axes.labelsize": 7.4,
    "legend.fontsize": 6.6, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200})

MARKERS = {
    "dense": ("o", "dense"),
    "dropout": ("^", "windows/dropout"),
    "static": ("s", "static"),
    "online": ("D", "chasing"),
    "ours": ("*", "ours (tag)"),
    "mod": ("p", "MoD"),
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
    arms = [("baseline_seed{}.json", "dense"),
            ("fixed-schedule_seed{}.json", "static"),
            ("fixed2stage_150m_seed{}.json", "static2stage"),
            ("predictor-supervised_150m_seed{}.json", "ours"),
            ("shuffled_150m_seed{}.json", "dropout"),
            ("mod_seed{}.json", "mod")]
    ref, pts = None, {}
    for pat, name in arms:
        accs, ratios = [], []
        for p in sorted(glob.glob(f"results-p3b/tagged-v2/{pat.format('*')}")):
            r = json.load(open(p))
            fpt = r["final"]["flops_per_token"]
            ref = fpt if name == "dense" else ref
            accs.append(100 * r["final"]["acc"])
            ratios.append(fpt)
        if not accs:
            continue
        if ref is None:
            ref = ratios[0]
        pts[name] = ([x / ref for x in ratios], accs)
    return pts


def main():
    fig, ax = plt.subplots(figsize=(4.2, 2.9))

    # ---- 11M: dense at 1.0; gated cluster on a stem at its mean budget
    m11 = fig11m()
    xs_gated = [m11[a][0] for a in ("dropout", "shuffled", "online", "static")]
    x_stem = float(np.mean(xs_gated))
    ax.scatter([m11["dense"][0]], [m11["dense"][1]], s=44, marker="o",
               color=C_11, edgecolor="black", linewidths=0.6, zorder=5)
    ax.annotate(f"dense {m11['dense'][1]:.1f}", (1.0, m11["dense"][1]),
                textcoords="offset points", xytext=(-2, 6), fontsize=6.6,
                color=C_11, ha="right")
    ys = {a: m11[a][1] for a in ("dropout", "shuffled", "online", "static")}
    ax.plot([x_stem, x_stem], [ys["static"], max(ys["dropout"], ys["shuffled"])],
            color=C_11, lw=1.0, alpha=0.55, zorder=3)
    for arm in ("static", "online"):
        ax.scatter([x_stem], [ys[arm]], s=34, marker=MARKERS[arm][0],
                   color=C_11, edgecolor="black", linewidths=0.6, zorder=5)
    tie_y = (ys["dropout"] + ys["shuffled"]) / 2
    ax.scatter([x_stem, x_stem], [ys["dropout"], ys["shuffled"]], s=34,
               marker="^", color=C_11, edgecolor="black", linewidths=0.6,
               zorder=5)
    ax.annotate("windows $\\approx$ dropout",
                (x_stem, tie_y), textcoords="offset points", xytext=(6, 3),
                fontsize=6.4, color=C_11, va="bottom")
    ax.annotate("field-chasing", (x_stem, ys["online"]),
                textcoords="offset points", xytext=(7, -1), fontsize=6.4,
                color=C_11, va="center")
    ax.annotate("static", (x_stem, ys["static"]),
                textcoords="offset points", xytext=(7, -1), fontsize=6.4,
                color=C_11, va="center")

    # ---- 150M: seed-range stems + mean markers. All gated arms share a
    # 0.57-0.59x budget, so on a log axis they would collapse into one
    # knot; x positions inside the cluster are evenly spaced for
    # readability (noted in the caption), spanning ~0.55-0.72x.
    p150 = fig150m()
    gated = [k for k in p150 if k != "dense"]
    dodge = dict(zip(gated, np.geomspace(0.55, 0.72, len(gated))))
    for name, (ratios, accs) in p150.items():
        if name == "dense":
            x = ratios[0]
            ax.scatter([x], [np.mean(accs)], s=44, marker="o", color=C_150,
                       edgecolor="black", linewidths=0.6, zorder=5)
            ax.annotate("dense", (x, np.mean(accs)),
                        textcoords="offset points", xytext=(-2, 6),
                        fontsize=6.6, color=C_150, ha="right")
            continue
        x = dodge[name]
        if len(accs) > 1:
            ax.plot([x, x], [min(accs), max(accs)], color=C_150, lw=1.0,
                    alpha=0.55, zorder=3)
        mk = MARKERS.get(name, ("s", None))[0]
        ax.scatter([x], [np.mean(accs)], s=40 if name != "ours" else 62,
                   marker=mk, color=C_150, edgecolor="black", linewidths=0.6,
                   zorder=5)


    # ---- 400M collapsed probe
    w = json.load(open("results-p1/tagged-v2/baseline_seed0.json"))
    acc = 100 * w["history"]["heldout_acc"][-1] if w["history"].get("heldout_acc") \
        else 100 * w["history"]["bin_acc"][0][-1]
    ax.scatter([1.0], [acc], s=40, marker="^", color=C_400,
               edgecolor="black", linewidths=0.6, zorder=5)
    ax.annotate("400M probe:\nuntrainable", (1.0, acc),
                textcoords="offset points", xytext=(-4, 8), fontsize=6.4,
                color=C_400, ha="right")

    ax.set_xscale("log")
    ax.set_xlim(0.40, 1.62)
    ax.set_ylim(0, 84)
    ax.set_xlabel("billed training FLOPs/token, relative to dense (=1.0)")
    ax.set_ylabel("held-out accuracy (%)")
    ax.axvline(1.0, color="gray", lw=0.7, ls=":")
    ax.set_title("Accuracy vs.\\ billed training compute, by scale",
                 fontsize=8, loc="left")

    # legend strip above the axes, outside the data
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker=m, color="0.25",
                      markeredgecolor="black", markeredgewidth=0.6,
                      markersize=6, label=lab)
               for m, lab in MARKERS.values()]
    handles += [Line2D([], [], ls="", marker="o", color=c,
                       markeredgecolor="black", markeredgewidth=0.6,
                       markersize=6, label=sc)
                for c, sc in [(C_11, "11M"), (C_150, "150M"), (C_400, "400M")]]
    ax.legend(handles=handles, frameon=False, fontsize=6.2, ncol=4,
              loc="lower left", bbox_to_anchor=(0.0, 1.01), columnspacing=0.9,
              handletextpad=0.25, borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig("paper/frontier.pdf")
    print("wrote paper/frontier.pdf")


if __name__ == "__main__":
    main()
