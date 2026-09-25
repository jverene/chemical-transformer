"""Figure 4 (punchline): the allocation ladder — even the perfect
allocator loses to doing nothing — and the converged coin flip.

(a) Starved-150M frozen-body oracle economics (the regime where the value
    field is most structured): zero-shot held-out loss for waterfill
    (oracle-best ordering), random, and anti (oracle-worst) allocation of
    the 0.5 budget; both seeds as dots, bars = means. The green dashed
    line is NO budgeting (uniform full compute). The whole ladder —
    including its best rung — pays +1.2..1.5 loss vs doing nothing.
    Connectors annotate: ordering right-vs-wrong is worth 2.1-2.3x
    ordering right-vs-random.
(b) Why there is nothing to chase at convergence: per-bin frac(score<=0)
    of the measured field on converged healthy bodies (42M, 150M) — a
    51/49 coin flip in every bin (Table migration in appendix).

Sources:
  results-p3b/headroom_starved_150m_seed{0,1}.json
  results-p1/tagged-v2/gategrad*_seed*.json  (oracle_diag bin_frac_nonpos)

Writes paper/ladder.pdf. Run from repo root:
  .venv/bin/python scripts/fig_ladder.py
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_GRAY = "#555555"
plt.rcParams.update({
    "font.size": 6.8, "axes.titlesize": 7.2, "axes.labelsize": 6.6,
    "legend.fontsize": 6.0, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200})


def ladder_data():
    runs = [json.load(open(p)) for p in
            sorted(glob.glob("results-p3b/headroom_starved_150m_seed*.json"))]
    return {
        "uniform": [r["uniform"] for r in runs],
        "waterfill": [r["waterfill"] for r in runs],
        "random": [r["random"] for r in runs],
        "anti": [r["anti"] for r in runs],
    }


def coinflip_data():
    """Per-bin frac(score<=0), converged bodies: 42M and 150M (as in the
    migration table generator)."""
    sizes = {"42M": ("medium", (768, 6)), "150M": ("150m", (1024, 12))}
    out = {}
    for label, (size, dimsig) in sizes.items():
        fracs = []
        for path in sorted(glob.glob("results-p1/tagged-v2/gategrad*_seed*.json")):
            r = json.load(open(path))
            cfg = r["config"]
            if cfg.get("size_name", size) != size and \
                    dimsig != (cfg.get("d_model"), cfg.get("n_layers")):
                continue
            od = r.get("oracle_diag") or {}
            if od.get("bin_frac_nonpos"):
                fracs.append(np.mean(od["bin_frac_nonpos"], axis=0))
        if fracs:
            out[label] = np.mean(fracs, axis=0)
    return out


def panel_a(ax, d):
    xs = np.arange(3)
    names = ["Waterfill\n(oracle-best)", "Random", "Anti\n(oracle-worst)"]
    means = [np.mean(d[k]) for k in ("waterfill", "random", "anti")]
    bars = ax.bar(xs, means, width=0.55,
                  color=["0.72", "0.82", C_VERM], zorder=2)
    for i, k in enumerate(("waterfill", "random", "anti")):
        ax.scatter([i] * 2, d[k], s=9, color="black", zorder=4, alpha=0.8)
        ax.plot([i - 0.14, i + 0.14], [means[i]] * 2, color="black", lw=1.2,
                zorder=5)
    # no-budgeting line, labeled in the clear strip right of the anti bar
    um = float(np.mean(d["uniform"]))
    ax.axhline(um, color=C_GREEN, ls="--", lw=1.2, zorder=3)
    ax.annotate("no budgeting\n(uniform)", (2.36, um), fontsize=5.8,
                color=C_GREEN, va="center", ha="left")
    ax.set_xticks(xs, names)
    ax.set_xlim(-0.6, 3.2)
    ax.set_ylabel("zero-shot held-out loss")
    ax.set_ylim(0, 5.6)
    ax.set_title("(a) Everything loses to doing nothing "
                 f"(best rung pays +{np.mean(np.array(d['waterfill']) - d['uniform']):.1f})",
                 loc="left")


def panel_b(ax, cf):
    bins = np.arange(3)
    for i, (label, col) in enumerate([("42M", C_BLUE), ("150M", C_VERM)]):
        if label in cf:
            ax.scatter(bins + (-0.12 if i == 0 else 0.12), cf[label], s=12,
                       color=col, zorder=3, label=label)
    ax.axhline(0.5, color=C_GRAY, ls=":", lw=0.9)
    ax.annotate("coin flip", (2.42, 0.503), fontsize=5.8, color=C_GRAY,
                ha="right", va="bottom")
    ax.set_xticks(bins, ["E", "M", "H"])
    ax.set_ylim(0.44, 0.56)
    ax.set_ylabel("frac(score$\\leq 0$)")
    ax.set_xlabel("difficulty bin")
    ax.legend(frameon=False, loc="lower right", borderaxespad=0.2,
              labelspacing=0.25, handletextpad=0.2)
    ax.set_title("(b) At convergence: 51/49", loc="left")


def main():
    d = ladder_data()
    cf = coinflip_data()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0),
                             gridspec_kw={"width_ratios": [1.75, 1]})
    panel_a(axes[0], d)
    panel_b(axes[1], cf)
    fig.tight_layout(w_pad=1.6)
    fig.savefig("paper/ladder.pdf")
    print("wrote paper/ladder.pdf; coinflip:", {k: np.round(v, 2).tolist()
                                                for k, v in cf.items()})


if __name__ == "__main__":
    main()
