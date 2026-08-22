"""Generate the 6 RAAAI paper figures (PDF) into figures/.

Standalone script; tolerates missing runs (warns and skips) so it can be
re-run as experiments finish. Run with: .venv/bin/python make_figures_raaai.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = "figures"

# --- style -----------------------------------------------------------------
OURS_COLOR = "#d62728"      # strong red for our method
MUTED = {
    "baseline":       {"label": "Baseline",        "color": "#0072B2", "marker": "s"},
    "fixed-schedule": {"label": "Fixed-schedule",  "color": "#E69F00", "marker": "^"},
    "mod":            {"label": "MoD",             "color": "#009E73", "marker": "D"},
    "chemical-off":   {"label": "Chemical-off",    "color": "#999999", "marker": "v"},
}
OURS_LABEL = "Difficulty-supervised (ours)"
DIFF3 = ["Easy", "Medium", "Hard"]
DIFF4 = ["1-digit", "2-digit", "3-digit", "4-digit"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "errorbar.capsize": 3,
    "figure.dpi": 110,
})


def load_seed_runs(pattern):
    """Load all JSON runs matching a glob pattern; returns list of dicts."""
    runs = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                runs.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  warning: could not read {path}: {e}")
    if not runs:
        print(f"  warning: no runs found for {pattern}")
    return runs


def mean_std(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)


def final_stats(runs, key):
    """Mean/std over seeds of final[key] (scalar or list)."""
    vals = [r["final"][key] for r in runs if key in r.get("final", {})]
    if not vals:
        return None, None
    arr = np.array(vals, dtype=float)  # (n_seeds,) or (n_seeds, n_bins)
    if arr.ndim == 1:
        m, s = mean_std(arr)
        return m, s
    ms, ss = [], []
    for j in range(arr.shape[1]):
        m, s = mean_std(arr[:, j])
        ms.append(m)
        ss.append(s)
    return np.array(ms), np.array(ss)


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bar_panel(ax, means, stds, labels, color, edgecolor=None):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=color, edgecolor=edgecolor or "none",
           capsize=3, width=0.65, error_kw=dict(ecolor="0.25", lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


# ---------------------------------------------------------------------------
def fig_journey():
    iters = [
        ("1: Attention entropy",      "results/tagged/chemical_seed*.json"),
        ("2: Tag-initialized",        "results-opt1/tagged/chemical_seed*.json"),
        ("3: Learned head (unsup.)",  "results-opt2/tagged/predictor_seed*.json"),
        ("4: Supervised (ours)",      "results-opt3b-s1-5k/tagged/predictor-supervised_seed*.json"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
    drew = False
    for ax, (title, pattern) in zip(axes, iters):
        runs = load_seed_runs(pattern)
        if not runs:
            ax.set_title(title)
            ax.text(0.5, 0.5, "data missing", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            continue
        means, stds = final_stats(runs, "gate_by_bin")
        if means is None or len(means) != 3:
            print(f"  warning: bad gate_by_bin for {pattern}")
            continue
        color = OURS_COLOR if "ours" in title else "0.55"
        bar_panel(ax, means, stds, DIFF3, color)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        drew = True
    axes[0].set_ylabel("Mean gate")
    if drew:
        save(fig, "fig_journey.pdf")
    else:
        print("  skipped fig_journey.pdf (no data)")


def fig_correlation():
    cpath = "results/correlations.json"
    if not os.path.exists(cpath):
        print(f"  skipped fig_correlation.pdf ({cpath} missing)")
        return
    with open(cpath) as f:
        corr = json.load(f)["iterations"]
    keys = list(corr.keys())
    labels = [corr[k]["label"] for k in keys]
    means, stds = [], []
    for k in keys:
        m, s = mean_std(corr[k]["per_problem_r"])
        means.append(m)
        stds.append(s)
    colors = [OURS_COLOR if "ours" in lab.lower() else "0.55" for lab in labels]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bar_panel(ax, means, stds, labels, colors)
    ax.axhline(0.0, color="0.25", ls="--", lw=1)
    ax.set_ylabel("Pearson r (signal vs. difficulty)")
    ax.set_ylim(-1.05, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, "fig_correlation.pdf")


def _frontier_panel(ax, base_runs, ours_runs, title):
    pts = []
    for method, st in MUTED.items():
        runs = base_runs.get(method)
        if not runs:
            print(f"  warning: missing {method} for {title} frontier")
            continue
        fm, fs = final_stats(runs, "flops_per_token")
        am, asd = final_stats(runs, "acc")
        if fm is None or am is None:
            continue
        ax.errorbar(fm / 1e6, am, xerr=(fs or 0) / 1e6, yerr=asd or 0,
                    label=st["label"], color=st["color"], marker=st["marker"],
                    ms=7, lw=0, elinewidth=1)
        pts.append((fm, am))
    if ours_runs:
        fm, fs = final_stats(ours_runs, "flops_per_token")
        am, asd = final_stats(ours_runs, "acc")
        if fm is not None and am is not None:
            ax.errorbar(fm / 1e6, am, xerr=(fs or 0) / 1e6, yerr=asd or 0,
                        label=OURS_LABEL, color=OURS_COLOR, marker="o",
                        ms=9, lw=0, elinewidth=1.2, zorder=5)
            pts.append((fm, am))
    pts.sort()
    frontier, best = [], -np.inf
    for f, a in pts:
        if a > best:
            frontier.append((f, a))
            best = a
    if len(frontier) > 1:
        fx, fa = zip(*frontier)
        ax.plot(np.array(fx) / 1e6, fa, "k--", lw=1, alpha=0.6,
                label="Pareto frontier")
    ax.set_xlabel("FLOPs / token (millions)")
    ax.set_title(title)
    ax.legend(loc="best")
    return bool(pts)


def fig_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    drew = False
    # tagged
    tagged = {}
    for m in MUTED:
        tagged[m] = load_seed_runs(f"results/tagged/{m}_seed*.json")
    ours_t = load_seed_runs("results-opt3b-s1-5k/tagged/predictor-supervised_seed*.json")
    drew |= _frontier_panel(axes[0], tagged, ours_t, "tagged")
    axes[0].set_ylabel("Held-out accuracy")
    # mixed (our method may not exist yet)
    mixed = {}
    for m in MUTED:
        mixed[m] = load_seed_runs(f"results/mixed/{m}_seed*.json")
    ours_m = load_seed_runs("results-opt3b-full/mixed/predictor-supervised_seed*.json")
    if not ours_m:
        print("  warning: results-opt3b-full/mixed missing; "
              "mixed frontier panel shows baselines only")
    drew |= _frontier_panel(axes[1], mixed, ours_m, "mixed")
    if drew:
        save(fig, "fig_frontier.pdf")
    else:
        print("  skipped fig_frontier.pdf (no data)")


def fig_gate_ordering():
    tagged = load_seed_runs("results-opt3b-s1-5k/tagged/predictor-supervised_seed*.json")
    mixed = load_seed_runs("results-opt3b-full/mixed/predictor-supervised_seed*.json")
    if not mixed:
        print("  warning: results-opt3b-full/mixed missing; "
              "fig_gate_ordering will have a single (tagged) panel")
    panels = []
    if tagged:
        panels.append(("tagged", tagged, DIFF3))
    if mixed:
        panels.append(("mixed", mixed, DIFF4))
    if not panels:
        print("  skipped fig_gate_ordering.pdf (no data)")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4),
                             squeeze=False)
    for ax, (title, runs, labels) in zip(axes[0], panels):
        means, stds = final_stats(runs, "gate_by_bin")
        if means is None or len(means) != len(labels):
            print(f"  warning: bad gate_by_bin for {title} in fig_gate_ordering")
            continue
        bar_panel(ax, means, stds, labels, OURS_COLOR)
        ax.set_title(f"{title} (ours)")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean gate")
    save(fig, "fig_gate_ordering.pdf")


def fig_perdifficulty_acc():
    groups = [
        ("Baseline", load_seed_runs("results/tagged/baseline_seed*.json"),
         MUTED["baseline"]["color"]),
        ("Fixed-schedule", load_seed_runs("results/tagged/fixed-schedule_seed*.json"),
         MUTED["fixed-schedule"]["color"]),
        ("Ours", load_seed_runs("results-opt3b-s1-5k/tagged/predictor-supervised_seed*.json"),
         OURS_COLOR),
    ]
    groups = [(n, r, c) for n, r, c in groups if r]
    if not groups:
        print("  skipped fig_perdifficulty_acc.pdf (no data)")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    n_diff, n_grp = 3, len(groups)
    width = 0.8 / n_grp
    x = np.arange(n_diff)
    for i, (name, runs, color) in enumerate(groups):
        means, stds = final_stats(runs, "bin_acc")
        if means is None or len(means) != n_diff:
            print(f"  warning: bad bin_acc for {name} in fig_perdifficulty_acc")
            continue
        ax.bar(x + (i - (n_grp - 1) / 2) * width, means, width, yerr=stds,
               label=name, color=color, capsize=3,
               error_kw=dict(ecolor="0.25", lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(DIFF3)
    ax.set_ylabel("Held-out accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    save(fig, "fig_perdifficulty_acc.pdf")


def _history_band(ax, runs, extract, ylabel, color):
    """Plot mean +/- std band over seeds. extract(run) -> (steps, values)."""
    series = []
    for r in runs:
        s, v = extract(r)
        if s is not None:
            series.append((np.asarray(s, float), np.asarray(v, float)))
    if not series:
        return False
    steps = series[0][0]
    vals = []
    for s, v in series:
        if not np.array_equal(s, steps):
            v = np.interp(steps, s, v)
        vals.append(v)
    vals = np.array(vals)
    m = vals.mean(axis=0)
    sd = vals.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros_like(m)
    ax.plot(steps, m, color=color, lw=1.8)
    ax.fill_between(steps, m - sd, m + sd, color=color, alpha=0.2, lw=0)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    return True


def fig_training_curves():
    runs = load_seed_runs("results-opt3b-s1-5k/tagged/predictor-supervised_seed*.json")
    if not runs:
        print("  skipped fig_training_curves.pdf (no data)")
        return

    def acc_series(r):
        h = r.get("history", {})
        if "step" not in h or "acc" not in h:
            return None, None
        return h["step"], h["acc"]

    def spread_series(r):
        h = r.get("history", {})
        gb = h.get("gate_by_bin")
        if not gb:
            return None, None
        steps, vals = [], []
        for step, g in zip(h["step"], gb):
            if g is None or len(g) < 3 or g[0] is None or g[-1] is None:
                continue  # tolerate nulls
            steps.append(step)
            vals.append(g[-1] - g[0])  # hard - easy
        if not steps:
            return None, None
        return steps, vals

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    drew = _history_band(axes[0], runs, acc_series, "Held-out accuracy", OURS_COLOR)
    axes[0].set_title("(a) Accuracy")
    drew |= _history_band(axes[1], runs, spread_series,
                          "Gate spread (hard - easy)", OURS_COLOR)
    axes[1].set_title("(b) Gate spread")
    axes[1].axhline(0.0, color="0.25", ls="--", lw=1)
    if drew:
        save(fig, "fig_training_curves.pdf")
    else:
        print("  skipped fig_training_curves.pdf (no data)")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig_journey()
    fig_correlation()
    fig_frontier()
    fig_gate_ordering()
    fig_perdifficulty_acc()
    fig_training_curves()


if __name__ == "__main__":
    main()
