"""Generate the paper figures (PDF) from results/*.json.

Tolerates missing runs: a panel is skipped (with a warning) if its data is not
on disk yet, so the script can be re-run as experiments finish.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "results"
FIGDIR = "figures"

# Colorblind-safe palette (Okabe-Ito): colors stay distinguishable under
# deuteranopia, protanopia, and tritanopia; markers add a second (non-color)
# channel. Keep new entries within this palette.
METHOD_STYLE = {
    "chemical":       {"label": "Chemical (ours)", "color": "#D55E00", "marker": "o"},
    "baseline":       {"label": "Baseline",        "color": "#0072B2", "marker": "s"},
    "chemical-off":   {"label": "Chemical-off",    "color": "#56B4E9", "marker": "^"},
    "mod":            {"label": "MoD",             "color": "#009E73", "marker": "D"},
    "random-gate":    {"label": "Random-gate",     "color": "#CC79A7", "marker": "v"},
    "fixed-schedule": {"label": "Fixed-schedule",  "color": "#E69F00", "marker": "<"},
    "tag-only":       {"label": "Tag-only",        "color": "#999999", "marker": ">"},
}


def load_runs(task):
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, task, "*_seed*.json"))):
        with open(path) as f:
            r = json.load(f)
        runs.setdefault(r["method"], []).append(r)
    return runs


def style(method):
    return METHOD_STYLE.get(method, {"label": method, "color": None, "marker": "o"})


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else np.zeros_like(vals.mean())


# ---------------------------------------------------------------------------
def fig1_pareto(tasks=("tagged", "mixed")):
    """Accuracy vs. FLOPs/token, all methods, Pareto frontier."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2), squeeze=False)
    drew = False
    for ax, task in zip(axes[0], tasks):
        runs = load_runs(task)
        if not runs:
            continue
        pts = []
        for method, rs in runs.items():
            accs = [r["final"]["acc"] for r in rs]
            flops = [r["final"]["flops_per_token"] for r in rs]
            am, asd = mean_std(accs)
            fm, fsd = mean_std(flops)
            st = style(method)
            ax.errorbar(fm / 1e6, am, xerr=fsd / 1e6, yerr=asd, label=st["label"],
                        color=st["color"], marker=st["marker"], ms=8, capsize=3)
            pts.append((fm, am))
            drew = True
        pts.sort()
        frontier, best = [], -1
        for f, a in pts:
            if a > best:
                frontier.append((f, a))
                best = a
        if len(frontier) > 1:
            fx, fa = zip(*frontier)
            ax.plot(np.array(fx) / 1e6, fa, "k--", lw=1, alpha=0.6, label="Pareto frontier")
        ax.set_xlabel("FLOPs / token (millions)")
        ax.set_ylabel("Held-out answer accuracy")
        ax.set_title(f"{task} arithmetic")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if drew:
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "fig1_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig1 done" if drew else "fig1 skipped: no results")


def fig2_acc_vs_flops_by_difficulty(task="tagged"):
    """Per-difficulty accuracy vs. per-difficulty FLOPs for gated methods."""
    runs = load_runs(task)
    if "chemical" not in runs:
        print("fig2 skipped: no chemical results")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bin_names = ["easy", "medium", "hard"] if task == "tagged" else ["1-digit", "2-digit", "3-digit", "4-digit"]
    for method in ("chemical", "mod", "fixed-schedule", "baseline"):
        if method not in runs:
            continue
        rs = runs[method]
        n_bins = len(rs[0]["final"]["bin_acc"])
        st = style(method)
        for b in range(n_bins):
            acc_m, acc_s = mean_std([r["final"]["bin_acc"][b] for r in rs])
            gates = [r["final"]["gate_by_bin"][b] for r in rs
                     if r["final"]["gate_by_bin"][b] is not None]
            if not gates:
                continue
            g = np.mean(gates)
            flops_b = _bin_flops(g, rs)
            ax.errorbar(flops_b / 1e6, acc_m, yerr=acc_s, color=st["color"],
                        marker=st["marker"], ms=7, capsize=3,
                        label=st["label"] if b == 0 else None)
            ax.annotate(bin_names[b], (flops_b / 1e6, acc_m), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("FLOPs / token (millions)")
    ax.set_ylabel("Answer accuracy")
    ax.set_title(f"Accuracy vs. compute by difficulty ({task})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_difficulty_pareto_{task}.pdf".format(task=task)), bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


def _bin_flops(g, rs):
    """Approximate per-token FLOPs for a difficulty bin with mean gate g.

    Uses the run's config: attention cost is gate-independent; FFN cost scales
    with the gate.
    """
    cfg = rs[0]["config"]
    d, dff, L, H, S = cfg["d_model"], cfg["d_ff"], cfg["n_layers"], cfg["n_heads"], cfg["seq_len"]
    attn = L * (4 * S * d * d + 2 * H * S * S * (d // H)) / S  # per token per layer sum
    ffn = L * 2 * (d * dff + dff * d) * 2
    return attn + ffn * g


def fig3_training_curves(task="tagged"):
    """Per-difficulty accuracy over training, mean +/- std shaded, Chemical vs Baseline."""
    runs = load_runs(task)
    if "chemical" not in runs or "baseline" not in runs:
        print("fig3 skipped: missing chemical or baseline")
        return
    n_bins = len(runs["chemical"][0]["history"]["bin_acc"][0])
    bin_names = ["easy", "medium", "hard"] if task == "tagged" else ["1-digit", "2-digit", "3-digit", "4-digit"]
    fig, axes = plt.subplots(1, n_bins, figsize=(4.2 * n_bins, 3.6), sharey=True)
    for b, ax in enumerate(axes):
        for method in ("chemical", "baseline"):
            rs = runs[method]
            steps = rs[0]["history"]["step"]
            curves = np.array([r["history"]["bin_acc"] for r in rs])[:, :, b]
            m = curves.mean(0)
            s = curves.std(0, ddof=1) if len(rs) > 1 else np.zeros_like(m)
            st = style(method)
            ax.plot(steps, m, label=st["label"], color=st["color"])
            ax.fill_between(steps, m - s, m + s, color=st["color"], alpha=0.2)
        ax.set_title(bin_names[b])
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("answer accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Per-difficulty accuracy over training ({task})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_training_curves_{task}.pdf".format(task=task)), bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


def fig4_gate_by_difficulty(task="tagged"):
    """Mean FFN gate value by difficulty (bar plot)."""
    runs = load_runs(task)
    methods = [m for m in ("chemical", "random-gate", "tag-only") if m in runs]
    if not methods:
        print("fig4 skipped: no gated results")
        return
    bin_names = ["easy", "medium", "hard"] if task == "tagged" else ["1-digit", "2-digit", "3-digit", "4-digit"]
    n_bins = len(bin_names)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    width = 0.8 / len(methods)
    xpos = np.arange(n_bins)
    for i, method in enumerate(methods):
        rs = runs[method]
        means, stds = [], []
        for b in range(n_bins):
            vals = [r["final"]["gate_by_bin"][b] for r in rs
                    if r["final"]["gate_by_bin"][b] is not None]
            m, s = mean_std(vals)
            means.append(m)
            stds.append(s)
        st = style(method)
        ax.bar(xpos + i * width, means, width * 0.9, yerr=stds, capsize=3,
               label=st["label"], color=st["color"])
    ax.set_xticks(xpos + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(bin_names)
    ax.set_ylabel("mean FFN gate")
    ax.set_title(f"Compute allocation by difficulty ({task})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_gate_by_difficulty_{task}.pdf".format(task=task)), bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


def fig5_entropy_scatter(task="tagged"):
    """Entropy vs. per-token correctness: binned accuracy + correlation."""
    runs = load_runs(task)
    if "chemical" not in runs:
        print("fig5 skipped: no chemical results")
        return
    r = runs["chemical"][0]["final"]
    if "entropy_samples" not in r:
        print("fig5 skipped: no entropy samples")
        return
    ent = np.array(r["entropy_samples"])
    cor = np.array(r["correct_samples"])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    qs = np.quantile(ent, np.linspace(0, 1, 21))
    centers, accs = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (ent >= lo) & (ent <= hi)
        if m.sum() > 10:
            centers.append(ent[m].mean())
            accs.append(cor[m].mean())
    ax.plot(centers, accs, "o-", color="#D55E00", label="binned accuracy")
    idx = np.random.RandomState(0).choice(len(ent), size=min(1500, len(ent)), replace=False)
    ax.scatter(ent[idx], cor[idx] + np.random.RandomState(1).uniform(-0.03, 0.03, len(idx)),
               s=3, alpha=0.15, color="gray", label="tokens (jittered)")
    pear = np.mean([rr["final"]["entropy_pearson"] for rr in runs["chemical"]])
    spear = np.mean([rr["final"]["entropy_spearman"] for rr in runs["chemical"]])
    ax.set_xlabel("normalized attention entropy")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"Entropy predicts errors ({task})\n"
                 f"Pearson r={pear:.3f}, Spearman rho={spear:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_entropy_scatter_{task}.pdf".format(task=task)), bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")


def fig6_probe(task="tagged"):
    """Probe accuracy by layer."""
    paths = sorted(glob.glob(os.path.join(RESULTS, "probe", f"{task}_seed*.json")))
    if not paths:
        print("fig6 skipped: no probe results")
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    per_layer = {}
    chance = None
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        chance = r["chance"]
        for k, v in r["probe_acc_by_layer"].items():
            per_layer.setdefault(int(k.split("_")[1]), []).append(v)
    layers = sorted(per_layer)
    m = [np.mean(per_layer[l]) for l in layers]
    s = [np.std(per_layer[l], ddof=1) if len(per_layer[l]) > 1 else 0 for l in layers]
    ax.errorbar(layers, m, yerr=s, marker="o", color="#D55E00", capsize=3)
    ax.axhline(chance, color="0.4", ls="--", label=f"chance ({chance:.2f})")
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy")
    ax.set_title(f"Linear probe on chemical state ({task})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_probe.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


def fig7_flops_histogram(task="tagged"):
    """Per-token FLOPs distribution, Chemical vs. Baseline."""
    runs = load_runs(task)
    if "chemical" not in runs or "baseline" not in runs:
        print("fig7 skipped: missing results")
        return
    if "compute_samples" not in runs["chemical"][0]["final"]:
        print("fig7 skipped: no compute samples")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for method in ("baseline", "chemical"):
        vals = np.concatenate([
            np.array(r["final"]["compute_samples"]) for r in runs[method]
            if "compute_samples" in r["final"]
        ])
        st = style(method)
        ax.hist(vals / 1e6, bins=60, alpha=0.55, label=st["label"], color=st["color"],
                density=True)
    ax.set_xlabel("FLOPs / token (millions)")
    ax.set_ylabel("density")
    ax.set_title(f"Per-token compute distribution ({task})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_flops_histogram_{task}.pdf".format(task=task)), bbox_inches="tight")
    plt.close(fig)
    print("fig7 done")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig1_pareto()
    for task in ("tagged", "mixed"):
        fig2_acc_vs_flops_by_difficulty(task)
        fig3_training_curves(task)
        fig4_gate_by_difficulty(task)
        fig5_entropy_scatter(task)
        fig7_flops_histogram(task)
    fig6_probe("tagged")


if __name__ == "__main__":
    main()
