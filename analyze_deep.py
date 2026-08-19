"""Deep analyses beyond aggregate JSON stats.

Runs a token-level collection pass on the checkpoints to recover per-token
gate / entropy / difficulty / correctness signals (the JSONs only store
aggregates), plus derives FLOPs-accuracy frontier and per-difficulty
breakdowns directly from the JSONs.

Usage:  .venv/bin/python analyze_deep.py
Output: figures/an{1..7}_*.pdf
"""
import glob
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats

from ct.config import Config, get_device
from ct.models import build_model
from ct.tasks import get_task
from ct.tasks.base import get_batch
from ct.models.common import make_causal_mask
from ct.tokenizer import TOKENIZER

RESULTS = "results"
FIGDIR = "figures"
DEVICE = get_device()

METHOD_STYLE = {
    "chemical":       {"label": "Chemical (ours)", "color": "#d62728", "marker": "o"},
    "baseline":       {"label": "Baseline",        "color": "#1f77b4", "marker": "s"},
    "chemical-off":   {"label": "Chemical-off",    "color": "#ff7f0e", "marker": "^"},
    "mod":            {"label": "MoD",             "color": "#2ca02c", "marker": "D"},
    "random-gate":    {"label": "Random-gate",     "color": "#9467bd", "marker": "v"},
    "fixed-schedule": {"label": "Fixed-schedule",  "color": "#8c564b", "marker": "<"},
    "tag-only":       {"label": "Tag-only",        "color": "#7f7f7f", "marker": ">"},
}


def style(method):
    return METHOD_STYLE.get(method, {"label": method, "color": None, "marker": "o"})


def load_run(method, task, seed):
    path = os.path.join(RESULTS, task, f"{method}_seed{seed}.json")
    with open(path) as f:
        return json.load(f)


def build_model_from_json(run, seed=0):
    cfg = Config(**{k: v for k, v in run["config"].items()
                    if k in Config.__dataclass_fields__})
    cfg.device = DEVICE
    model = build_model(run["method"], cfg)
    path = os.path.join(RESULTS, run["task"], f"{run['method']}_seed{seed}.pt")
    sd = torch.load(path, map_location=DEVICE)
    model.load_state_dict(sd)
    model.to(DEVICE)
    return model, cfg


@torch.no_grad()
def collect_token_level(model, task, cfg, n_batches=8, meta_layers=True):
    """Per-token signals over the frozen held-out set.

    Returns dict of numpy arrays over real (non-pad) tokens:
      gate, entropy, diff,  - full sequence tokens
      gate_a, ent_a, diff_a, corr_a - answer tokens (m mask)
      layer_gates           - (L, S) for the first batch's first sequence
    """
    model.eval()
    rng = random.Random(999)
    held_out = task.build_held_out(cfg)

    G, E, D = [], [], []
    Ga, Ea, Da, Ca = [], [], [], []
    layer_gates = None

    for bi in range(n_batches):
        x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size, held_out=held_out)
        pad = x != TOKENIZER.pad_id
        logits, info = model(x, pad_mask=pad, meta={"diffs": diffs})

        preds = logits.argmax(dim=-1)
        tgt_valid = y != TOKENIZER.pad_id
        ans_t = torch.roll(ans, -1, dims=1)
        ans_t[:, -1] = 0
        m = tgt_valid & (ans_t == 1)
        corr = (preds == y) & m

        g = info["gate_mean"]
        ent = info["entropy_mean"]
        G.append(g[pad].float().cpu().numpy())
        D.append(diffs[pad].float().cpu().numpy())
        if ent is not None:
            E.append(ent[pad].float().cpu().numpy())

        Ga.append(g[m].float().cpu().numpy())
        Ca.append(corr[m].float().cpu().numpy())
        Da.append(diffs_t_of(m, diffs).float().cpu().numpy())
        if ent is not None:
            Ea.append(ent[m].float().cpu().numpy())

        if meta_layers and bi == 0:
            layer_gates = collect_layer_gates(model, x[0:1], pad[0:1], diffs[0:1])

    out = {
        "gate": np.concatenate(G),
        "diff": np.concatenate(D),
        "entropy": np.concatenate(E) if E else None,
        "gate_a": np.concatenate(Ga),
        "corr_a": np.concatenate(Ca),
        "diff_a": np.concatenate(Da),
        "entropy_a": np.concatenate(Ea) if Ea else None,
        "layer_gates": layer_gates,
    }
    return out


def diffs_t_of(m, diffs):
    dt = torch.roll(diffs, -1, dims=1)
    dt[:, -1] = -1
    return dt[m]


@torch.no_grad()
def collect_layer_gates(model, x, pad, diffs):
    """Replicate forward to capture per-layer gate maps for a single sequence."""
    model.eval()
    B, S = x.shape
    pos = torch.arange(S, device=x.device).unsqueeze(0)
    h = model.dropout(model.token_emb(x) + model.pos_emb(pos))
    causal_mask = make_causal_mask(S, x.device, x.dtype if x.dtype.is_floating_point else torch.float32)
    chemical = None
    gates = []
    for layer in model.layers:
        h, chemical, g, flops, ent = layer(h, chemical, causal_mask, pad, {"diffs": diffs})
        gates.append(g)
    return torch.stack(gates).float().cpu().numpy()  # (L, S)


def annotate(ax, tag, r, title=None):
    if title:
        ax.set_title(title)
    ax.set_ylabel("mean FFN gate")
    ax.set_xticks(range(len(tag)))
    ax.set_xticklabels(tag)
    ax.grid(alpha=0.3, axis="y")
    return ax


def fig1_gate_by_difficulty(task):
    """Mean gate value by true difficulty (E/M/H)."""
    rs = [load_run("chemical", task, s) for s in (0, 1, 2)]
    tags = get_task(task).bin_names
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    means, stds = [], []
    for b in range(len(tags)):
        vals = [r["final"]["gate_by_bin"][b] for r in rs if r["final"]["gate_by_bin"][b] is not None]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
    ax.bar(range(len(tags)), means, yerr=stds, capsize=4, color="#d62728", alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="gate=0.5 (neutral)")
    annotate(ax, tags, rs, f"Mean FFN gate by difficulty ({task})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"an1_gate_by_difficulty_{task}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("an1 done", task)


def fig2_gate_histogram(task, coll):
    """Gate distribution histogram per difficulty. Bimodal (good) vs uniform (bad)."""
    tags = get_task(task).bin_names
    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 3.6), sharey=True)
    for b, (ax, tag) in enumerate(zip(axes, tags)):
        vals = coll["gate"][coll["diff"] == b]
        ax.hist(vals, bins=40, color="#d62728", alpha=0.8, density=True)
        ax.axvline(np.mean(vals), color="k", ls="--", lw=1, label=f"mean {np.mean(vals):.2f}")
        ax.set_title(f"{tag} (n={len(vals)})")
        ax.set_xlabel("FFN gate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle(f"Gate distribution by difficulty ({task})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"an2_gate_hist_{task}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("an2 done", task)


def fig3_gate_vs_accuracy(task, coll):
    """Gate value vs token accuracy: do skipped (gate<0.5) tokens get answered less?"""
    gate = coll["gate_a"]
    corr = coll["corr_a"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    qs = np.quantile(gate, np.linspace(0, 1, 21))
    centers, accs = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (gate >= lo) & (gate <= hi)
        if m.sum() > 10:
            centers.append(gate[m].mean())
            accs.append(corr[m].mean())
    ax.plot(centers, accs, "o-", color="#d62728")
    ax.set_xlabel("FFN gate (mean over layers)")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"Gate vs answer accuracy ({task})")
    ax.grid(alpha=0.3)

    ax = axes[1]
    low = gate < 0.5
    high = ~low
    ax.bar(["gate<0.5", "gate>=0.5"],
           [corr[low].mean() if low.any() else 0, corr[high].mean() if high.any() else 0],
           color=["#7f7f7f", "#1f77b4"])
    for i, m in enumerate((low, high)):
        if m.any():
            ax.text(i, corr[m].mean() + 0.01, f"n={m.sum()}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(correct)")
    ax.set_title(f"Correct by gate threshold ({task})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"an3_gate_vs_accuracy_{task}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("an3 done", task)


def fig4_entropy_difficulty(task, coll):
    """Entropy vs difficulty: Pearson/Spearman. Is the core mechanism's premise real?"""
    ent = coll["entropy"]
    diff = coll["diff"]
    r_p = scipy_stats.pearsonr(ent, diff).statistic
    r_s = scipy_stats.spearmanr(ent, diff).statistic
    tags = get_task(task).bin_names
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    for b, tag in enumerate(tags):
        vals = ent[diff == b]
        ax.hist(vals, bins=40, alpha=0.55, label=tag, density=True)
    ax.set_xlabel("normalized attention entropy")
    ax.set_ylabel("density")
    ax.set_title(f"Entropy by difficulty ({task})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ent_a = coll["entropy_a"]
    corr_a = coll["corr_a"]
    r_pc = scipy_stats.pearsonr(ent_a, corr_a).statistic
    r_sc = scipy_stats.spearmanr(ent_a, corr_a).statistic
    qs = np.quantile(ent_a, np.linspace(0, 1, 21))
    centers, accs = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (ent_a >= lo) & (ent_a <= hi)
        if m.sum() > 10:
            centers.append(ent_a[m].mean())
            accs.append(corr_a[m].mean())
    ax.plot(centers, accs, "o-", color="#d62728")
    ax.set_xlabel("normalized attention entropy")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"Entropy vs correctness ({task})\n"
                 f"pearson {r_pc:.3f}, spearman {r_sc:.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Entropy-difficulty: pearson {r_p:.3f}, spearman {r_s:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"an4_entropy_difficulty_{task}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("an4 done", task, f"(r_p={r_p:.3f}, r_s={r_s:.3f})")


def fig5_layer_trajectory(task):
    """Gate across layers for a single hard example: adaptive drop or flat?"""
    run = load_run("chemical", task, 0)
    cfg = Config(**{k: v for k, v in run["config"].items()
                    if k in Config.__dataclass_fields__})
    cfg.device = DEVICE
    model, cfg = build_model_from_json(run)

    # a hard example (tagged) or 4-digit (mixed)
    if task == "tagged":
        s = "[H]4317*682=2944194"
        tag = "hard multiplication"
    else:
        s = "9321*482=4492722"
        tag = "4-digit multiplication"
    toks = TOKENIZER.encode(s) + [TOKENIZER.eos_id]
    x = torch.tensor([toks], device=DEVICE)
    diffs = torch.full((1, len(toks)), -1, dtype=torch.long, device=DEVICE)
    layer_gates = collect_layer_gates(model, x, torch.ones_like(x, dtype=torch.bool), diffs)
    layer_gates = layer_gates[:, 0, :]  # (L, S)
    L, S = layer_gates.shape

    # pick representative positions
    label_pos = {c: i for i, c in enumerate(s)}
    pos_names = ["first op digit", "operator", "second op start", "equals", "first ans digit"]
    picks = []
    if task == "tagged":
        picks = [4, 7, 9, 12, 13]   # '4'(4317), '1'(4317 op...), adjust below
    # general: first digit of a, operator, first digit of b, '=', first answer digit
    a = s[:s.index('*')]
    b = s[s.index('*') + 1:s.index('=')]
    picks = [s.index(a[0]), s.index('*'), s.index(b[0]), s.index('='), s.index('=') + 1]
    pos_names = ["first op digit", "operator", "first op2 digit", "equals", "first ans digit"]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for p, nm in zip(picks, pos_names):
        ax.plot(range(1, L + 1), layer_gates[:, p], marker="o", label=nm)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("FFN gate")
    ax.set_title(f"Layer-wise gate trajectory: {tag} ({task})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"an5_layer_trajectory_{task}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("an5 done", task)


def fig6_frontier():
    """All methods on FLOPs/acc scatter, both tasks."""
    tasks = ("tagged", "mixed")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, task in zip(axes, tasks):
        for method in METHOD_STYLE:
            paths = glob.glob(os.path.join(RESULTS, task, f"{method}_seed*.json"))
            if not paths:
                continue
            accs, flops = [], []
            for p in paths:
                with open(p) as f:
                    r = json.load(f)
                accs.append(r["final"]["acc"])
                flops.append(r["final"]["flops_per_token"])
            st = style(method)
            ax.errorbar(np.mean(flops) / 1e6, np.mean(accs),
                        xerr=np.std(flops) / 1e6, yerr=np.std(accs),
                        label=st["label"], color=st["color"], marker=st["marker"],
                        ms=8, capsize=3)
        ax.set_xlabel("FLOPs / token (millions)")
        ax.set_ylabel("Held-out answer accuracy")
        ax.set_title(f"{task} arithmetic")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "an6_frontier_all_methods.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("an6 done")


def fig7_per_difficulty():
    """Accuracy per difficulty E/M/H: baseline vs chemical vs fixed-schedule."""
    task = "tagged"
    tags = get_task(task).bin_names
    methods = ("baseline", "chemical", "fixed-schedule")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        rs = [load_run(method, task, s) for s in (0, 1, 2)]
        means, stds = [], []
        for b in range(len(tags)):
            vals = [r["final"]["bin_acc"][b] for r in rs]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        st = style(method)
        ax.bar(np.arange(len(tags)) + i * width, means, width * 0.9, yerr=stds,
               capsize=3, label=st["label"], color=st["color"])
    ax.set_xticks(np.arange(len(tags)) + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(tags)
    ax.set_ylabel("answer accuracy")
    ax.set_title(f"Per-difficulty accuracy (tagged)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "an7_per_difficulty_accuracy.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("an7 done")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig6_frontier()
    fig7_per_difficulty()
    for task in ("tagged", "mixed"):
        run = load_run("chemical", task, 0)
        model, cfg = build_model_from_json(run)
        coll = collect_token_level(model, get_task(task), cfg)
        fig1_gate_by_difficulty(task)
        fig2_gate_histogram(task, coll)
        fig3_gate_vs_accuracy(task, coll)
        fig4_entropy_difficulty(task, coll)
        fig5_layer_trajectory(task)
        del model


if __name__ == "__main__":
    main()