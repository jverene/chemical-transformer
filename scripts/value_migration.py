"""Compute-value migration across model sizes (pre-registered P1 readout).

Scans results-p1/<task>/gategrad_seed*.json (one per size) and reports, per
size, the per-bin mean gate-grad score, per-bin frac(score<=0), and the
mean target vs pinned budget. Writes a printed table plus
figures/value_migration.pdf.

Pre-registered prediction (DECISIONS.md): reducible mass migrates toward
harder bins as scale grows — H's frac_nonpos falls toward M's; per-bin score
gaps (H-M) shrink / cross. If H's frac_nonpos stays pinned high across
sizes, the irreducibility story is in trouble.
"""
import argparse
import glob
import json
import os

import numpy as np

SIZES = ["medium", "150m", "400m", "1b"]
SIZE_LABEL = {"medium": "42M", "150m": "150M", "400m": "400M", "1b": "1B"}


DIM_SIGNATURE = {  # fallback for runs whose config predates size_name
    "medium": (768, 6), "150m": (1024, 12), "400m": (1280, 20), "1b": (2048, 20),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results-p1")
    p.add_argument("--task", default="tagged-v2")
    p.add_argument("--figdir", default="figures")
    args = p.parse_args()

    rows = []
    for size in SIZES:
        paths = sorted(glob.glob(os.path.join(
            args.outdir, args.task, "gategrad*_seed*.json")))
        for path in paths:
            with open(path) as f:
                r = json.load(f)
            cfg = r["config"]
            sig = (cfg.get("d_model"), cfg.get("n_layers"))
            matches_size = cfg.get("size_name") == size or \
                DIM_SIGNATURE.get(size) == sig
            if not matches_size:
                continue
            od = r.get("oracle_diag") or {}
            if not od.get("bin_frac_nonpos"):
                continue
            # average over logged steps
            nbins = len(od["bin_frac_nonpos"][0])
            frac = np.mean([x for x in od["bin_frac_nonpos"] if x], axis=0)
            score = np.mean([x for x in od["bin_score_mean"] if x], axis=0)
            tgt = float(np.mean(od.get("mean_target_problem", [np.nan])))
            rows.append(dict(size=size, label=SIZE_LABEL.get(size, size),
                             params=r["params"], frac=frac.tolist(),
                             score=score.tolist(), mean_target=tgt,
                             realized_spread=_spread(r),
                             realized_gates=_gates(r)))

    if not rows:
        print("no gategrad probes found (sizes: config.size_name missing?)")
        return

    print(f"\n=== compute-value migration ({args.task}) ===")
    print(f"{'size':>6} {'params':>12} | per-bin frac(score<=0) | "
          f"per-bin mean score (a.u.) | tgt  spread  gates")
    for r in rows:
        fr = " ".join(f"{v:.2f}" for v in r["frac"])
        sc = " ".join(f"{v:+.1e}" for v in r["score"])
        gb = r.get("realized_gates") or []
        gbs = "/".join(f"{v:.2f}" for v in gb) if gb else "-"
        print(f"{r['label']:>6} {r['params']:>12,} | {fr} | {sc} | "
              f"{r['mean_target']:.2f} {r['realized_spread']:+.2f}  {gbs}")

    # The prediction, stated numerically: gap between H and M frac_nonpos
    if all(len(r["frac"]) >= 3 for r in rows):
        print("\nH-minus-M frac_nonpos by size (prediction: shrinks with scale):")
        for r in rows:
            print(f"  {r['label']:>6}: {r['frac'][2] - r['frac'][1]:+.3f}")

    os.makedirs(args.figdir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for r in rows:
            b = np.arange(1, len(r["frac"]) + 1)
            axes[0].plot(b, r["frac"], marker="o", label=r["label"])
            axes[1].plot(b, r["score"], marker="o", label=r["label"])
        for ax, yl in zip(axes, ["frac(score<=0)", "mean score (a.u.)"]):
            ax.set_xlabel("difficulty bin")
            ax.set_ylabel(yl)
            ax.legend(title="size")
        fig.tight_layout()
        out = os.path.join(args.figdir, "value_migration.pdf")
        fig.savefig(out)
        print(f"saved {out}")
    except Exception as e:  # matplotlib missing on remote nodes is fine
        print(f"(no figure: {e})")


def _spread(r):
    gb = (r.get("final") or {}).get("gate_by_bin") or []
    if len(gb) >= 3 and gb[0] is not None:
        return gb[-1] - gb[0]
    return float("nan")


def _gates(r):
    gb = (r.get("final") or {}).get("gate_by_bin") or []
    return [g for g in gb if g is not None]


if __name__ == "__main__":
    main()
