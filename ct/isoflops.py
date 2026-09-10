"""Iso-FLOPs analysis: compare methods at matched billed training FLOPs.

Rules (plan §4):
  - Two-stage runs concatenate the Stage-1 history with the Stage-2 history,
    offset by the Stage-1's cumulative billed FLOPs; the stage boundary is a
    kink (billing rate and loss landscape both change).
  - Interpolation happens WITHIN a segment only. A budget falling in a gap
    between segments (should be ~one eval interval) yields None, never a
    bridged value.
  - Common budgets are capped at the minimum achieved budget across all
    compared curves (no extrapolation past the shortest run).
"""
import argparse
import glob
import json
import os

import numpy as np


def _load(path):
    with open(path) as f:
        return json.load(f)


def _stage1_json(r, outdir, task, seed):
    """Resolve the Stage-1 JSON for a Stage-2 run (fork-aware)."""
    cand = r.get("stage1_ckpt")
    if cand:
        j = cand.replace(".pt", ".json")
        if os.path.exists(j):
            return j
    fallback = os.path.join(outdir, task,
                            f"predictor-supervised_stage1_seed{seed}.json")
    return fallback if os.path.exists(fallback) else None


def run_curves(outdir, task, method, seed, metric="acc"):
    """One seed's (segment list). Segment = dict(cum, val, stage)."""
    p = os.path.join(outdir, task, f"{method}_seed{seed}.json")
    if not os.path.exists(p):
        return None
    r = _load(p)
    hist = r.get("history", {})
    if "cum_train_flops" not in hist:
        return None  # legacy run without billing -> excluded (steps-matched only)
    segs = []
    if r["config"].get("stage") == 2:
        s1p = _stage1_json(r, outdir, task, seed)
        if s1p:
            r1 = _load(s1p)
            h1 = r1.get("history", {})
            if "cum_train_flops" in h1:
                segs.append({"cum": np.asarray(h1["cum_train_flops"], float),
                             "val": np.asarray(h1[metric], float),
                             "stage": 1})
    segs.append({"cum": np.asarray(hist["cum_train_flops"], float),
                 "val": np.asarray(hist[metric], float),
                 "stage": r["config"].get("stage", 1)})
    return segs


def interp_at(segs, budget):
    """Value at `budget` from the single segment covering it; None otherwise."""
    for seg in segs:
        if len(seg["cum"]) and seg["cum"][0] <= budget <= seg["cum"][-1]:
            return float(np.interp(budget, seg["cum"], seg["val"]))
    return None  # gap between segments or beyond the curve


def iso_table(outdir, task, methods, seeds, metric="acc", n_budgets=6,
              save=True):
    curves = {}
    for m in methods:
        cs = [run_curves(outdir, task, m, s, metric) for s in seeds]
        cs = [c for c in cs if c]
        if cs:
            curves[m] = cs
    missing = [m for m in methods if m not in curves]
    if missing:
        print(f"(no billed-FLOPs curves for: {', '.join(missing)})")
    if not curves:
        return None

    # No extrapolation: cap at the shortest seed curve's achieved budget;
    # start once every method has meaningful history (curves begin near zero).
    max_budget = min(c[-1]["cum"][-1] for cs in curves.values() for c in cs)
    min_budget = 0.15 * max_budget
    budgets = np.linspace(min_budget, max_budget, n_budgets)

    table = {"task": task, "metric": metric, "budgets": budgets.tolist(),
             "methods": {}}
    print(f"\n=== iso-FLOPs: {metric} at matched billed training FLOPs "
          f"({task}) ===")
    header = f"{'budget(GFLOPs)':>15} | " + " | ".join(f"{m:>22}" for m in curves)
    print(header)
    for B in budgets:
        row = f"{B / 1e9:15.0f} | "
        cells = []
        for m, cs in curves.items():
            vals = [interp_at(c, B) for c in cs]
            vals = [v for v in vals if v is not None]
            if vals:
                cells.append(f"{np.mean(vals):6.4f}±{np.std(vals):.4f} (n={len(vals)})")
            else:
                cells.append(f"{'--':>22}")
        print(row + " | ".join(f"{c:>22}" for c in cells))
        table["methods"][m] = cells

    # per-seed ordering consistency at the top budget (paired-by-seed where
    # methods share a Stage-1 body)
    top = budgets[-1]
    if len(curves) >= 2:
        names = list(curves)
        a, b = names[0], names[1]
        wins = ties = n = 0
        for i in range(min(len(curves[a]), len(curves[b]))):
            va, vb = interp_at(curves[a][i], top), interp_at(curves[b][i], top)
            if va is None or vb is None:
                continue
            n += 1
            wins += va > vb
            ties += va == vb
        print(f"ordering at top budget: {a} > {b} in {wins}/{n} seeds "
              f"({ties} ties)")
        table["ordering"] = {"a": a, "b": b, "a_wins": wins, "n": n,
                             "ties": ties}

    if save:
        out = os.path.join(outdir, task, "isoflops.json")
        with open(out, "w") as f:
            json.dump(table, f, indent=2)
        print(f"saved {out}")
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results-p0")
    p.add_argument("--task", default="tagged")
    p.add_argument("--methods", required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--metric", default="acc")
    p.add_argument("--budgets", type=int, default=6)
    args = p.parse_args()
    iso_table(args.outdir, args.task, args.methods.split(","),
              [int(s) for s in args.seeds.split(",")],
              metric=args.metric, n_budgets=args.budgets)


if __name__ == "__main__":
    main()
