"""Generate ICLR paper tables + number macros from experiment JSONs.

Reads (all optional — missing stages emit `??` and the paper still compiles):
  results-p0/tagged/{predictor-supervised,gategrad,shuffled}_seed*.json
  results-p0/tagged/predictor-supervised_stage1_seed*.json   (stage-1 ref)
  results-p1/tagged-v2/baseline_seed*.json                   (window probes)
  results-p1/tagged-v2/gategrad*_seed*.json                  (value migration)
  results-p2, results-p3, results-nl                         (later stages)

Writes paper/results_iclr.tex (table bodies + \\newcommand macros) and prints
a preview. Run from repo root:
  .venv/bin/python paper/generate_tables_iclr.py
"""
import glob
import json
import os
import statistics as st

OUT = os.path.join("paper", "results_iclr.tex")


def load(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return runs


def ms(vals, scale=1.0, nd=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = scale * sum(vals) / len(vals)
    s = scale * (st.stdev(vals) if len(vals) > 1 else 0.0)
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}"


def mval(vals, scale=1.0, nd=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return f"{scale * sum(vals) / len(vals):.{nd}f}"


def spread(r):
    gb = r["final"].get("gate_by_bin") or []
    if len(gb) < 2 or gb[0] is None or gb[-1] is None:
        return None
    return gb[-1] - gb[0]


def p0_block():
    arms = [("tag", "predictor-supervised_seed{}.json", "Tag targets (labels)"),
            ("gategrad", "gategrad_seed{}.json", "Gate-grad (measured value)"),
            ("shuffled", "shuffled_seed{}.json", "Shuffled control")]
    rows = []
    for key, pat, label in arms:
        runs = [json.load(open(p)) for p in
                sorted(glob.glob(f"results-p0/tagged/{pat.format('*')}"))]
        if not runs:
            continue
        acc = ms([r["final"]["acc"] for r in runs], 100, 1)
        bal = ms([r["final"]["balanced_acc"] for r in runs], 100, 1)
        fl = ms([r["final"]["flops_per_token"] for r in runs], 1e-6, 1)
        sp = ms([spread(r) for r in runs], 1, 2)
        rows.append(f"{label} & {acc} & {bal} & {fl} & {sp} \\\\")
    if not rows:
        return "% P0 table pending\n", {}
    body = ("\\begin{tabular}{lcccc}\n\\toprule\n"
            "Stage-2 arm & Acc & Balanced & MFLOPs/tok & Gate spread \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    macros = {}
    for key, pat, _ in arms:
        runs = [json.load(open(p)) for p in
                sorted(glob.glob(f"results-p0/tagged/{pat.format('*')}"))]
        if runs:
            a = ms([r["final"]["acc"] for r in runs], 100, 1)
            if a:
                macros[f"{key.capitalize()}Acc"] = a
    return body, macros


def budget_block():
    """Targets vs realized gates (the budget-redistribution table)."""
    rows = []
    sources = [("Tag targets", "results-p0/tagged/predictor-supervised_seed*.json",
                None, 0.51, 0.60),
               ("Gate-grad", "results-p0/tagged/gategrad_seed*.json",
                "mean_target_problem", 0.31, 0.70),
               ("Shuffled", "results-p0/tagged/shuffled_seed*.json",
                "mean_target_problem", 0.31, 0.70)]
    for label, pat, key, nom_mean, nom_spread in sources:
        runs = [json.load(open(p)) for p in sorted(glob.glob(pat))]
        if not runs:
            continue
        if key:
            vals = []
            for r in runs:
                od = r.get("oracle_diag") or {}
                vals += od.get(key, [])
            mt = mval(vals, 1, 2) if vals else "?"
        else:
            mt = f"{nom_mean:.2f} (fixed)"
        gates = [r["final"].get("mean_gate") for r in runs]
        sp = ms([spread(r) for r in runs], 1, 2)
        rows.append(f"{label} & {mt} & {nom_spread:.2f} & "
                    f"{mval(gates, 1, 2)} & {sp} \\\\")
    if not rows:
        return "% budget table pending\n"
    return ("\\begin{tabular}{lcccc}\n\\toprule\n"
            "Arm & mean target & nominal spread & realized $\\bar g$ & "
            "realized spread \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")


def migration_block():
    """Per-bin measured value by model size (the pre-registered readout)."""
    import numpy as np
    sizes = [("medium", "42M"), ("150m", "150M"), ("400m", "400M")]
    dimsig = {"medium": (768, 6), "150m": (1024, 12), "400m": (1280, 20)}
    rows = []
    for size, label in sizes:
        for path in sorted(glob.glob(
                "results-p1/tagged-v2/gategrad*_seed*.json")):
            r = json.load(open(path))
            cfg = r["config"]
            if cfg.get("size_name", size) != size and \
                    dimsig.get(size) != (cfg.get("d_model"), cfg.get("n_layers")):
                continue
            od = r.get("oracle_diag") or {}
            if not od.get("bin_frac_nonpos"):
                continue
            frac = np.mean(od["bin_frac_nonpos"], axis=0)
            score = np.mean(od["bin_score_mean"], axis=0)
            cells_f = "/".join(f"{v:.2f}" for v in frac)
            cells_s = "/".join(f"{v:+.1e}" for v in score)
            rows.append(f"{label} & {cells_f} & {cells_s} \\\\")
    if not rows:
        return "% migration table pending\n"
    return ("\\begin{tabular}{lcc}\n\\toprule\n"
            "Size & frac(score$\\le$0) per bin (E/M/H) & mean score per bin \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def window_block():
    """P1a window probes: hard-bin acc at the largest token budget per size."""
    out = []
    for path in sorted(glob.glob("results-p1/tagged-v2/baseline_seed*.json")):
        r = json.load(open(path))
        hist = r["history"]
        if not hist.get("bin_acc"):
            continue
        hard = [h[-1] for h in hist["bin_acc"]]
        toks = r["history"]["step"][-1] * r["config"]["batch_size"] * \
            r["config"]["seq_len"]
        size = r["config"].get("size_name", f"{r['params']/1e6:.0f}M")
        out.append(f"{size} & {toks/1e6:.0f}M & "
                   f"{100*hard[0]:.0f}/{100*hard[1]:.0f}/{100*hard[-1]:.0f} \\\\")
    if not out:
        return "% window table pending\n"
    return ("\\begin{tabular}{lccc}\n\\toprule\n"
            "Size & tokens & E/M/H acc (\\%) \\\\\n\\midrule\n"
            + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    blocks = {}
    body, macros = p0_block()
    blocks["pzerotable"] = body
    blocks["budgettable"] = budget_block()
    blocks["migrationtable"] = migration_block()
    blocks["windowtable"] = window_block()
    with open(OUT, "w") as fh:
        fh.write("% auto-generated by paper/generate_tables_iclr.py — do not edit\n")
        for name, m in sorted(macros.items()):
            fh.write(f"\\expandafter\\def\\csname {name}\\endcsname{{{m}}}\n")
        for name, b in blocks.items():
            fh.write(f"\\newcommand{{\\{name}}}[0]{{\n{b}}}\n")
    print(f"wrote {OUT}")
    for name, b in blocks.items():
        has_data = not b.startswith("%")
        print(f"  {name}: {'OK' if has_data else 'PENDING'}")


if __name__ == "__main__":
    main()
