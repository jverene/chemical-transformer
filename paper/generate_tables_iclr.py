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
            mt = mval(vals, 1, 2) if vals else "$\\sim$0.31"
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




def p150_perseed_block():
    """150M grid, per-seed accuracies (n=2; mean+-std in the main table
    hides the seed pairing reviewers will ask about)."""
    arms = [("baseline_seed{}.json", "Dense"),
            ("fixed-schedule_seed{}.json", "Fixed $g{=}0.5$ (scratch)"),
            ("fixed2stage_150m_seed{}.json", "Fixed (2-stage)"),
            ("predictor-supervised_150m_seed{}.json", "Ours (tag targets)"),
            ("shuffled_150m_seed{}.json", "Shuffled (2-stage)"),
            ("mod_seed{}.json", "MoD (single-stage)")]
    rows = []
    for pat, label in arms:
        runs = []
        for p in sorted(glob.glob(f"results-p3b/tagged-v2/{pat.format('*')}")):
            try:
                runs.append(json.load(open(p)))
            except (json.JSONDecodeError, OSError):
                pass
        if not runs:
            continue
        accs = [f"{100*r['final']['acc']:.1f}" for r in runs]
        fl = f"{runs[0]['final']['flops_per_token']/1e6:.0f}"
        while len(accs) < 2:
            accs.append("--")
        rows.append(f"{label} & {accs[0]} & {accs[1]} & {fl} \\\\")
    if not rows:
        return "% 150M per-seed table pending\n"
    return ("\\begin{tabular}{lccc}\n\\toprule\n"
            "Arm & seed 0 & seed 1 & MFLOPs/tok \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def causal_combined_block():
    """Both causal tests side by side: 11M singles + starved-150M singles."""
    rows_11, rows_150 = {}, {}
    for arm in ["dense", "dropout", "shuffled", "online", "static"]:
        try:
            h = json.load(open(f"results-headroom/single_{arm}_seed0.json"))["history"]
            rows_11[arm] = (100 * h["heldout_acc"][-1], h["heldout_loss"][-1])
        except (json.JSONDecodeError, OSError):
            pass
        try:
            h = json.load(open(
                f"results-p3b/tagged-v2/single_{arm}_150m_seed0.json"))["history"]
            rows_150[arm] = (100 * h["heldout_acc"][-1], h["heldout_loss"][-1])
        except (json.JSONDecodeError, OSError):
            pass
    if not rows_11:
        return "% combined causal table pending\n"
    order = [("dense", "Dense (reference)"), ("dropout", "Token-level FFN dropout"),
             ("shuffled", "Random windows"), ("online", "Online field-chasing"),
             ("static", "Static $g{=}0.5$")]

    def cell(d, nd=1):
        return f"{d[0]:.{nd}f}" if d else "--"

    def cell3(d):
        return f"{d[1]:.3f}" if d else "--"

    rows = []
    for arm, label in order:
        a, b = rows_11.get(arm), rows_150.get(arm)
        rows.append(f"{label} & {cell(a)} & {cell3(a)} & "
                    f"{cell(b)} & {cell3(b)} \\\\")
    macros = {}
    for arm in rows_11:
        macros[f"Causal{arm.capitalize()}Acc"] = f"{rows_11[arm][0]:.1f}"
        macros[f"Causal{arm.capitalize()}Loss"] = f"{rows_11[arm][1]:.3f}"
    for arm in rows_150:
        macros[f"Starved{arm.capitalize()}Acc"] = f"{rows_150[arm][0]:.1f}"
        macros[f"Starved{arm.capitalize()}Loss"] = f"{rows_150[arm][1]:.3f}"
    body = ("\\begin{tabular}{lcccc}\n\\toprule\n"
            "& \\multicolumn{2}{c}{11M (3000 steps)} & "
            "\\multicolumn{2}{c}{starved-150M (2500 steps)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Arm & acc (\\%) & loss & acc (\\%) & loss \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return body, macros


def p150_block():
    """The 150M probe-recipe grid (methods x 2 seeds, iso-FLOPs)."""
    arms = [("baseline_seed{}.json", "Dense (reference)"),
            ("fixed-schedule_seed{}.json", "Fixed $g{=}0.5$ (from scratch)"),
            ("fixed2stage_150m_seed{}.json", "Fixed (2-stage)"),
            ("predictor-supervised_150m_seed{}.json", "Ours (tag targets)"),
            ("shuffled_150m_seed{}.json", "Shuffled (2-stage)"),
            ("mod_seed{}.json", "MoD (single-stage)")]
    rows, macros = [], {}
    for pat, label in arms:
        runs = []
        for p in sorted(glob.glob(f"results-p3b/tagged-v2/{pat.format('*')}")):
            try:
                runs.append(json.load(open(p)))
            except (json.JSONDecodeError, OSError):
                pass
        if not runs:
            continue
        acc = ms([r["final"]["acc"] for r in runs], 100, 1)
        bal = ms([r["final"]["balanced_acc"] for r in runs], 100, 1)
        fl = ms([r["final"]["flops_per_token"] for r in runs], 1e-6, 0)
        gt = mval([r["final"].get("mean_gate") for r in runs], 1, 2)
        sp = ms([spread(r) for r in runs], 1, 2)
        rows.append(f"{label} & {acc} & {bal} & {fl} & {gt} & {sp} \\\\")
        key = pat.split("_seed")[0].split("-")[0]
        macros[f"Grid{key}Acc"] = acc
    if not rows:
        return "% 150M table pending\n", macros
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "Arm & Acc & Balanced & MFLOPs/tok & $\\bar g$ & "
            "gate spread \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
    return body, macros



def convergence_macros():
    """The 10k-step 150M parity pair (dense vs token-level FFN dropout)."""
    macros = {}
    try:
        d = json.load(open("results-headroom/convergence_dense_seed1.json"))
        r = json.load(open("results-headroom/convergence_rotation_seed1.json"))
        dl, rl = d["history"]["heldout_loss"][-1], r["history"]["heldout_loss"][-1]
        da, ra = d["history"]["heldout_acc"][-1], r["history"]["heldout_acc"][-1]
        df, rf = d["history"]["cum_flops"][-1], r["history"]["cum_flops"][-1]
        macros["ConvDenseLoss"] = f"{dl:.3f}"
        macros["ConvRotLoss"] = f"{rl:.3f}"
        macros["ConvRotGapPct"] = f"{100*(rl/dl-1):.1f}"
        macros["ConvDenseAcc"] = f"{100*da:.1f}"
        macros["ConvRotAcc"] = f"{100*ra:.1f}"
        macros["ConvRotFlopsPct"] = f"{100*rf/df:.0f}"
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return macros


def nl_macros(nl_domain):
    """Headline NL numbers MUST come from the same computation as the table:
    mean over per-seed paired gaps (2 decimals), mean losses (3 decimals).
    Kept in one place so abstract, intro, and Table can never drift apart."""
    macros = {}
    try:
        b = json.load(open("results-nl/webtext/baseline_seed0.json"))["final"]
        r = json.load(open("results-nl/webtext/rotation_seed0.json"))["final"]
        macros["NlDenseMFl"] = f"{b['flops_per_token']/1e6:.0f}"
        macros["NlSteps"] = str(json.load(
            open("results-nl/webtext/baseline_seed0.json"))["config"]["steps"])
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    per_seed = nl_domain.get("webtext")
    if per_seed:
        bl, gl, gaps = per_seed
        macros["NlDenseLoss"] = f"{sum(bl)/len(bl):.3f}"
        macros["NlRotLoss"] = f"{sum(gl)/len(gl):.3f}"
        macros["NlRotGapPct"] = f"{100*sum(gaps)/len(gaps):.2f}"
        macros["NlRotGapA"] = f"{100*gaps[0]:.2f}"
        macros["NlRotGapB"] = f"{100*gaps[-1]:.2f}"
        macros["NlRotGapList"] = "/".join(f"{100*g:.2f}" for g in gaps)
        macros["NlRotGapMax"] = f"{100*max(gaps):.2f}"
        macros["NlRotGapMin"] = f"{100*min(gaps):.2f}"
        macros["NlRotFlopsPct"] = "50"
    return macros


def nl_domain_block():
    """NL results: per-domain dense vs rotation pairs + webtext seeds.

    Loss/gap cells list per-seed values ('a/b' for 2 seeds) — with n<=2 a
    mean+-std is misleadingly tight ('+-0.000'), so we show the seeds."""
    rows, macros = [], {}
    per_seed_data = {}
    domains = [("webtext", "Web text (FineWeb-Edu)"),
               ("code", "Code (CodeSearchNet-py)"),
               ("math", "Math (MetaMathQA)")]
    for dom, label in domains:
        try:
            bl = [json.load(open(p))["final"] for p in
                  sorted(glob.glob(f"results-nl/{dom}/baseline_seed*.json"))]
            rl = [json.load(open(p))["final"] for p in
                  sorted(glob.glob(f"results-nl/{dom}/rotation_seed*.json"))]
        except (json.JSONDecodeError, OSError):
            continue
        if not bl or not rl:
            continue
        dl = [b["held_loss"] for b in bl]
        gl = [r.get("dense_inference_held_loss") for r in rl]
        gl = [g for g in gl if g is not None]
        if not gl:
            continue
        n = min(len(dl), len(gl))
        gaps = [gl[i] / dl[i] - 1 for i in range(n)]
        fr = json.load(open(f"results-nl/{dom}/rotation_seed0.json"))["final"]
        br = bl[0]
        fr_pct = 100 * fr["cum_train_flops"] / br["cum_train_flops"]

        def per_seed(vals, nd, scale=1.0):
            return "/".join(f"{scale*v:.{nd}f}" for v in vals[:n])
        rows.append(f"{label} & {n} & {per_seed(dl,3)} & {per_seed(gl,3)} & "
                    f"{per_seed(gaps,2,100)} & {fr_pct:.0f} \\\\")
        macros[f"Nl{dom.capitalize()}Dense"] = per_seed(dl, 3)
        macros[f"Nl{dom.capitalize()}Gap"] = per_seed(gaps, 2, 100)
        per_seed_data[dom] = (dl[:n], gl[:n], gaps)
    try:
        wc = json.load(open("results-nl/wallclock.json"))
        r = {x["mode"] + str(x["budget"]).replace(".", ""): x["ratio"]
             for x in wc["rows"]}
        macros["WcDenseTok"] = f"{wc['dense_tok_s']:.0f}"
        macros["WcMaskedHalf"] = f"{r.get('rotation05', 0):.2f}"
        macros["WcSkipHalf"] = f"{r.get('rotation_skip05', 0):.2f}"
        macros["WcSkipQuarter"] = f"{r.get('rotation_skip025', 0):.2f}"
        macros["WcSkipThreeQ"] = f"{r.get('rotation_skip075', 0):.2f}"
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    if not rows:
        return "% NL table pending\n", macros, per_seed_data
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "Domain & $n$ & dense loss & rot.@dense & gap (\\%) & FLOPs (\\%) \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return body, macros, per_seed_data


def stepsmatched_block():
    """Steps-matched view at 150M: every arm at 5000 optimizer steps."""
    rows = [("Dense 5000 steps", "results-p3b/tagged-v2/baseline_seed*.json"),
            ("Stage-1 body @5000 (pipeline's own base)",
             "results-p3b/tagged-v2/predictor-supervised_stage1_150m_seed*.json"),
            ("Ours (tag), full pipeline @7000",
             "results-p3b/tagged-v2/predictor-supervised_150m_seed*.json"),
            ("Shuffled, full pipeline @7000",
             "results-p3b/tagged-v2/shuffled_150m_seed*.json")]
    out = []
    for label, pat in rows:
        runs = []
        for path in sorted(glob.glob(pat)):
            try:
                runs.append(json.load(open(path)))
            except (json.JSONDecodeError, OSError):
                pass
        if not runs:
            continue
        acc = ms([r["final"]["acc"] for r in runs], 100, 1)
        fl = ms([r["final"]["flops_per_token"] for r in runs], 1e-6, 0)
        out.append(f"{label} & {acc} & {fl} \\\\")
    if not out:
        return "% steps-matched pending\n"
    return ("\\begin{tabular}{lcc}\n\\toprule\n"
            "150M arm & Acc & MFLOPs/token \\\\\n\\midrule\n" +
            "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    blocks = {}
    all_macros = {}
    body, macros = p0_block()
    blocks["pzerotable"] = body
    all_macros.update(macros)
    blocks["budgettable"] = budget_block()
    blocks["migrationtable"] = migration_block()
    body, macros = causal_combined_block()
    blocks["causalcombinedtable"] = body
    all_macros.update(macros)
    body, macros = p150_block()
    blocks["pgridtable"] = body
    all_macros.update(macros)
    blocks["pgridperseedtable"] = p150_perseed_block()
    all_macros.update(convergence_macros())
    body, macros, nl_domain = nl_domain_block()
    blocks["nltab"] = body
    all_macros.update(macros)
    all_macros.update(nl_macros(nl_domain))
    blocks["stepsmatchedtable"] = stepsmatched_block()
    with open(OUT, "w") as fh:
        fh.write("% auto-generated by paper/generate_tables_iclr.py — do not edit\n")
        for name, m in sorted(all_macros.items()):
            fh.write(f"\\expandafter\\def\\csname {name}\\endcsname{{{m}}}\n")
        for name, b in blocks.items():
            fh.write(f"\\newcommand{{\\{name}}}[0]{{\n{b}}}\n")
    print(f"wrote {OUT}")
    for name, b in blocks.items():
        has_data = not b.startswith("%")
        print(f"  {name}: {'OK' if has_data else 'PENDING'}")


if __name__ == "__main__":
    main()
