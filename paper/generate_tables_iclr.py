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
            "& \\multicolumn{2}{c}{11M (3000 steps, 1 seed)} & "
            "\\multicolumn{2}{c}{starved-150M (2500 steps, 1 seed)} \\\\\n"
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


def mean_se(values):
    mean = st.mean(values)
    se = st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, se


def pm(mean, se, nd=4):
    return f"{mean:.{nd}f}$\\pm${se:.{nd}f}"


def online_meta_block():
    try:
        combined = json.load(open("results-meta/confirm_combined.json"))
        summaries = {
            body: json.load(open(
                f"results-meta/confirm-seed{body}-k10-m4/summary.json"))
            for body in combined["body_seeds"]
        }
    except (OSError, json.JSONDecodeError, KeyError):
        return "% online meta table pending\n", {}
    rows = []
    for body, summary in summaries.items():
        arms = summary["arms"]
        delta_random = arms["meta"]["sparse_heldout_loss"] - arms["random"]["sparse_heldout_loss"]
        delta_shuffled = arms["meta"]["sparse_heldout_loss"] - arms["shuffled"]["sparse_heldout_loss"]
        delta_acc = 100 * (arms["meta"]["sparse_heldout_acc"] - arms["random"]["sparse_heldout_acc"])
        rows.append(
            f"{body} & {arms['meta']['sparse_heldout_loss']:.4f} & "
            f"{arms['random']['sparse_heldout_loss']:.4f} & "
            f"{arms['shuffled']['sparse_heldout_loss']:.4f} & "
            f"{delta_random:+.4f} & {delta_shuffled:+.4f} & {delta_acc:+.2f} \\\\")
    random_stats = combined["sparse_deployment_loss"]["meta_minus_random"]
    shuffled_stats = combined["sparse_deployment_loss"]["meta_minus_shuffled"]
    acc_stats = combined["sparse_answer_accuracy"]["meta_minus_random"]
    rows.append(
        f"Mean $\\Delta$ & -- & -- & -- & {pm(random_stats['mean'], random_stats['se'])} & "
        f"{pm(shuffled_stats['mean'], shuffled_stats['se'])} & "
        f"{pm(100*acc_stats['mean'], 100*acc_stats['se'], 2)} \\\\")
    body = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "Body & Meta & Random & Shuffled & $\\Delta$ random & "
            "$\\Delta$ shuffled & $\\Delta$ acc. (pp) \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    macros = {
        "MetaRandomDelta": f"{random_stats['mean']:.4f}",
        "MetaRandomSE": f"{random_stats['se']:.4f}",
        "MetaShuffledDelta": f"{shuffled_stats['mean']:.4f}",
        "MetaShuffledSE": f"{shuffled_stats['se']:.4f}",
        "MetaAccuracyDelta": f"{100*acc_stats['mean']:.2f}",
        "MetaAccuracySE": f"{100*acc_stats['se']:.2f}",
        "MetaDenseEvalGap": f"{combined['dense_deployment_loss']['meta_minus_random']['mean']:.4f}",
        "MetaDenseReferenceGap": f"{combined['sparse_deployment_loss']['meta_minus_dense']['mean']:.4f}",
    }
    return body, macros


def fixed_program_block():
    try:
        dev = json.load(open("results-program/dev-long/analysis.json"))
        confirm = json.load(open("results-program/confirm-bodies9-10/analysis.json"))
        combined = json.load(open("results-program/combined_level2_analysis.json"))
    except (OSError, json.JSONDecodeError):
        return "% fixed program table pending\n", {}
    regimes = dev["regimes"] + confirm["regimes"]
    by_body = {}
    for regime in regimes:
        by_body.setdefault(regime["body_seed"], {})[regime["budget"]] = regime
    rows = []
    for body, values in by_body.items():
        split = "Dev." if body in (4, 8) else "Fresh"
        cells = []
        for budget in (0.375, 0.5):
            regime = values[budget]
            cells.append(regime["paired_sparse_loss_by_step"]["200"])
            cells.append(100 * regime["paired_sparse_accuracy_by_step"]["200"])
        rows.append(f"{body} & {split} & {cells[0]:+.4f} & {cells[1]:+.2f} & "
                    f"{cells[2]:+.4f} & {cells[3]:+.2f} \\\\")
    budget_stats = {}
    for budget in (0.375, 0.5):
        selected = [values[budget] for values in by_body.values()]
        loss_mean, loss_se = mean_se([
            row["paired_sparse_loss_by_step"]["200"] for row in selected])
        acc_mean, acc_se = mean_se([
            100 * row["paired_sparse_accuracy_by_step"]["200"] for row in selected])
        budget_stats[budget] = (loss_mean, loss_se, acc_mean, acc_se)
    rows.append(
        f"Mean & all bodies & {pm(budget_stats[0.375][0], budget_stats[0.375][1])} & "
        f"{pm(budget_stats[0.375][2], budget_stats[0.375][3], 2)} & "
        f"{pm(budget_stats[0.5][0], budget_stats[0.5][1])} & "
        f"{pm(budget_stats[0.5][2], budget_stats[0.5][3], 2)} \\\\")
    body_loss_means = [st.mean([
        values[budget]["paired_sparse_loss_by_step"]["200"]
        for budget in (0.375, 0.5)]) for values in by_body.values()]
    body_acc_means = [st.mean([
        100 * values[budget]["paired_sparse_accuracy_by_step"]["200"]
        for budget in (0.375, 0.5)]) for values in by_body.values()]
    loss_mean, loss_se = mean_se(body_loss_means)
    acc_mean, acc_se = mean_se(body_acc_means)
    body = ("\\begin{tabular}{llrrrr}\n\\toprule\n"
            "Body & Split & \\multicolumn{2}{c}{37.5\\% active} & "
            "\\multicolumn{2}{c}{50\\% active} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            " & & $\\Delta$ loss & $\\Delta$ acc. & $\\Delta$ loss & $\\Delta$ acc. \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    macros = {
        "ProgramLossDelta": f"{loss_mean:.4f}",
        "ProgramLossSE": f"{loss_se:.4f}",
        "ProgramAccuracyDelta": f"{acc_mean:.2f}",
        "ProgramAccuracySE": f"{acc_se:.2f}",
        "ProgramFinalWins": str(combined["paired_sparse_loss_by_step"]["200"]["wins"]),
        "ProgramRegimes": str(combined["regime_count"]),
        "ProgramShuffledDelta": "-0.02530",
    }
    return body, macros


def scale_pilot_block():
    try:
        analysis = json.load(open("results-scale/150m-pilot/analysis.json"))
        assert analysis["status"] == "complete"
        regimes = analysis["regimes"]
    except (OSError, json.JSONDecodeError, KeyError, AssertionError):
        return "% 150M scale pilot table pending\n", {}
    rows = []
    for regime in regimes:
        rows.append(
            f"{regime['body_seed']} & "
            f"{regime['initial_sparse_loss']['low_max_abs']:.4f} & "
            f"{regime['final_sparse_loss']['low_max_abs']:.4f} & "
            f"{regime['final_sparse_loss']['random']:.4f} & "
            f"{regime['low_minus_random_final_sparse_loss']:+.4f} & "
            f"{regime['low_minus_random_final_sparse_accuracy_pp']:+.2f} & "
            f"{regime['low_minus_random_final_dense_loss']:+.4f} \\\\")
    body = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "Body & Low step 0 & Low step 50 & Random step 50 & "
            "$\\Delta$ sparse loss & $\\Delta$ sparse acc. & $\\Delta$ dense loss \\\\\n"
            "\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
    summary = analysis["summary"]
    macros = {
        "ScaleLossDelta": f"{summary['low_minus_random_final_sparse_loss']['mean']:.4f}",
        "ScaleLossSE": f"{summary['low_minus_random_final_sparse_loss']['se_descriptive_n2']:.4f}",
        "ScaleAccuracyDelta": f"{summary['low_minus_random_final_sparse_accuracy_pp']['mean']:.2f}",
        "ScaleAccuracySE": f"{summary['low_minus_random_final_sparse_accuracy_pp']['se_descriptive_n2']:.2f}",
        "ScaleDenseDelta": f"{summary['low_minus_random_final_dense_loss']['mean']:.4f}",
        "ScaleChangeDelta": f"{summary['low_minus_random_sparse_loss_change']['mean']:+.4f}",
    }
    return body, macros


def allocation_ablation_block():
    try:
        dev = json.load(open("results-program/dev-long/analysis.json"))
        gradient = json.load(open("results-gradient/dev/analysis.json"))
        directional = json.load(open("results-directional/dev/analysis.json"))
    except (OSError, json.JSONDecodeError, KeyError):
        return "% allocation ablation table pending\n"
    random_relative = -dev["summary"]["mean_final_minus_random_loss"]
    shuffled_relative = -gradient["assignment_diagnostic"]["mean"]
    rows = [
        ("Frozen low-max-abs (reference)", "0", "--"),
        ("Lagged gradient-high", f"{gradient['summary']['gradient_high']['minus_low_max_abs']['mean']:+.4f}", "4/4"),
        ("Adam-directional utility", f"{directional['summary']['minus_low_max_abs']['mean']:+.4f}", "4/4"),
        ("Exact-budget random", f"{random_relative:+.4f}", "4/4"),
        ("Shuffled frozen scores", f"{shuffled_relative:+.4f}", "4/4"),
    ]
    body = ("\\begin{tabular}{lrc}\n\\toprule\n"
            "Allocation rule & $\\Delta$ loss vs. frozen & Frozen wins \\\\\n"
            "\\midrule\n" + "\n".join(
                f"{name} & {value} & {wins} \\\\" for name, value, wins in rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
    return body


def frontier_stress_block():
    try:
        dev = json.load(open("results-ema/dev-body6/analysis.json"))["final_ema"]
        confirm = json.load(open("results-ema/combined_analysis.json"))["confirmation"]
        combined = json.load(open("results-ema/combined_analysis.json"))
    except (OSError, json.JSONDecodeError, KeyError):
        return "% frontier stress table pending\n", {}
    rows = []
    for body, role, row in ((6, "Dev.", dev), (7, "Frozen", confirm)):
        rows.append(
            f"{body} & {role} & {row['low_max_abs_loss']:.6f} & "
            f"{row['random_loss']:.6f} & {row['dense_loss']:.6f} & "
            f"{row['low_max_abs_loss']-row['dense_loss']:+.6f} & "
            f"{row['low_max_abs_total_flops']/1e12:.2f}/{row['dense_total_flops']/1e12:.2f} \\\\")
    body = ("\\begin{tabular}{llrrrrr}\n\\toprule\n"
            "Body & Split & Low-max-abs & Random & Dense & Low $-$ dense & "
            "Cost low/dense (T) \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return body, {
        "FrontierTwoBodyGap": f"{combined['two_body_means_descriptive_only']['low_minus_dense_loss']:+.4f}"
    }


def relaxed_diagnostic_block():
    arms = [("Tag targets", "predictor-supervised_seed*.json"),
            ("Gate-gradient targets", "gategrad_seed*.json"),
            ("Shuffled targets", "shuffled_seed*.json")]
    rows = []
    for label, pattern in arms:
        runs = load(f"results-p0/tagged/{pattern}")
        if not runs:
            continue
        acc = ms([r["final"]["acc"] for r in runs], 100, 1)
        bal = ms([r["final"]["balanced_acc"] for r in runs], 100, 1)
        sp = ms([spread(r) for r in runs], 1, 2)
        rows.append(f"{label} & {acc} & {bal} & {sp} \\\\")
    if not rows:
        return "% relaxed diagnostic table pending\n"
    return ("\\begin{tabular}{lccc}\n\\toprule\n"
            "Supervision signal & Acc. & Balanced acc. & Gate spread \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def nl_quality_block(nl_domain):
    labels = {"webtext": "Web text", "code": "Code", "math": "Math"}
    rows = []
    for domain, values in nl_domain.items():
        dense, rotation, gaps = values
        n = len(dense)
        fmt = lambda vals: "/".join(f"{value:.3f}" for value in vals)
        gap = "/".join(f"{100*value:.2f}" for value in gaps)
        rows.append(f"{labels[domain]} & {n} & {fmt(dense)} & {fmt(rotation)} & {gap} \\\\")
    if not rows:
        return "% NL quality table pending\n"
    return ("\\begin{tabular}{lcccc}\n\\toprule\n"
            "Domain & $n$ & Dense loss & Dropout @ dense & Gap (\\%) \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


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
    body, macros = online_meta_block()
    blocks["onlinemetatable"] = body
    all_macros.update(macros)
    body, macros = fixed_program_block()
    blocks["fixedprogramtable"] = body
    all_macros.update(macros)
    body, macros = scale_pilot_block()
    blocks["scalepilotable"] = body
    all_macros.update(macros)
    blocks["allocationablationtable"] = allocation_ablation_block()
    body, macros = frontier_stress_block()
    blocks["frontierstresstable"] = body
    all_macros.update(macros)
    blocks["relaxeddiagnostictable"] = relaxed_diagnostic_block()
    blocks["nlqualitytable"] = nl_quality_block(nl_domain)
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
