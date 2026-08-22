"""Aggregate per-seed JSONs into results/summary.json with mean +/- std and
Welch t-tests of Chemical vs. every other method on final answer accuracy."""
import glob
import json
import os
import sys

from ct.stats import mean_std, welch_ttest

METRICS = ["acc", "balanced_acc", "flops_per_token", "wallclock_per_token",
           "mean_gate", "entropy_pearson", "entropy_spearman"]


def main(outdir="results"):
    summary = {}
    for task_dir in sorted(glob.glob(os.path.join(outdir, "*"))):
        if not os.path.isdir(task_dir):
            continue
        task = os.path.basename(task_dir)
        runs = {}
        for path in sorted(glob.glob(os.path.join(task_dir, "*_seed*.json"))):
            with open(path) as f:
                r = json.load(f)
            runs.setdefault(r["method"], []).append(r)

        tsummary = {}
        chem_acc = [r["final"]["acc"] for r in runs.get("chemical", [])]
        for method, rs in runs.items():
            entry = {"n_seeds": len(rs)}
            for metric in METRICS:
                vals = [r["final"].get(metric) for r in rs]
                m, s = mean_std(vals)
                if m is not None:
                    entry[metric] = {"mean": m, "std": s}
            # per-bin accuracy
            n_bins = len(rs[0]["final"]["bin_acc"])
            entry["bin_acc"] = []
            for b in range(n_bins):
                m, s = mean_std([r["final"]["bin_acc"][b] for r in rs])
                entry["bin_acc"].append({"mean": m, "std": s})
            if method != "chemical" and len(chem_acc) >= 2 and len(rs) >= 2:
                t, p = welch_ttest(chem_acc, [r["final"]["acc"] for r in rs])
                entry["ttest_vs_chemical_acc"] = {"t": t, "p": p}
            tsummary[method] = entry
        summary[task] = tsummary

    out_path = os.path.join(outdir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_path}")

    for task, tsummary in summary.items():
        print(f"\n== {task} ==")
        for method, e in tsummary.items():
            if "acc" not in e:
                continue
            line = (f"{method:15s} acc {e['acc']['mean']:.4f}+-{e['acc']['std']:.4f} "
                    f"bal {e['balanced_acc']['mean']:.4f} "
                    f"MFLOPs/tok {e['flops_per_token']['mean'] / 1e6:.2f}")
            if "ttest_vs_chemical_acc" in e:
                line += f"  p={e['ttest_vs_chemical_acc']['p']:.4f}"
            print(line)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
