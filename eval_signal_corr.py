"""Per-token signal vs difficulty correlation for any run with a checkpoint.

Reconstructs each run from its saved JSON config, loads the sibling .pt
checkpoint, runs frozen held-out batches, and reports Pearson/Spearman r
between a chosen per-token signal (gate_mean, entropy_mean, diff_score_mean)
and the difficulty bin index. Used for the "did the signal encode difficulty?"
comparison across iterations.

Examples:
  python eval_signal_corr.py --runs "results/tagged/chemical_seed*.json" --signal entropy_mean
  python eval_signal_corr.py --runs "results-opt3b-full/tagged/predictor-supervised_seed*.json" --signal gate_mean
"""
import argparse
import glob
import json
import random

import numpy as np
import torch
from scipy import stats as scipy_stats

from ct.config import Config
from ct.models import build_model
from ct.tasks import get_task
from ct.tasks.base import get_batch
from ct.tokenizer import TOKENIZER


def cfg_from_json(d, device):
    fields = set(Config.__dataclass_fields__)
    c = {k: v for k, v in d["config"].items() if k in fields}
    c["device"] = device
    return Config(**c)


@torch.no_grad()
def corr_for_run(json_path, signal, n_batches, device):
    with open(json_path) as f:
        run = json.load(f)
    cfg = cfg_from_json(run, device)
    task = get_task(run["task"])
    held_out = task.build_held_out(cfg)
    model = build_model(run["method"], cfg).to(device)
    model.load_state_dict(torch.load(json_path.replace(".json", ".pt"),
                                     map_location=device),
                          strict=False)  # tolerate dropped params (e.g. old diff_head.tau)
    model.eval()

    rng = random.Random(999)
    sigs, diffs_all = [], []
    for _ in range(n_batches):
        x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size,
                                     held_out=held_out)
        pad = x != TOKENIZER.pad_id
        _, info = model(x, pad_mask=pad, meta={"diffs": diffs})
        s = info.get(signal)
        if s is None:
            return None
        m = pad & (diffs >= 0)
        sigs.append(s[m].float().cpu().numpy())
        diffs_all.append(diffs[m].float().cpu().numpy())

    s = np.concatenate(sigs)
    d = np.concatenate(diffs_all)
    pear = scipy_stats.pearsonr(s, d).statistic
    spear = scipy_stats.spearmanr(s, d).statistic
    return pear, spear, len(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="glob for run JSONs")
    p.add_argument("--signal", default="gate_mean",
                   choices=["gate_mean", "entropy_mean", "diff_score_mean"])
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    pears = []
    for path in sorted(glob.glob(args.runs)):
        out = corr_for_run(path, args.signal, args.batches, device)
        if out is None:
            print(f"{path}: signal '{args.signal}' not produced, skipped")
            continue
        pear, spear, n = out
        pears.append(pear)
        print(f"{path}: pearson {pear:.4f}  spearman {spear:.4f}  (n={n})")
    if pears:
        print(f"mean pearson r = {np.mean(pears):.4f} over {len(pears)} seeds")


if __name__ == "__main__":
    main()
