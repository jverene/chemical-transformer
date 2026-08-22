"""Per-token gate vs difficulty-target correlation for predictor-supervised runs.

Loads each Stage 2 checkpoint, runs frozen held-out batches, and reports
Pearson r between the mean per-token gate and the difficulty target used in
training (tag targets 0.2/0.5/0.8, or digit targets 0.2/0.4/0.6/0.8 with
--digit-targets), plus per-bin gate means.
"""
import argparse
import random

import numpy as np
import torch
from scipy import stats as scipy_stats

from ct.config import Config
from ct.models import build_model
from ct.tasks import get_task
from ct.tasks.base import get_batch
from ct.tokenizer import TOKENIZER


def make_targets(diffs: torch.Tensor, digit_targets: bool) -> torch.Tensor:
    if digit_targets:
        target = torch.full(diffs.shape, 0.5, device=diffs.device)
        valid = diffs >= 0
        target[valid] = 0.2 + 0.2 * diffs[valid].float()
        return target
    return torch.where(diffs == 0, 0.2,
                       torch.where(diffs == 1, 0.5,
                                   torch.where(diffs == 2, 0.8, 0.5)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="tagged")
    p.add_argument("--outdir", default="results-opt3b-full")
    p.add_argument("--method", default="predictor-supervised")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--digit-targets", action="store_true")
    args = p.parse_args()

    cfg = Config.for_size("small")
    cfg.stage = 2
    task = get_task(args.task)
    held_out = task.build_held_out(cfg)

    for seed in [int(s) for s in args.seeds.split(",")]:
        path = f"{args.outdir}/{args.task}/{args.method}_seed{seed}.pt"
        model = build_model(args.method, cfg).to(cfg.device)
        model.load_state_dict(torch.load(path, map_location=cfg.device))
        model.eval()

        rng = random.Random(999)
        gates, targets = [], []
        for _ in range(args.batches):
            x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size,
                                         held_out=held_out)
            pad = x != TOKENIZER.pad_id
            _, info = model(x, pad_mask=pad, meta={"diffs": diffs})
            g = info["gate_mean"]
            t = make_targets(diffs, args.digit_targets)
            gates.append(g[pad].float().cpu().numpy())
            targets.append(t[pad].float().cpu().numpy())

        g = np.concatenate(gates)
        t = np.concatenate(targets)
        r = scipy_stats.pearsonr(g, t).statistic
        uniq = sorted(set(t.tolist()))
        per_bin = {u: float(g[t == u].mean()) for u in uniq}
        print(f"seed {seed}: pearson r = {r:.4f} (n={len(g)})")
        print(f"  gate by target: " +
              " ".join(f"{u:.1f}:{v:.3f}" for u, v in per_bin.items()))


if __name__ == "__main__":
    main()
