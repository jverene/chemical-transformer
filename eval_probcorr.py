"""Problem-level gate vs difficulty correlation for predictor-supervised runs.

Each packed sequence contains many problems; every problem ends with an EOS
token, so we split sequences at EOS to get exact problem spans. The mean gate
over a problem's span is one allocation decision; we correlate it with the
difficulty target (0.2/0.5/0.8 tagged, or digit targets with --digit-targets).
Also reports the per-token correlation for reference.
"""
import argparse
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
from eval_gatecorr import make_targets
from eval_signal_corr import cfg_from_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="tagged")
    p.add_argument("--outdir", default="results-opt3b-full")
    p.add_argument("--method", default="predictor-supervised")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--digit-targets", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    task = get_task(args.task)

    for seed in [int(s) for s in args.seeds.split(",")]:
        json_path = f"{args.outdir}/{args.task}/{args.method}_seed{seed}.json"
        with open(json_path) as f:
            run = json.load(f)
        cfg = cfg_from_json(run, device)
        if args.method in ("predictor", "predictor-supervised"):
            cfg.stage = 2
        held_out = task.build_held_out(cfg)
        path = json_path.replace(".json", ".pt")
        model = build_model(args.method, cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device),
                              strict=False)  # tolerate dropped params (old tau)
        model.eval()

        rng = random.Random(999)
        prob_g, prob_t, tok_g, tok_t = [], [], [], []
        for _ in range(args.batches):
            x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size,
                                         held_out=held_out)
            pad = x != TOKENIZER.pad_id
            _, info = model(x, pad_mask=pad, meta={"diffs": diffs})
            g = info["gate_mean"].float().cpu().numpy()
            t = make_targets(diffs, args.digit_targets).cpu().numpy()
            d = diffs.cpu().numpy()
            ids = x.cpu().numpy()
            pr = pad.cpu().numpy()
            tok_g.append(g[pr])
            tok_t.append(t[pr])
            for row in range(g.shape[0]):
                start = None
                for i in range(g.shape[1]):
                    if not pr[row, i]:
                        break
                    if start is None:
                        if d[row, i] >= 0:
                            start = i
                        continue
                    if ids[row, i] == TOKENIZER.eos_id:
                        prob_g.append(g[row, start:i + 1].mean())
                        prob_t.append(t[row, start])
                        start = None

        r_tok = scipy_stats.pearsonr(np.concatenate(tok_g),
                                     np.concatenate(tok_t)).statistic
        r_prob = scipy_stats.pearsonr(prob_g, prob_t).statistic
        print(f"seed {seed}: per-problem r = {r_prob:.4f} (n={len(prob_g)})  "
              f"per-token r = {r_tok:.4f}")


if __name__ == "__main__":
    main()
