"""Headroom trajectory: does per-token compute value structure exist early
in training and decay at convergence? (Figure 1 candidate.)

Trains the 11M stage-1 body with the standard recipe, saving checkpoints at
log-spaced steps; at each checkpoint measures the PERMUTATION-CONTROLLED
ordering gap (waterfill vs random vs anti, all hard 0/1 at budget 0.5) plus
uniform, on held-out batches. Zero local cost (MPS).

Pre-registered read: ordering gap ~ 0 at all steps -> flat field throughout;
ordering gap nonzero early -> structure exists and training consumes it.
"""
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .models import build_model
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed
from .headroom import (alloc_from_scores, random_alloc, eval_alloc,
                       uniform_alloc)

CKPT_STEPS = [50, 100, 200, 400, 800, 1600, 3200, 5000]


def measure(oracle, task, cfg, budget, batches, gen):
    res = {k: [] for k in ("uniform", "waterfill", "random", "anti")}
    rng = random.Random(777)
    for _ in range(batches):
        x, y, diffs, _ = get_batch(task, cfg, rng, cfg.batch_size)
        pad = x != TOKENIZER.pad_id
        leaves = [torch.ones(x.shape, device=x.device, requires_grad=True)
                  for _ in range(cfg.n_layers)]
        logits, _ = oracle(x, pad_mask=pad, meta={"diffs": diffs},
                           oracle_gates=leaves)
        ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                             ignore_index=TOKENIZER.pad_id, reduction="none")
        ce.mean().backward()
        score = -torch.stack([l.grad for l in leaves], 0).sum(0).detach()
        score = score * pad + torch.rand_like(score) * 1e-12
        for l in leaves:
            l.grad = None
        allocs = {
            "uniform": uniform_alloc(x.shape, budget, x.device),
            "waterfill": alloc_from_scores(score, budget),
            "random": random_alloc(x.shape, budget, x.device, gen),
            "anti": alloc_from_scores(score, budget, invert=True),
        }
        for name, g in allocs.items():
            res[name].append(eval_alloc(oracle, x, y, diffs, g, cfg))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(args.seed)
    cfg = Config(device=device)
    cfg.stage = 1
    task = get_task("tagged")
    # chemical arch, stage-1 mode: gate=1 everywhere (dense-equivalent) and
    # the state dict loads straight into the measurement oracle.
    model = build_model("predictor-supervised", cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = random.Random(args.seed * 100003 + 7)
    gen = torch.Generator().manual_seed(0)

    traj = []
    saved = set()
    os_dir = "results-headroom"
    import os
    os.makedirs(os_dir, exist_ok=True)
    oracle = build_model("predictor-supervised", cfg).to(device)
    for p in oracle.parameters():
        p.requires_grad_(False)

    def snapshot(step):
        oracle.load_state_dict(model.state_dict())
        oracle.eval()
        m = measure(oracle, task, cfg, args.budget, args.batches, gen)
        # dense (g=1) loss for the loss-based x-axis
        rng_d = random.Random(31337)
        dl = []
        was_train = model.training
        model.eval()
        with torch.no_grad():
            for _ in range(args.batches):
                x, y, diffs, _ = get_batch(task, cfg, rng_d, cfg.batch_size)
                logits, _ = model(x, pad_mask=x != TOKENIZER.pad_id,
                                  meta={"diffs": diffs})
                dl.append(float(F.cross_entropy(
                    logits.view(-1, cfg.vocab_size), y.view(-1),
                    ignore_index=TOKENIZER.pad_id)))
        model.train(was_train)
        m["step"] = step
        m["dense_loss"] = float(np.mean(dl))
        traj.append(m)
        gap_wf = m["waterfill"] - m["random"]
        gap_anti = m["anti"] - m["waterfill"]
        print(f"[step {step:5d}] uniform {m['uniform']:.4f}  "
              f"waterfill {m['waterfill']:.4f}  random {m['random']:.4f}  "
              f"anti {m['anti']:.4f}  | ordering gap (wf-rand) {gap_wf:+.4f}  "
              f"(anti-wf) {gap_anti:+.4f}", flush=True)

    for step in range(1, args.steps + 1):
        model.train()
        x, y, diffs, _ = get_batch(task, cfg, rng, cfg.batch_size)
        pad = x != TOKENIZER.pad_id
        logits, _ = model(x, pad_mask=pad, meta={"diffs": diffs})
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                               ignore_index=TOKENIZER.pad_id)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in CKPT_STEPS and step not in saved:
            saved.add(step)
            snapshot(step)

    args.out = args.out or f"results-headroom/trajectory_seed{args.seed}.json"
    with open(args.out, "w") as f:
        json.dump({"seed": args.seed, "budget": args.budget,
                   "trajectory": traj}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
