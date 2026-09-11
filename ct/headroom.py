"""Allocation-headroom measurement (the keystone diagnostic).

On a frozen body, compare REALIZED held-out loss under different gate
allocations at the same mean-gate budget:
  uniform   : g = budget everywhere (the static schedule)
  waterfill : gates sorted by measured score (-sum_l dL/dg), top-k style
  random    : gates sorted by random noise, same budget (noise control)
  anti      : water-filling INVERTED (compute to the least-valuable tokens)

If waterfill beats uniform and random beats nothing -> the value field has
realizable structure. If all four tie -> end-state headroom is ~0 and
target-content irrelevance is explained: no allocation can beat static on
this body, so training-time differences must act through dynamics, not
through end-state value.

All allocations are evaluated zero-shot via oracle gate leaves (no training).
Run on any checkpoint: python -m ct.headroom --ckpt results-p0/.../...stage1_seed0.pt
"""
import argparse
import random
import json

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, get_device
from .gategrad import build_oracle
from .models import build_model
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER


def alloc_from_scores(score, budget, invert=False):
    """Binary top-k gate pattern; mean gate == budget exactly (k = b*S at 1)."""
    k = max(1, int(round(budget * score.shape[1])))
    s = -score if invert else score  # invert -> allocate to least valuable
    _, idx = torch.topk(s, k, dim=1)
    g = torch.zeros_like(score)
    g.scatter_(1, idx, 1.0)
    return g


def uniform_alloc(shape, budget, device):
    return torch.full(shape, budget, device=device)


def random_alloc(shape, budget, device, gen):
    k = max(1, int(round(budget * shape[1])))
    g = torch.zeros(shape, device=device)
    noise = torch.rand(shape, generator=gen).cpu().to(device)
    idx = torch.argsort(noise, dim=1)[:, :k]
    g.scatter_(1, idx, 1.0)
    return g


@torch.no_grad()
def eval_alloc(oracle, x, y, diffs, gate, cfg):
    """Held-out loss with the body's gates hard-set to `gate` (no grad)."""
    logits, _ = oracle(x, pad_mask=x != TOKENIZER.pad_id,
                      meta={"diffs": diffs}, oracle_gates=[gate] * cfg.n_layers)
    ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                         ignore_index=TOKENIZER.pad_id, reduction="none")
    pad = x != TOKENIZER.pad_id
    return float(ce.view_as(y)[pad].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="tagged")
    p.add_argument("--size", default="small")
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = args.device or get_device()

    cfg = Config.for_size(args.size, device=device, seq_len=256)
    oracle = build_oracle(args.ckpt, cfg)

    task = get_task(args.task)
    rng = random.Random(777)
    results = {k: [] for k in ("uniform", "waterfill", "random", "anti")}
    gen = torch.Generator().manual_seed(0)  # CPU generator; noise moved to device

    for _ in range(args.batches):
        x, y, diffs, _ = get_batch(task, cfg, rng, cfg.batch_size)
        pad = x != TOKENIZER.pad_id
        # measure per-token value on this batch (one fwd+bwd, gate leaves)
        leaves = [torch.ones(x.shape, device=device, requires_grad=True)
                  for _ in range(cfg.n_layers)]
        logits, _ = oracle(x, pad_mask=pad, meta={"diffs": diffs},
                           oracle_gates=leaves)
        ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                             ignore_index=TOKENIZER.pad_id, reduction="none")
        ce.mean().backward()
        score = -torch.stack([l.grad for l in leaves], 0).sum(0).detach()
        score = score * pad  # ignore padding positions
        for l in leaves:
            l.grad = None

        B, S = x.shape
        real_score = score + (torch.rand_like(score) * 1e-12)  # tie-break
        allocs = {
            "uniform": uniform_alloc((B, S), args.budget, device),
            "waterfill": alloc_from_scores(real_score, args.budget),
            "random": random_alloc((B, S), args.budget, device, gen),
            "anti": alloc_from_scores(real_score, args.budget, invert=True),
        }
        for name, g in allocs.items():
            results[name].append(eval_alloc(oracle, x, y, diffs, g, cfg))

    out = {k: float(np.mean(v)) for k, v in results.items()}
    out["headroom_vs_uniform"] = out["uniform"] - out["waterfill"]
    out["noise_gap"] = out["random"] - out["waterfill"]
    out["anti_penalty"] = out["anti"] - out["waterfill"]
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
