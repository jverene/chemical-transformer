"""Single-stage causal test: does allocation CONTENT matter while the value
field is alive?

Three arms, identical recipe (11M from scratch, gates active from step 0,
hard 0/1 allocation at budget 0.5, recomputed every K steps):
  online   : allocate by the CURRENT body's measured score field
             (waterfill on -sum_l dL/dg, one fwd+bwd per refresh)
  shuffled : allocate by random ordering, same budget
  static   : g = 0.5 everywhere

Trajectory prediction: online < shuffled while the field is alive (early/
mid), converging later. Matched budget, matched steps, matched data order.
"""
import argparse
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
from .train import set_seed, eval_loop


def measure_scores(model, x, y, diffs, cfg):
    leaves = [torch.ones(x.shape, device=x.device, requires_grad=True)
              for _ in range(cfg.n_layers)]
    logits, _ = model(x, pad_mask=x != TOKENIZER.pad_id,
                      meta={"diffs": diffs}, oracle_gates=leaves)
    ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                         ignore_index=TOKENIZER.pad_id, reduction="none")
    ce.mean().backward()
    score = -torch.stack([l.grad for l in leaves], 0).sum(0).detach()
    for l in leaves:
        l.grad = None
    return score * (x != TOKENIZER.pad_id)


def hard_alloc(score, budget, shuffle=False, gen=None):
    k = max(1, int(round(budget * score.shape[1])))
    s = score if not shuffle else torch.rand_like(score)
    _, idx = torch.topk(s, k, dim=1)
    g = torch.zeros_like(score)
    g.scatter_(1, idx, 1.0)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["online", "shuffled", "static"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--refresh-every", type=int, default=100)
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(args.seed)
    cfg = Config(device=device)
    cfg.stage = 1
    task = get_task("tagged")
    held_out = task.build_held_out(cfg)
    model = build_model("predictor-supervised", cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = random.Random(args.seed * 100003 + 7)

    history = {"step": [], "heldout_loss": [], "heldout_acc": [],
               "cum_flops": []}
    dense_flops_tok = 33.2e6  # 11M dense, FLOPsCounter accounting
    cum_flops = 0.0
    g_cur = None
    t0 = __import__("time").time()

    for step in range(1, args.steps + 1):
        x, y, diffs, _ = get_batch(task, cfg, rng, cfg.batch_size)
        pad = x != TOKENIZER.pad_id

        if args.arm == "static":
            g = torch.full(x.shape, args.budget, device=device)
        elif step == 1 or step % args.refresh_every == 0:
            # both adaptive arms refresh at the same cadence
            if args.arm == "online":
                model.eval()
                score = measure_scores(model, x, y, diffs, cfg)
                model.train()
                g = hard_alloc(score, args.budget)
            else:
                g = hard_alloc(torch.rand(x.shape, device=device), args.budget,
                               shuffle=True)
            g_cur = g
        else:
            g = g_cur

        logits, info = model(x, pad_mask=pad, meta={"diffs": diffs},
                             oracle_gates=[g] * cfg.n_layers)
        ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                             ignore_index=TOKENIZER.pad_id)
        opt.zero_grad()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cum_flops += 3.0 * dense_flops_tok * cfg.batch_size * \
            float(g.mean())

        if step % 250 == 0 or step == args.steps:
            m = eval_loop(model, held_out, task, cfg, 4)
            history["step"].append(step)
            history["heldout_loss"].append(m["loss"])
            history["heldout_acc"].append(m["acc"])
            history["cum_flops"].append(cum_flops)
            print(f"[{args.arm}/s{args.seed}] step {step} | loss "
                  f"{float(ce):.4f} | held {m['loss']:.4f} | acc "
                  f"{m['acc']:.3f} | {__import__('time').time()-t0:.0f}s",
                  flush=True)

    out = args.out or f"results-headroom/single_{args.arm}_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump({"arm": args.arm, "seed": args.seed, "steps": args.steps,
                   "budget": args.budget, "history": dict(history)},
                  f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
