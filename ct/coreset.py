"""Paired diagnostic for update-alignment-based token compute allocation.

At a shared checkpoint, candidate hard masks produce sparse body updates on
batch A. Alignment with the dense A update is the selection metric; loss
improvement on an independent batch C is the confirmation metric. At 11M the
full gradient is used directly instead of a random projection.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, get_device
from .models import build_model
from .singletest import measure_scores
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def ce_loss(model, x, y, gate, cfg):
    gates = [gate] * cfg.n_layers if gate is not None else None
    logits, _ = model(x, pad_mask=x != TOKENIZER.pad_id,
                      meta={"diffs": None}, oracle_gates=gates)
    return F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                           ignore_index=TOKENIZER.pad_id)


def budget_mask(score, pad, budget, generator=None, random_mask=False):
    B, S = score.shape
    mask = torch.zeros_like(score)
    for b in range(B):
        n_valid = int(pad[b].sum())
        if n_valid == 0:
            continue
        k = min(n_valid, max(1, int(round(budget * n_valid))))
        if random_mask:
            noise = torch.rand(S, generator=generator).to(score.device)
            ranking = noise.masked_fill(~pad[b], -torch.inf)
        else:
            ranking = score[b].masked_fill(~pad[b], -torch.inf)
        idx = torch.topk(ranking, k).indices
        mask[b, idx] = 1.0
    return mask


def gradient_vector(model):
    grads = [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError("candidate update produced no parameter gradients")
    return torch.cat(grads).detach()


def alignment(candidate, dense):
    cosine = F.cosine_similarity(candidate, dense, dim=0)
    relative_error = (candidate - dense).norm() / dense.norm().clamp_min(1e-12)
    return float(cosine), float(relative_error)


@torch.no_grad()
def confirmation_loss(model, x, y, cfg):
    return float(ce_loss(model, x, y, None, cfg))


def evaluate_candidate(model, initial_state, mask, batch_a, batch_c, dense_grad,
                       cfg, lr, c_before, optimizer_name):
    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)
    x_a, y_a = batch_a
    loss_a = ce_loss(model, x_a, y_a, mask, cfg)
    loss_a.backward()
    grad = gradient_vector(model)
    cosine, relative_error = alignment(grad, dense_grad)
    optimizer_cls = torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
    opt = optimizer_cls(model.parameters(), lr=lr)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    x_c, y_c = batch_c
    c_after = confirmation_loss(model, x_c, y_c, cfg)
    return {
        "cosine": cosine,
        "relative_error": relative_error,
        "train_loss": float(loss_a.detach()),
        "c_loss_before": c_before,
        "c_loss_after": c_after,
        "c_improvement": c_before - c_after,
        "active_fraction": float(mask.sum() / mask.numel()),
    }


def correlation(xs, ys):
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def run_replicate(model, initial_state, task, cfg, rng, generator,
                  args, replicate):
    x_a, y_a, _, _ = get_batch(task, cfg, rng, args.batch_size)
    x_c, y_c, _, _ = get_batch(task, cfg, rng, args.batch_size)
    pad_a = x_a != TOKENIZER.pad_id
    batch_a = (x_a, y_a)
    batch_c = (x_c, y_c)

    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)
    dense_loss = ce_loss(model, x_a, y_a, None, cfg)
    dense_loss.backward()
    dense_grad = gradient_vector(model).clone()
    model.zero_grad(set_to_none=True)
    c_before = confirmation_loss(model, x_c, y_c, cfg)
    gategrad_score = measure_scores(model, x_a, y_a, None, cfg)
    model.zero_grad(set_to_none=True)

    candidates = [("gategrad", -1, budget_mask(
        gategrad_score, pad_a, args.budget, generator=generator))]
    for i in range(args.candidates):
        candidates.append(("random", i, budget_mask(
            gategrad_score, pad_a, args.budget, generator=generator,
            random_mask=True)))
    candidates.append(("dense", -1, torch.ones_like(gategrad_score)))

    records = []
    for kind, index, mask in candidates:
        record = evaluate_candidate(
            model, initial_state, mask, batch_a, batch_c, dense_grad,
            cfg, args.lr, c_before, args.optimizer)
        record.update({"kind": kind, "index": index})
        records.append(record)
        print(f"  r{replicate} {kind:8s} {index:2d} | "
              f"cos {record['cosine']:+.5f} | "
              f"C gain {record['c_improvement']:+.6f}", flush=True)

    random_records = [r for r in records if r["kind"] == "random"]
    selected = max(random_records, key=lambda r: r["cosine"])
    selected["core_selected"] = True
    best_reward = max(random_records, key=lambda r: r["c_improvement"])
    best_reward["best_random_reward"] = True
    return {
        "replicate": replicate,
        "records": records,
        "core_selected_index": selected["index"],
        "core_selected_improvement": selected["c_improvement"],
        "random_mean_improvement": float(np.mean(
            [r["c_improvement"] for r in random_records])),
        "gategrad_improvement": next(
            r["c_improvement"] for r in records if r["kind"] == "gategrad"),
        "dense_improvement": next(
            r["c_improvement"] for r in records if r["kind"] == "dense"),
        "random_alignment_reward_pearson": correlation(
            [r["cosine"] for r in random_records],
            [r["c_improvement"] for r in random_records]),
    }


def summarize(replicates):
    random_gains = []
    selected_gains = []
    gategrad_gains = []
    dense_gains = []
    for replicate in replicates:
        records = replicate["records"]
        random_records = [r for r in records if r["kind"] == "random"]
        random_gains.extend(r["c_improvement"] for r in random_records)
        selected_gains.append(replicate["core_selected_improvement"])
        gategrad_gains.append(replicate["gategrad_improvement"])
        dense_gains.append(replicate["dense_improvement"])
    return {
        "random_mean_c_improvement": float(np.mean(random_gains)),
        "core_selected_mean_c_improvement": float(np.mean(selected_gains)),
        "gategrad_mean_c_improvement": float(np.mean(gategrad_gains)),
        "dense_mean_c_improvement": float(np.mean(dense_gains)),
        "core_minus_random_per_replicate": [
            r["core_selected_improvement"] - r["random_mean_improvement"]
            for r in replicates],
        "core_minus_gategrad_per_replicate": [
            r["core_selected_improvement"] - r["gategrad_improvement"]
            for r in replicates],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="tagged")
    p.add_argument("--size", default="small")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--allow-legacy-extra-keys", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not 0 < args.budget <= 1:
        raise ValueError("--budget must be in (0, 1]")
    if args.candidates < 2 or args.replicates < 1:
        raise ValueError("need at least two random candidates and one replicate")

    device = args.device or get_device()
    set_seed(args.seed)
    cfg = Config.for_size(args.size, device=device, seq_len=args.seq_len)
    args.lr = args.lr or (0.01 if args.optimizer == "sgd" else cfg.lr)
    model = build_model("predictor-supervised", cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if args.allow_legacy_extra_keys:
        incompatible = model.load_state_dict(state, strict=False)
        allowed = {f"layers.{i}.diff_head.tau" for i in range(cfg.n_layers)}
        if incompatible.missing_keys or set(incompatible.unexpected_keys) - allowed:
            raise ValueError(f"incompatible checkpoint: {incompatible}")
    else:
        model.load_state_dict(state)
    model.eval()
    initial_state = {k: v.detach().cpu().clone()
                     for k, v in model.state_dict().items()}
    task = get_task(args.task)
    rng = random.Random(args.seed * 100003 + 177)
    generator = torch.Generator().manual_seed(args.seed * 100003 + 911)

    print(f"CORE diagnostic | {sum(p.numel() for p in model.parameters()):,} params | "
          f"{cfg.n_layers} layers | d={cfg.d_model} | device={device} | "
          f"budget={args.budget}", flush=True)
    replicates = []
    start = time.time()
    for r in range(args.replicates):
        replicates.append(run_replicate(
            model, initial_state, task, cfg, rng, generator, args, r))
    result = {
        "checkpoint": os.path.abspath(args.ckpt),
        "resolved_model": {
            "params": sum(p.numel() for p in model.parameters()),
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "size_name": cfg.size_name,
        },
        "config": vars(args),
        "wallclock_s": time.time() - start,
        "replicates": replicates,
        "summary": summarize(replicates),
    }
    print(json.dumps(result["summary"], indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
