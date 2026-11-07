"""One-step true-model inverse allocation with held-out confirmation."""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from .config import Config, get_device
from .coreset import alignment, budget_mask
from .models import build_model
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def model_state(model):
    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


def forward_loss(model, state, batch, gates, cfg):
    x, y = batch
    kwargs = {"pad_mask": x != TOKENIZER.pad_id, "meta": {"diffs": None}}
    if gates is not None:
        kwargs["oracle_gates"] = gates
    logits, _ = functional_call(model, state, (x,), kwargs)
    return F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                           ignore_index=TOKENIZER.pad_id)


def body_gradients(model, state, batch, gates, cfg, create_graph=False):
    params = dict(model.named_parameters())
    loss = forward_loss(model, state, batch, gates, cfg)
    grads = torch.autograd.grad(
        loss, tuple(params.values()), create_graph=create_graph,
        allow_unused=True)
    return loss, dict(zip(params, grads))


def virtual_update(model, state, batch_a, gates, cfg, lr, create_graph=False):
    train_loss, grads = body_gradients(
        model, state, batch_a, gates, cfg, create_graph=create_graph)
    updated = dict(state)
    for name, param in model.named_parameters():
        grad = grads[name]
        if grad is not None:
            updated[name] = param - lr * grad
    return updated, train_loss


def mean_dense_loss(model, state, batches, cfg):
    if not isinstance(batches, list):
        batches = [batches]
    losses = [forward_loss(model, state, batch, None, cfg) for batch in batches]
    return torch.stack(losses).mean()


def inverse_objective(model, state, batch_a, batch_c_opt, gates, cfg, lr,
                      create_graph=True):
    updated, train_loss = virtual_update(
        model, state, batch_a, gates, cfg, lr, create_graph=create_graph)
    q = mean_dense_loss(model, updated, batch_c_opt, cfg)
    return q, train_loss


def sequence_budget(pad_row, budget, n_layers):
    n_valid = int(pad_row.sum())
    if n_valid == 0:
        return 0
    k = min(n_valid, max(1, int(round(budget * n_valid))))
    return n_layers * k


@torch.no_grad()
def project_gates(z, pad, budget):
    out = torch.zeros_like(z)
    n_layers = z.shape[0]
    for b in range(z.shape[1]):
        valid = pad[b].unsqueeze(0).expand(n_layers, -1)
        values = z[:, b, :][valid]
        target = sequence_budget(pad[b], budget, n_layers)
        if values.numel() == 0:
            continue
        lo = (values - 1).min()
        hi = values.max()
        for _ in range(60):
            mid = (lo + hi) / 2
            if (values + mid).clamp(0, 1).sum() > target:
                hi = mid
            else:
                lo = mid
        projected = (values + (lo + hi) / 2).clamp(0, 1)
        out[:, b, :][valid] = projected
    return out


@torch.no_grad()
def initialize_gates(shape, pad, budget, initialization, device, generator):
    if initialization == "random":
        z = torch.rand(shape, generator=generator).to(device)
    else:
        z = torch.zeros(shape, device=device)
        for b in range(shape[1]):
            n_valid = int(pad[b].sum())
            if n_valid:
                target = sequence_budget(pad[b], budget, shape[0])
                z[:, b, :][pad[b].unsqueeze(0).expand(shape[0], -1)] = \
                    target / (shape[0] * n_valid)
    return project_gates(z, pad, budget).detach().requires_grad_(True)


@torch.no_grad()
def harden_gates(gates, pad, budget):
    return scores_to_layer_mask(gates.detach(), pad, budget)


@torch.no_grad()
def scores_to_layer_mask(score, pad, budget, generator=None, random_mask=False):
    n_layers, batch_size, seq_len = score.shape
    out = torch.zeros_like(score)
    for b in range(batch_size):
        target = sequence_budget(pad[b], budget, n_layers)
        if target == 0:
            continue
        valid = pad[b].unsqueeze(0).expand(n_layers, -1)
        if random_mask:
            ranking = torch.rand(
                (n_layers, seq_len), generator=generator).to(score.device)
        else:
            ranking = score[:, b, :]
        ranking = ranking.masked_fill(~valid, -torch.inf)
        indices = torch.topk(ranking.reshape(-1), target).indices
        mask_b = torch.zeros(
            (n_layers, seq_len), dtype=score.dtype, device=score.device)
        mask_b.view(-1)[indices] = 1.0
        out[:, b, :] = mask_b
    return out


def first_order_scores(model, state, batch_a, cfg):
    x, _ = batch_a
    dtype = next(model.parameters()).dtype
    leaves = [torch.ones(x.shape, device=x.device, dtype=dtype,
                         requires_grad=True) for _ in range(cfg.n_layers)]
    loss = forward_loss(model, state, batch_a, leaves, cfg)
    grads = torch.autograd.grad(loss, leaves)
    return -torch.stack(grads).detach()


def flat_gradient(grads):
    values = [grad.reshape(-1) for grad in grads.values() if grad is not None]
    if not values:
        raise RuntimeError("candidate update produced no parameter gradients")
    return torch.cat(values).detach()


@torch.no_grad()
def dense_batch_losses(model, state, batches, cfg):
    if not isinstance(batches, list):
        batches = [batches]
    return [float(forward_loss(model, state, batch, None, cfg))
            for batch in batches]


def dense_batch_loss(model, state, batches, cfg):
    values = dense_batch_losses(model, state, batches, cfg)
    return float(np.mean(values))


def evaluate_candidate(model, state, mask, batch_a, batch_c_opt, batch_c_hold,
                       dense_grad, cfg, lr, before_opt, before_hold):
    gates = None if mask is None else [mask[i] for i in range(mask.shape[0])]
    train_loss, grads = body_gradients(
        model, state, batch_a, gates, cfg, create_graph=False)
    updated = dict(state)
    for name, param in model.named_parameters():
        grad = grads[name]
        if grad is not None:
            updated[name] = param - lr * grad
    cosine, relative_error = alignment(flat_gradient(grads), dense_grad)
    after_opt_batches = dense_batch_losses(model, updated, batch_c_opt, cfg)
    after_hold_batches = dense_batch_losses(model, updated, batch_c_hold, cfg)
    after_opt = float(np.mean(after_opt_batches))
    after_hold = float(np.mean(after_hold_batches))
    valid = batch_a[0] != TOKENIZER.pad_id
    active_fraction = 1.0 if mask is None else float(
        mask.sum() / (mask.shape[0] * valid.sum()))
    return {
        "train_loss": float(train_loss.detach()),
        "cosine": cosine,
        "relative_error": relative_error,
        "c_opt_loss_after": after_opt,
        "c_opt_loss_after_batches": after_opt_batches,
        "c_opt_improvement": before_opt - after_opt,
        "c_hold_loss_after": after_hold,
        "c_hold_loss_after_batches": after_hold_batches,
        "c_hold_improvement": before_hold - after_hold,
        "active_fraction": active_fraction,
    }


def optimize_gates(model, state, batch_a, batch_c_opt, pad, cfg, body_lr,
                   gate_lr, gate_steps, budget, initialization, generator):
    gates = initialize_gates(
        (cfg.n_layers, *batch_a[0].shape), pad, budget, initialization,
        batch_a[0].device, generator)
    history = []
    for step in range(gate_steps + 1):
        q, train_loss = inverse_objective(
            model, state, batch_a, batch_c_opt, gates, cfg, body_lr,
            create_graph=True)
        record = {
            "step": step,
            "q": float(q.detach()),
            "train_loss": float(train_loss.detach()),
        }
        if step < gate_steps:
            hypergrad = torch.autograd.grad(q, gates)[0]
            record["hypergrad_norm"] = float(hypergrad.norm())
            if not torch.isfinite(hypergrad).all():
                raise RuntimeError("non-finite inverse hypergradient")
            gates = project_gates(
                gates - gate_lr * hypergrad, pad, budget).detach()
            gates.requires_grad_(True)
        history.append(record)
    return gates.detach(), history


def correlation(xs, ys):
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1 else None,
        "positive": int((values > 0).sum()),
        "n": int(len(values)),
    }


def run_replicate(model, state, task, cfg, rng, generator, args, replicate):
    batch_a = get_batch(task, cfg, rng, args.batch_size)[:2]
    batch_c_opt = [get_batch(task, cfg, rng, args.batch_size)[:2]
                   for _ in range(args.c_opt_batches)]
    batch_c_hold = [get_batch(task, cfg, rng, args.batch_size)[:2]
                    for _ in range(args.c_hold_batches)]
    pad = batch_a[0] != TOKENIZER.pad_id
    before_opt_batches = dense_batch_losses(model, state, batch_c_opt, cfg)
    before_hold_batches = dense_batch_losses(model, state, batch_c_hold, cfg)
    before_opt = float(np.mean(before_opt_batches))
    before_hold = float(np.mean(before_hold_batches))

    _, dense_grads = body_gradients(
        model, state, batch_a, None, cfg, create_graph=False)
    dense_grad = flat_gradient(dense_grads).clone()
    layer_scores = first_order_scores(model, state, batch_a, cfg)
    shared_scores = layer_scores.sum(dim=0)
    shared_mask = budget_mask(shared_scores, pad, args.budget)
    shared_mask = shared_mask.unsqueeze(0).expand(cfg.n_layers, -1, -1).contiguous()
    layer_mask = scores_to_layer_mask(layer_scores, pad, args.budget)

    starts = []
    for start in range(args.multistart):
        soft_gates, history = optimize_gates(
            model, state, batch_a, batch_c_opt, pad, cfg, args.lr,
            args.gate_lr, args.gate_steps, args.budget, args.initialization,
            generator)
        starts.append({"start": start, "history": history, "gates": soft_gates})
        print(f"  r{replicate} inverse start {start} | "
              f"Q {history[0]['q']:.6f} -> {history[-1]['q']:.6f}", flush=True)
    selected_start = min(starts, key=lambda r: r["history"][-1]["q"])
    soft_gates = selected_start["gates"]
    inverse_hard_mask = harden_gates(soft_gates, pad, args.budget)

    candidates = [
        ("gategrad_shared", -1, shared_mask),
        ("gategrad_layer", -1, layer_mask),
        ("dense", -1, None),
        ("inverse_soft", selected_start["start"], soft_gates),
        ("inverse_hard", selected_start["start"], inverse_hard_mask),
    ]
    for i in range(args.candidates):
        candidates.append(("random", i, scores_to_layer_mask(
            layer_scores, pad, args.budget, generator=generator,
            random_mask=True)))

    records = []
    for kind, index, mask in candidates:
        record = evaluate_candidate(
            model, state, mask, batch_a, batch_c_opt, batch_c_hold,
            dense_grad, cfg, args.lr, before_opt, before_hold)
        record.update({"kind": kind, "index": index})
        records.append(record)
        print(f"  r{replicate} {kind:15s} {index:2d} | "
              f"C-opt {record['c_opt_improvement']:+.6f} | "
              f"C-hold {record['c_hold_improvement']:+.6f}", flush=True)

    random_records = [r for r in records if r["kind"] == "random"]
    core = max(random_records, key=lambda r: r["cosine"])
    best_random = max(random_records, key=lambda r: r["c_opt_improvement"])
    core["core_selected"] = True
    best_random["best_of_k_selected"] = True
    inverse_soft = next(r for r in records if r["kind"] == "inverse_soft")
    q_difference = abs(
        selected_start["history"][-1]["q"] - inverse_soft["c_opt_loss_after"])
    if q_difference > 2e-5:
        raise RuntimeError(f"inverse objective/evaluation mismatch: {q_difference}")

    def one(kind):
        return next(r for r in records if r["kind"] == kind)

    return {
        "replicate": replicate,
        "c_opt_loss_before": before_opt,
        "c_opt_loss_before_batches": before_opt_batches,
        "c_hold_loss_before": before_hold,
        "c_hold_loss_before_batches": before_hold_batches,
        "inverse_starts": [
            {"start": r["start"], "history": r["history"]} for r in starts],
        "selected_start": selected_start["start"],
        "records": records,
        "random_mean_c_hold_improvement": float(np.mean(
            [r["c_hold_improvement"] for r in random_records])),
        "core_selected_c_hold_improvement": core["c_hold_improvement"],
        "best_of_k_random_c_hold_improvement": best_random["c_hold_improvement"],
        "random_alignment_c_hold_pearson": correlation(
            [r["cosine"] for r in random_records],
            [r["c_hold_improvement"] for r in random_records]),
        "inverse_q_improvement": (
            selected_start["history"][0]["q"]
            - selected_start["history"][-1]["q"]),
        "inverse_hard_minus_shared": (
            one("inverse_hard")["c_hold_improvement"]
            - one("gategrad_shared")["c_hold_improvement"]),
        "inverse_hard_minus_layer": (
            one("inverse_hard")["c_hold_improvement"]
            - one("gategrad_layer")["c_hold_improvement"]),
        "inverse_hard_minus_best_of_k": (
            one("inverse_hard")["c_hold_improvement"]
            - best_random["c_hold_improvement"]),
        "search_evaluations": {
            "inverse_c_opt_q_per_start": args.gate_steps + 1,
            "inverse_c_opt_forward_passes_per_start": (
                (args.gate_steps + 1) * args.c_opt_batches),
            "inverse_hypergradients_per_start": args.gate_steps,
            "inverse_starts": args.multistart,
            "best_of_k_random_candidates": args.candidates,
            "c_opt_batches_per_candidate": args.c_opt_batches,
            "c_hold_batches_per_candidate": args.c_hold_batches,
        },
    }


def summarize(replicates):
    kinds = ["gategrad_shared", "gategrad_layer", "inverse_soft",
             "inverse_hard", "dense"]
    summary = {}
    for kind in kinds:
        values = [next(r["c_hold_improvement"] for r in rep["records"]
                       if r["kind"] == kind) for rep in replicates]
        summary[f"{kind}_c_hold"] = mean_se(values)
        summary[f"{kind}_c_hold_per_replicate"] = values
    for name, key in [
        ("random_mean", "random_mean_c_hold_improvement"),
        ("core_selected", "core_selected_c_hold_improvement"),
        ("best_of_k_random", "best_of_k_random_c_hold_improvement"),
    ]:
        values = [r[key] for r in replicates]
        summary[f"{name}_c_hold"] = mean_se(values)
        summary[f"{name}_c_hold_per_replicate"] = values
    for name in ["inverse_q_improvement", "inverse_hard_minus_shared",
                 "inverse_hard_minus_layer", "inverse_hard_minus_best_of_k"]:
        values = [r[name] for r in replicates]
        summary[name] = mean_se(values)
        summary[f"{name}_per_replicate"] = values
    inverse = summary["inverse_hard_c_hold"]["mean"]
    summary["positive_mean_branch"] = {
        "inverse_hard_positive_and_beats_both_controls": bool(
            inverse > 0
            and summary["inverse_hard_minus_layer"]["mean"] > 0
            and summary["inverse_hard_minus_best_of_k"]["mean"] > 0),
        "per_layer_first_order_positive": bool(
            summary["gategrad_layer_c_hold"]["mean"] > 0),
        "best_of_k_random_positive": bool(
            summary["best_of_k_random_c_hold"]["mean"] > 0),
        "all_inverse_q_decreased": bool(all(
            r["inverse_q_improvement"] > 0 for r in replicates)),
    }
    return summary


def load_model(args):
    device = args.device or get_device()
    overrides = {}
    if getattr(args, "held_out_seed", None) is not None:
        overrides["held_out_seed"] = args.held_out_seed
    cfg = Config.for_size(
        args.size, device=device, seq_len=args.seq_len, dropout=0.0, **overrides)
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
    return model, cfg


def self_test():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(812)
    cfg = Config.for_size(
        "small", device="cpu", d_model=16, n_layers=2, n_heads=4, d_ff=32,
        max_len=16, seq_len=6, dropout=0.0).to_dict()
    cfg = Config(**cfg)
    model = build_model("predictor-supervised", cfg).double().eval()
    state = model_state(model)
    before = {name: value.detach().clone() for name, value in state.items()}
    x = torch.randint(0, 19, (1, cfg.seq_len))
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = TOKENIZER.pad_id
    c = torch.randint(0, 19, (1, cfg.seq_len))
    cy = torch.roll(c, -1, dims=1)
    cy[:, -1] = TOKENIZER.pad_id
    c2 = torch.randint(0, 19, (1, cfg.seq_len))
    c2y = torch.roll(c2, -1, dims=1)
    c2y[:, -1] = TOKENIZER.pad_id
    reward_batches = [(c, cy), (c2, c2y)]
    pad = x != TOKENIZER.pad_id
    generator = torch.Generator().manual_seed(813)
    gates = initialize_gates(
        (cfg.n_layers, *x.shape), pad, 0.5, "uniform", "cpu", generator)
    direction = torch.zeros_like(gates)
    direction[0, 0, 0] = 1.0
    direction[1, 0, 0] = -1.0
    epsilon = 1e-4
    body_lr = 1e-3
    q, _ = inverse_objective(
        model, state, (x, y), reward_batches, gates, cfg, body_lr,
        create_graph=True)
    analytic = float(torch.autograd.grad(q, gates)[0].mul(direction).sum())
    q_plus, _ = inverse_objective(
        model, state, (x, y), reward_batches, gates + epsilon * direction,
        cfg, body_lr, create_graph=True)
    q_minus, _ = inverse_objective(
        model, state, (x, y), reward_batches, gates - epsilon * direction,
        cfg, body_lr, create_graph=True)
    numeric = float(((q_plus - q_minus) / (2 * epsilon)).detach())
    relative_error = abs(analytic - numeric) / max(abs(numeric), 1e-12)
    single, _ = inverse_objective(
        model, state, (x, y), (c, cy), gates, cfg, body_lr,
        create_graph=True)
    single_list, _ = inverse_objective(
        model, state, (x, y), [(c, cy)], gates, cfg, body_lr,
        create_graph=True)
    single_list_error = abs(float((single - single_list).detach()))
    unchanged = all(torch.equal(state[name], before[name]) for name in state)
    projected = project_gates(gates + 0.1 * direction, pad, 0.5)
    expected = sequence_budget(pad[0], 0.5, cfg.n_layers)
    budget_error = abs(float(projected.sum()) - expected)
    hard = harden_gates(projected, pad, 0.5)
    hard_count_error = abs(int(hard.sum()) - expected)
    batched_pad = torch.ones(2, cfg.seq_len, dtype=torch.bool)
    batched_score = torch.zeros(cfg.n_layers, 2, cfg.seq_len)
    random_a = scores_to_layer_mask(
        batched_score, batched_pad, 0.5, generator=generator, random_mask=True)
    random_b = scores_to_layer_mask(
        batched_score, batched_pad, 0.5, generator=generator, random_mask=True)
    batched_random_counts_correct = bool(torch.all(
        random_a.sum(dim=(0, 2)) == expected))
    batched_random_masks_differ = not torch.equal(random_a, random_b)
    optimized, history = optimize_gates(
        model, state, (x, y), reward_batches, pad, cfg, body_lr=body_lr,
        gate_lr=0.1, gate_steps=3, budget=0.5, initialization="uniform",
        generator=generator)
    result = {
        "dtype": str(next(model.parameters()).dtype),
        "device": "cpu",
        "reward_batches": len(reward_batches),
        "analytic_directional_hypergradient": analytic,
        "central_finite_difference": numeric,
        "relative_error": relative_error,
        "single_batch_list_equivalence_error": single_list_error,
        "original_state_unchanged": unchanged,
        "soft_budget_error": budget_error,
        "hard_budget_count_error": hard_count_error,
        "batched_random_counts_correct": batched_random_counts_correct,
        "batched_random_masks_differ": batched_random_masks_differ,
        "q_history": [r["q"] for r in history],
        "q_improvement": history[0]["q"] - history[-1]["q"],
        "optimized_gate_budget_error": abs(float(optimized.sum()) - expected),
    }
    result["passed"] = bool(
        relative_error < 2e-4 and single_list_error < 1e-12 and unchanged
        and budget_error < 1e-8 and hard_count_error == 0
        and batched_random_counts_correct and batched_random_masks_differ
        and result["q_improvement"] > 0
        and result["optimized_gate_budget_error"] < 1e-8)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("inverse-oracle self-test failed")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--task", default="tagged")
    p.add_argument("--size", default="small")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--c-opt-batches", type=int, default=1)
    p.add_argument("--c-hold-batches", type=int, default=1)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float)
    p.add_argument("--gate-lr", type=float, default=0.1)
    p.add_argument("--gate-steps", type=int, default=5)
    p.add_argument("--multistart", type=int, default=1)
    p.add_argument("--initialization", choices=["uniform", "random"],
                   default="uniform")
    p.add_argument("--split", choices=["smoke", "development", "confirmation"],
                   default="development")
    p.add_argument("--allow-legacy-extra-keys", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.ckpt or args.lr is None:
        p.error("--ckpt and --lr are required unless --self-test is used")
    if not 0 < args.budget <= 1:
        raise ValueError("--budget must be in (0, 1]")
    if args.candidates < 2 or args.replicates < 1:
        raise ValueError("need at least two random candidates and one replicate")
    if args.c_opt_batches < 1 or args.c_hold_batches < 1:
        raise ValueError("C-opt and C-hold batch counts must be positive")
    if args.gate_steps < 1 or args.multistart < 1 or args.gate_lr <= 0:
        raise ValueError("invalid gate optimization settings")
    if args.split == "confirmation" and not args.out:
        p.error("confirmation runs require --out")

    set_seed(args.seed)
    model, cfg = load_model(args)
    state = model_state(model)
    task = get_task(args.task)
    rng = random.Random(args.seed * 100003 + 177)
    generator = torch.Generator().manual_seed(args.seed * 100003 + 911)
    print(f"Inverse-oracle {args.split} | "
          f"{sum(p.numel() for p in model.parameters()):,} params | "
          f"{cfg.n_layers} layers | device={cfg.device} | "
          f"budget={args.budget} | body_lr={args.lr} | "
          f"gate_lr={args.gate_lr} | steps={args.gate_steps} | "
          f"C-opt={args.c_opt_batches} | C-hold={args.c_hold_batches}",
          flush=True)
    replicates = []
    start = time.time()
    for r in range(args.replicates):
        replicates.append(run_replicate(
            model, state, task, cfg, rng, generator, args, r))
    result = {
        "protocol": "one_step_inverse_oracle_v2",
        "split": args.split,
        "checkpoint": os.path.abspath(args.ckpt),
        "resolved_model": {
            "params": sum(p.numel() for p in model.parameters()),
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "size_name": cfg.size_name,
            "dropout": cfg.dropout,
        },
        "budget_scope": "per_sequence_valid_token_layer",
        "body_update": "unclipped_sgd_no_optimizer_state",
        "execution": "binary_output_masking_dense_ffn_not_gather_scatter",
        "confirmation": "dense_body_mean_of_disjoint_c_hold_batches",
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
