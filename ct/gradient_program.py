"""Executable lagged-gradient allocation using only ordinary backpropagation."""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from .inverse_oracle import load_model, scores_to_layer_mask
from .meta_router import (
    batch_metrics, evaluation_batches, forward_train, permute_valid_scores)
from .program_router import (
    PROGRAM_BY_NAME, base_values, feature_ops_per_token, random_gates)
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def new_gradient_state(n_layers, vocab_size, device, beta):
    return {
        "ema": torch.zeros(n_layers, vocab_size, device=device),
        "seen": torch.zeros(n_layers, vocab_size, dtype=torch.bool, device=device),
        "observations": torch.zeros(n_layers, vocab_size, device=device),
        "beta": beta,
        "updates": 0,
    }


def forward_with_sink(model, batch, cfg, gates, sink):
    x, y = batch
    logits, info = model(
        x, pad_mask=x != TOKENIZER.pad_id,
        meta={"diffs": None, "gather_ffn": gates is not None,
              "pre_ffn_sink": sink},
        oracle_gates=gates)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), y.view(-1),
        ignore_index=TOKENIZER.pad_id)
    return loss, info


@torch.no_grad()
def update_gradient_state(state, features, tokens, pad):
    if len(features) != state["ema"].shape[0]:
        raise ValueError("gradient feature count does not match model layers")
    mean_norms = []
    newly_seen = 0
    auxiliary_ops = 0
    for layer, feature in enumerate(features):
        if feature.grad is None:
            raise RuntimeError("pre-FFN gradient was not retained")
        norm = feature.grad.detach().square().sum(dim=-1).sqrt()
        values = norm[pad]
        ids = tokens[pad]
        sums = torch.zeros(
            state["ema"].shape[1], dtype=norm.dtype, device=norm.device)
        counts = torch.zeros_like(sums)
        sums.index_add_(0, ids, values)
        counts.index_add_(0, ids, torch.ones_like(values))
        observed = counts > 0
        means = sums / counts.clamp_min(1)
        first = observed & ~state["seen"][layer]
        state["ema"][layer, first] = means[first]
        continuing = observed & state["seen"][layer]
        beta = state["beta"]
        state["ema"][layer, continuing] = (
            (1 - beta) * state["ema"][layer, continuing]
            + beta * means[continuing])
        state["seen"][layer] |= observed
        state["observations"][layer] += counts
        mean_norms.append(float(values.mean()))
        newly_seen += int(first.sum())
        auxiliary_ops += int(pad.sum()) * (feature.shape[-1] + 4) + 4 * int(observed.sum())
    state["updates"] += 1
    return {
        "step_update": state["updates"],
        "mean_gradient_norm": float(np.mean(mean_norms)),
        "newly_seen_token_types": newly_seen,
        "seen_token_types": int(state["seen"].sum()),
        "aux_ops": auxiliary_ops,
    }


def gradient_gates(state, tokens, pad, budget, sign, generator,
                   active_sink=None, op_sink=None):
    n_layers = state["ema"].shape[0]
    if state["updates"] == 0:
        return random_gates(
            n_layers, tokens.shape, pad, budget, generator, active_sink, op_sink)
    gates = []
    for layer in range(n_layers):
        seen = state["seen"][layer]
        table = state["ema"][layer]
        if not bool(seen.all()):
            fallback = table[seen].mean()
            table = torch.where(seen, table, torch.full_like(table, fallback))
        scores = sign * table[tokens]
        hard = scores_to_layer_mask(scores.detach().unsqueeze(0), pad, budget)[0]
        gates.append(hard)
        if active_sink is not None:
            active_sink.append(float(hard.sum()))
        if op_sink is not None:
            op_sink.append(2 * int(pad.sum()) + tokens.shape[0] * tokens.shape[1])
    return gates


def shuffled_low_max_gates(n_layers, pad, budget, generator,
                           active_sink=None, op_sink=None):
    program = PROGRAM_BY_NAME["low_max_abs"]

    def make_gate():
        def gate(x):
            scores = -base_values("max_abs", x, pad)
            scores = permute_valid_scores(scores, pad, generator)
            hard = scores_to_layer_mask(scores.unsqueeze(0), pad, budget)[0]
            if active_sink is not None:
                active_sink.append(float(hard.sum()))
            if op_sink is not None:
                op_sink.append(
                    int(pad.sum()) * (feature_ops_per_token(program, x.shape[-1]) + 1)
                    + x.shape[0] * x.shape[1])
            return hard
        return gate
    return [make_gate() for _ in range(n_layers)]


def run_arm(args, body_seed, budget, arm):
    online_seed = args.seed + body_seed
    set_seed(online_seed)
    local = argparse.Namespace(**vars(args))
    local.ckpt = args.ckpt_template.format(seed=body_seed)
    local.allow_legacy_extra_keys = True
    model, cfg = load_model(local)
    cfg.batch_size = args.batch_size
    cfg.dropout = args.dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.dropout
    task = get_task(args.task)
    held_out = task.build_held_out(cfg)
    rng = random.Random(online_seed * 100003 + 7)
    batches = [get_batch(task, cfg, rng, args.batch_size)[:2]
               for _ in range(args.steps)]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.body_lr, weight_decay=args.weight_decay)
    mask_generator = torch.Generator().manual_seed(online_seed * 100003 + 811)
    eval_generator = torch.Generator().manual_seed(online_seed * 100003 + 813)
    set_seed(online_seed * 100003 + 17)
    gradient_state = None
    if arm in ("gradient_high", "gradient_low"):
        gradient_state = new_gradient_state(
            cfg.n_layers, cfg.vocab_size, cfg.device, args.gradient_beta)
    cum_train = 0.0
    cum_aux = 0.0
    cum_eval = 0.0
    cum_eval_aux = 0.0
    history = []
    active_history = []
    gradient_history = []
    start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{arm}_seed{online_seed}.json")

    def gates_for(batch, active_sink, op_sink, evaluation=False):
        pad = batch[0] != TOKENIZER.pad_id
        generator = eval_generator if evaluation else mask_generator
        if gradient_state is not None:
            return gradient_gates(
                gradient_state, batch[0], pad, budget,
                1 if arm == "gradient_high" else -1, generator,
                active_sink, op_sink)
        if arm == "shuffled_low_max_abs":
            return shuffled_low_max_gates(
                cfg.n_layers, pad, budget, generator, active_sink, op_sink)
        raise ValueError(f"unknown arm: {arm}")

    def save(completed=False):
        result = {
            "format": "lagged_gradient_program_v1",
            "split": args.split,
            "completed": completed,
            "arm": arm,
            "body_seed": body_seed,
            "online_seed": online_seed,
            "checkpoint": os.path.abspath(local.ckpt),
            "execution": "previous ordinary-backprop token-conditional gradient EMA or shuffled low-max-abs; exact per-layer Top-B; gather/scatter",
            "causality": "current-step routing never consumes current-step gradients; first gradient step is exact-budget random; unseen types use the observed layer mean",
            "accounting": "executed matmul/attention FLOPs plus separately estimated lookup, gradient-statistic, permutation, and selection operations; zero extra gradient forward passes",
            "config": {k: v for k, v in vars(args).items() if k != "handler"},
            "budget": budget,
            "history": history,
            "active_history": active_history,
            "gradient_history": gradient_history,
            "gradient_state": None if gradient_state is None else {
                "updates": gradient_state["updates"],
                "seen_token_types": int(gradient_state["seen"].sum()),
                "ema": gradient_state["ema"].detach().cpu().tolist(),
                "observations": gradient_state["observations"].detach().cpu().tolist(),
            },
            "final": history[-1] if history else None,
        }
        temporary = out + ".tmp"
        with open(temporary, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(temporary, out)

    def evaluate(step):
        nonlocal cum_eval, cum_eval_aux
        model.eval()
        eval_batches = evaluation_batches(task, cfg, held_out, args.eval_batches)
        eval_ops = []
        updates_before = None if gradient_state is None else gradient_state["updates"]
        with FlopCounterMode(display=False) as counter:
            dense = batch_metrics(model, eval_batches, cfg)
            sparse = batch_metrics(
                model, eval_batches, cfg,
                lambda batch: gates_for(batch, [], eval_ops, evaluation=True))
        if gradient_state is not None and gradient_state["updates"] != updates_before:
            raise RuntimeError("evaluation changed gradient state")
        cum_eval += float(counter.get_total_flops())
        cum_eval_aux += sum(eval_ops)
        row = {
            "step": step,
            "heldout_loss": dense["loss"],
            "heldout_acc": dense["acc"],
            "sparse_heldout_loss": sparse["loss"],
            "sparse_heldout_acc": sparse["acc"],
            "cum_train_flops": cum_train,
            "cum_aux_ops": cum_aux,
            "cum_total_flops_with_aux_estimate": cum_train + cum_aux,
            "cum_eval_flops": cum_eval,
            "cum_eval_aux_ops": cum_eval_aux,
            "wallclock_s": time.time() - start,
            "mean_active_fraction": (float(np.mean(active_history))
                                     if active_history else None),
        }
        history.append(row)
        print(f"[{arm}/b{body_seed}/B{budget:g}] step {step} | dense "
              f"{dense['loss']:.6f} | sparse {sparse['loss']:.6f} | total+aux "
              f"{row['cum_total_flops_with_aux_estimate'] / 1e12:.3f}T | active "
              f"{row['mean_active_fraction']} | {row['wallclock_s']:.0f}s", flush=True)
        save(False)

    evaluate(0)
    for index, batch in enumerate(batches):
        step = index + 1
        model.train()
        active = []
        step_ops = []
        gates = gates_for(batch, active, step_ops)
        updates_before = None if gradient_state is None else gradient_state["updates"]
        if updates_before is not None and updates_before != step - 1:
            raise RuntimeError("gradient statistic is not exactly one-step lagged")
        with FlopCounterMode(display=False) as train_counter:
            optimizer.zero_grad(set_to_none=True)
            if gradient_state is None:
                loss, info = forward_train(model, batch, cfg, gates, True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                features = []
                loss, info = forward_with_sink(model, batch, cfg, gates, features)
                for feature in features:
                    feature.retain_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                update = update_gradient_state(
                    gradient_state, features, batch[0], batch[0] != TOKENIZER.pad_id)
                gradient_history.append(update)
                step_ops.append(update["aux_ops"])
        cum_train += float(train_counter.get_total_flops())
        cum_aux += sum(step_ops)
        active_history.append(
            sum(active) / (cfg.n_layers * int((batch[0] != TOKENIZER.pad_id).sum())))
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
    save(True)
    print(f"saved {out}", flush=True)
    return {"arm": arm, "path": out, "history": history, "final": history[-1]}


def run_comparison(args):
    for body_seed in [int(value) for value in args.body_seeds.split(",")]:
        for budget in [float(value) for value in args.budgets.split(",")]:
            regime_args = argparse.Namespace(**vars(args))
            regime_args.outdir = os.path.join(
                args.outdir, f"body{body_seed}_budget{budget:g}")
            for arm in args.arms.split(","):
                run_arm(regime_args, body_seed, budget, arm)
            print(f"completed gradient regime body={body_seed} budget={budget}", flush=True)


def self_test():
    from .config import Config
    from .models import build_model
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(8128)
    cfg = Config.for_size(
        "small", device="cpu", d_model=16, n_layers=2, n_heads=4,
        d_ff=32, max_len=16, seq_len=6, dropout=0.0)
    model = build_model("predictor-supervised", cfg).double().train()
    x = torch.randint(0, 19, (2, 6))
    x[0, -1] = TOKENIZER.pad_id
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = TOKENIZER.pad_id
    pad = x != TOKENIZER.pad_id
    state = new_gradient_state(2, cfg.vocab_size, "cpu", 0.2)
    features = []
    loss, info = forward_with_sink(model, (x, y), cfg, None, features)
    for feature in features:
        feature.retain_grad()
    loss.backward()
    update = update_gradient_state(state, features, x, pad)
    assert state["updates"] == 1 and update["seen_token_types"] > 0
    active = []
    operations = []
    gates = gradient_gates(
        state, x, pad, 0.5, 1, torch.Generator().manual_seed(7), active, operations)
    for mask in gates:
        for b in range(2):
            assert int(mask[b].sum()) == max(1, int(round(0.5 * int(pad[b].sum()))))
        assert float(mask[~pad].sum()) == 0
    assert state["updates"] == 1 and all(value > 0 for value in operations)
    shuffled_active = []
    shuffled_ops = []
    shuffled = shuffled_low_max_gates(
        2, pad, 0.5, torch.Generator().manual_seed(8),
        shuffled_active, shuffled_ops)
    for gate in shuffled:
        mask = gate(features[0].detach())
        for b in range(2):
            assert int(mask[b].sum()) == max(1, int(round(0.5 * int(pad[b].sum()))))
        assert float(mask[~pad].sum()) == 0
    result = {
        "ordinary_backward_gradient_update": "passed",
        "one_step_lagged_state": "passed",
        "evaluation_state_unchanged": True,
        "exact_gradient_budget": "passed",
        "exact_shuffled_budget": "passed",
        "padding_error": 0.0,
        "auxiliary_operations_positive": True,
        "passed": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(handler=lambda args: self_test())
    run = sub.add_parser("run")
    run.add_argument("--ckpt-template", default=(
        "results-inverse-ckpt/tagged/predictor_stage1_seed{seed}.pt"))
    run.add_argument("--outdir", required=True)
    run.add_argument("--device", default=None)
    run.add_argument("--size", default="small")
    run.add_argument("--seq-len", type=int, default=256)
    run.add_argument("--task", default="tagged")
    run.add_argument("--body-seeds", default="4,8")
    run.add_argument("--budgets", default="0.375,0.5")
    run.add_argument("--arms", default="gradient_high,gradient_low,shuffled_low_max_abs")
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--body-lr", type=float, default=0.0003)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--gradient-beta", type=float, default=0.2)
    run.add_argument("--eval-every", type=int, default=50)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--seed", type=int, default=9400)
    run.add_argument("--split", choices=["development", "confirmation"],
                     default="development")
    run.set_defaults(handler=run_comparison)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
