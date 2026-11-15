"""Lagged Adam-preconditioned directional utility from sampled FFN gradients."""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .gradient_program import gradient_gates, new_gradient_state
from .inverse_oracle import load_model
from .meta_router import batch_metrics, evaluation_batches, forward_train
from .program_router import random_gates
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def make_parameter_samples(model, seed, sample_size):
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for layer in model.layers:
        layer_samples = {}
        for name, parameter in (("w1", layer.ffn[0].weight),
                                ("w2", layer.ffn[3].weight)):
            if sample_size is None:
                indices = torch.arange(parameter.numel())
            else:
                indices = torch.randperm(
                    parameter.numel(), generator=generator)[:sample_size]
            rows = (indices // parameter.shape[1]).to(parameter.device)
            columns = (indices % parameter.shape[1]).to(parameter.device)
            layer_samples[name] = (rows, columns)
        samples.append(layer_samples)
    return samples


def attach_ffn_hooks(model, gates, tokens):
    records = [{"tokens": tokens.reshape(-1)[
        gates[layer].reshape(-1).nonzero(as_tuple=False).squeeze(-1)].detach()}
        for layer in range(len(model.layers))]
    handles = []
    for index, layer in enumerate(model.layers):
        record = records[index]

        def w1_hook(module, inputs, output, record=record):
            record["w1_input"] = inputs[0].detach()
            record["w1_output"] = output
            if output.requires_grad:
                output.retain_grad()

        def w2_hook(module, inputs, output, record=record):
            record["w2_input"] = inputs[0].detach()
            record["w2_output"] = output
            if output.requires_grad:
                output.retain_grad()

        handles.append(layer.ffn[0].register_forward_hook(w1_hook))
        handles.append(layer.ffn[3].register_forward_hook(w2_hook))
    return records, handles


def adam_direction(parameter, optimizer):
    state = optimizer.state[parameter]
    step = int(state["step"])
    beta1, beta2 = optimizer.defaults["betas"]
    first = state["exp_avg"] / (1 - beta1 ** step)
    second = state["exp_avg_sq"] / (1 - beta2 ** step)
    return first / (second.sqrt() + optimizer.defaults["eps"])


@torch.no_grad()
def layer_alignment(record, first_direction, second_direction, samples):
    first_rows, first_columns = samples["w1"]
    second_rows, second_columns = samples["w2"]
    first_gradient = record["w1_output"].grad
    second_gradient = record["w2_output"].grad
    if first_gradient is None or second_gradient is None:
        raise RuntimeError("sampled FFN gradients were not retained")
    first_samples = (first_gradient[:, first_rows]
                     * record["w1_input"][:, first_columns])
    second_samples = (second_gradient[:, second_rows]
                      * record["w2_input"][:, second_columns])
    first_update = first_direction[first_rows, first_columns]
    second_update = second_direction[second_rows, second_columns]
    numerator = ((first_samples * first_update).sum(dim=-1)
                 + (second_samples * second_update).sum(dim=-1))
    gradient_square = (first_samples.square().sum(dim=-1)
                       + second_samples.square().sum(dim=-1))
    update_square = first_update.square().sum() + second_update.square().sum()
    denominator = gradient_square.sqrt() * update_square.sqrt() + 1e-12
    alignment = numerator / denominator
    alignment = torch.where(denominator > 1e-12, alignment, torch.zeros_like(alignment))
    return alignment


@torch.no_grad()
def update_directional_state(state, records, model, optimizer, samples):
    mean_alignments = []
    newly_seen = 0
    auxiliary_ops = 0
    sample_size = samples[0]["w1"][0].numel() + samples[0]["w2"][0].numel()
    for index, layer in enumerate(model.layers):
        record = records[index]
        first_direction = adam_direction(layer.ffn[0].weight, optimizer)
        second_direction = adam_direction(layer.ffn[3].weight, optimizer)
        alignment = layer_alignment(
            record, first_direction, second_direction, samples[index])
        ids = record["tokens"]
        sums = torch.zeros(
            state["ema"].shape[1], dtype=alignment.dtype, device=alignment.device)
        counts = torch.zeros_like(sums)
        sums.index_add_(0, ids, alignment)
        counts.index_add_(0, ids, torch.ones_like(alignment))
        observed = counts > 0
        means = sums / counts.clamp_min(1)
        first = observed & ~state["seen"][index]
        state["ema"][index, first] = means[first]
        continuing = observed & state["seen"][index]
        beta = state["beta"]
        state["ema"][index, continuing] = (
            (1 - beta) * state["ema"][index, continuing]
            + beta * means[continuing])
        state["seen"][index] |= observed
        state["observations"][index] += counts
        mean_alignments.append(float(alignment.mean()))
        newly_seen += int(first.sum())
        auxiliary_ops += 6 * sample_size * len(ids) + 4 * int(observed.sum())
    state["updates"] += 1
    return {
        "step_update": state["updates"],
        "mean_sampled_adam_alignment": float(np.mean(mean_alignments)),
        "newly_seen_token_types": newly_seen,
        "seen_token_types": int(state["seen"].sum()),
        "aux_ops": auxiliary_ops,
    }


def run_arm(args, body_seed, budget):
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
    state = new_gradient_state(
        cfg.n_layers, cfg.vocab_size, cfg.device, args.gradient_beta)
    samples = make_parameter_samples(
        model, online_seed * 100003 + 817, args.parameter_samples)
    arm = "adam_alignment"
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
        return gradient_gates(
            state, batch[0], batch[0] != TOKENIZER.pad_id, budget, 1,
            eval_generator if evaluation else mask_generator,
            active_sink, op_sink)

    def save(completed=False):
        result = {
            "format": "sampled_adam_directional_program_v1",
            "split": args.split,
            "completed": completed,
            "arm": arm,
            "body_seed": body_seed,
            "online_seed": online_seed,
            "checkpoint": os.path.abspath(local.ckpt),
            "execution": "strictly lagged token-conditional EMA of sampled FFN-gradient/Adam-update cosine; exact per-layer Top-B; gather/scatter",
            "causality": "current-step routing uses only completed previous backward passes; first step is exact-budget random; unseen types use the observed layer mean",
            "approximation": "FFN weight gradients only; fixed uniform parameter subsampling; no biases, attention, weight decay, Hessian, or full per-token parameter gradients",
            "accounting": "ordinary backward only; executed model FLOPs plus separately estimated sampling, alignment, lookup, EMA, and selection operations; zero extra forward passes",
            "config": {k: v for k, v in vars(args).items() if k != "handler"},
            "budget": budget,
            "history": history,
            "active_history": active_history,
            "gradient_history": gradient_history,
            "gradient_state": {
                "updates": state["updates"],
                "seen_token_types": int(state["seen"].sum()),
                "ema": state["ema"].detach().cpu().tolist(),
                "observations": state["observations"].detach().cpu().tolist(),
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
        updates_before = state["updates"]
        with FlopCounterMode(display=False) as counter:
            dense = batch_metrics(model, eval_batches, cfg)
            sparse = batch_metrics(
                model, eval_batches, cfg,
                lambda batch: gates_for(batch, [], eval_ops, evaluation=True))
        if state["updates"] != updates_before:
            raise RuntimeError("evaluation changed directional state")
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
        if state["updates"] != step - 1:
            raise RuntimeError("directional statistic is not exactly one-step lagged")
        records, handles = attach_ffn_hooks(model, gates, batch[0])
        try:
            with FlopCounterMode(display=False) as train_counter:
                optimizer.zero_grad(set_to_none=True)
                loss, info = forward_train(model, batch, cfg, gates, True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                update = update_directional_state(
                    state, records, model, optimizer, samples)
                gradient_history.append(update)
                step_ops.append(update["aux_ops"])
        finally:
            for handle in handles:
                handle.remove()
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
            run_arm(regime_args, body_seed, budget)
            print(f"completed directional regime body={body_seed} budget={budget}", flush=True)


def self_test():
    from .config import Config
    from .models import build_model
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(8128)
        cfg = Config.for_size(
            "small", device="cpu", d_model=16, n_layers=2, n_heads=4,
            d_ff=32, max_len=16, seq_len=6, dropout=0.0)
        model = build_model("predictor-supervised", cfg).double().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 19, (2, 6))
        x[0, -1] = TOKENIZER.pad_id
        y = torch.roll(x, -1, dims=1)
        y[:, -1] = TOKENIZER.pad_id
        pad = x != TOKENIZER.pad_id
        gates = random_gates(
            2, x.shape, pad, 0.5, torch.Generator().manual_seed(5))
        records, handles = attach_ffn_hooks(model, gates, x)
        try:
            loss, info = forward_train(model, (x, y), cfg, gates, True)
            loss.backward()
            optimizer.step()
            samples = make_parameter_samples(model, 7, None)
            errors = []
            for index, layer in enumerate(model.layers):
                first_direction = adam_direction(layer.ffn[0].weight, optimizer)
                second_direction = adam_direction(layer.ffn[3].weight, optimizer)
                estimated = layer_alignment(
                    records[index], first_direction, second_direction, samples[index])
                first_gradient = records[index]["w1_output"].grad
                second_gradient = records[index]["w2_output"].grad
                exact_first = (first_gradient.unsqueeze(-1)
                               * records[index]["w1_input"].unsqueeze(1))
                exact_second = (second_gradient.unsqueeze(-1)
                                * records[index]["w2_input"].unsqueeze(1))
                numerator = ((exact_first * first_direction).sum(dim=(1, 2))
                             + (exact_second * second_direction).sum(dim=(1, 2)))
                gradient_square = (exact_first.square().sum(dim=(1, 2))
                                   + exact_second.square().sum(dim=(1, 2)))
                update_square = (first_direction.square().sum()
                                 + second_direction.square().sum())
                exact = numerator / (gradient_square.sqrt() * update_square.sqrt() + 1e-12)
                errors.append(float((estimated - exact).abs().max()))
            state = new_gradient_state(2, cfg.vocab_size, "cpu", 0.2)
            update = update_directional_state(
                state, records, model, optimizer, samples)
            next_gates = gradient_gates(
                state, x, pad, 0.5, 1, torch.Generator().manual_seed(8))
            for mask in next_gates:
                for b in range(2):
                    assert int(mask[b].sum()) == max(1, int(round(0.5 * int(pad[b].sum()))))
                assert float(mask[~pad].sum()) == 0
            assert state["updates"] == 1 and update["aux_ops"] > 0
            passed = max(errors) < 1e-10
            result = {
                "full_parameter_estimator_max_error": max(errors),
                "adam_direction_reconstruction": "passed" if passed else "failed",
                "exact_next_step_budget": "passed",
                "one_step_lagged_state": "passed",
                "padding_error": 0.0,
                "passed": passed,
            }
            print(json.dumps(result, indent=2), flush=True)
            if not passed:
                raise RuntimeError("directional estimator self-test failed")
            return result
        finally:
            for handle in handles:
                handle.remove()
    finally:
        torch.set_default_dtype(previous)


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
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--body-lr", type=float, default=0.0003)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--gradient-beta", type=float, default=0.2)
    run.add_argument("--parameter-samples", type=int, default=128)
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
