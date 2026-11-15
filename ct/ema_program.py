"""Final exponential or uniform-tail averaging for sparse-update training."""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .calibration_program import calibrate, training_forward
from .inverse_oracle import load_model
from .meta_router import batch_metrics, evaluation_batches
from .program_router import PROGRAM_BY_NAME, program_gates, random_gates
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


@torch.no_grad()
def update_ema(model, state, decay):
    for name, parameter in model.named_parameters():
        if name not in state:
            state[name] = parameter.detach().clone()
        else:
            state[name].mul_(decay).add_(parameter.detach(), alpha=1 - decay)


@torch.no_grad()
def install_ema(model, state):
    for name, parameter in model.named_parameters():
        parameter.copy_(state[name])


@torch.no_grad()
def install_tail(model, states, method="mean"):
    for name, parameter in model.named_parameters():
        values = torch.stack([state[name] for state in states])
        if method == "median":
            parameter.copy_(values.median(dim=0).values)
        else:
            parameter.copy_(values.mean(dim=0))


def run_arm(args, body_seed, budget, program):
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
    set_seed(online_seed * 100003 + 17)
    dense = args.arm == "dense"
    suffix = "_all" if args.all_backward and not dense else ""
    if args.backward_scale != 1 and not dense:
        suffix += "_scaled"
    if args.anchor_every > 0 and not dense:
        suffix += f"_anchor{args.anchor_every}"
    if args.dense_tail > 0 and not dense:
        suffix += f"_densetail{args.dense_tail}"
    if args.dense_warmup > 0 and not dense:
        suffix += f"_warmup{args.dense_warmup}"
    if args.batch_size != 32:
        suffix += f"_batch{args.batch_size}"
    if args.tail_average and args.tail_method == "median":
        name = f"{args.arm}{suffix}_median{args.tail_average}"
    else:
        name = (f"{args.arm}{suffix}_tail{args.tail_average}"
                if args.tail_average else f"{args.arm}{suffix}_ema")
    if args.calibration_batches:
        name += "_calibrated"
    ema_state = {}
    tail_states = []
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    cum_train = 0.0
    cum_calibrated = 0.0
    cum_aux = 0.0
    cum_ema_aux = 0.0
    cum_eval = 0.0
    history = []
    active_history = []
    ema_result = None
    start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{name}_seed{online_seed}.json")

    def gates_for(batch, active, operations):
        if dense:
            return None
        pad = batch[0] != TOKENIZER.pad_id
        if program is not None:
            return program_gates(
                program, cfg.n_layers, pad, budget, active, operations)
        return random_gates(
            cfg.n_layers, batch[0].shape, pad, budget,
            mask_generator, active, operations)

    def save(completed=False):
        result = {
            "format": "equal_opportunity_final_ema_v1",
            "split": args.split,
            "completed": completed,
            "arm": name,
            "base_arm": args.arm,
            "program": None if program is None else program.to_dict(),
            "body_seed": body_seed,
            "online_seed": online_seed,
            "checkpoint": os.path.abspath(local.ckpt),
            "execution": "ordinary dense training" if dense else (
                "dense forward with custom active-row-only attention-projection and FFN backward"
                if args.all_backward else
                "dense forward with custom active-row-only FFN backward"),
            "ema_decay": args.ema_decay,
            "tail_average": args.tail_average,
            "tail_method": args.tail_method,
            "anchor_every": 0 if dense else args.anchor_every,
            "dense_tail": 0 if dense else args.dense_tail,
            "dense_warmup": 0 if dense else args.dense_warmup,
            "ema_result": ema_result,
            "config": {k: v for k, v in vars(args).items() if k != "handler"},
            "budget": budget,
            "history": history,
            "active_history": active_history,
            "final": history[-1] if history else None,
        }
        temporary = out + ".tmp"
        with open(temporary, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(temporary, out)

    def evaluate(step):
        nonlocal cum_eval
        model.eval()
        eval_batches = evaluation_batches(task, cfg, held_out, args.eval_batches)
        with FlopCounterMode(display=False) as counter:
            metrics = batch_metrics(model, eval_batches, cfg)
        cum_eval += float(counter.get_total_flops())
        row = {
            "step": step,
            "heldout_loss": metrics["loss"],
            "heldout_acc": metrics["acc"],
            "cum_train_flops": cum_train,
            "cum_full_cpu_calibrated_flops": cum_calibrated,
            "cum_aux_ops": cum_aux,
            "cum_ema_aux_ops": cum_ema_aux,
            "cum_total_flops_with_aux_estimate": cum_calibrated + cum_aux + cum_ema_aux,
            "cum_eval_flops": cum_eval,
            "wallclock_s": time.time() - start,
            "mean_active_fraction": (float(np.mean(active_history))
                                     if active_history else None),
        }
        history.append(row)
        print(f"[{name}/b{body_seed}] step {step} | raw "
              f"{metrics['loss']:.6f} | acc {metrics['acc']:.6f} | total+aux "
              f"{row['cum_total_flops_with_aux_estimate'] / 1e12:.3f}T | "
              f"{row['wallclock_s']:.0f}s", flush=True)
        save(False)

    evaluate(0)
    for index, batch in enumerate(batches):
        step = index + 1
        model.train()
        anchor = args.anchor_every > 0 and step % args.anchor_every == 0
        terminal_dense = args.dense_tail > 0 and step > args.steps - args.dense_tail
        warmup = args.dense_warmup > 0 and step <= args.dense_warmup
        dense_update = dense or anchor or terminal_dense or warmup
        active = []
        operations = []
        gates = None if dense_update else gates_for(batch, active, operations)
        with FlopCounterMode(display=False) as train_counter:
            optimizer.zero_grad(set_to_none=True)
            loss, info = training_forward(
                model, batch, cfg, gates, dense_update,
                all_backward=args.all_backward,
                backward_scale=args.backward_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if args.tail_average:
            tail_states.append({
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()})
            if len(tail_states) > args.tail_average:
                tail_states.pop(0)
            cum_ema_aux += parameter_count
        else:
            update_ema(model, ema_state, args.ema_decay)
            cum_ema_aux += 2 * parameter_count
        cum_train += float(train_counter.get_total_flops())
        tokens_per_layer = batch[0].numel()
        active_count = (tokens_per_layer * cfg.n_layers if dense_update
                        else int(sum(active)))
        inactive_count = tokens_per_layer * cfg.n_layers - active_count
        dense_step_flops = 580235821056 * args.batch_size // 32
        saved_per_row = 8 * cfg.d_model * cfg.d_ff
        if args.all_backward and not dense_update:
            saved_per_row += 16 * cfg.d_model * cfg.d_model
        step_calibrated = dense_step_flops - saved_per_row * inactive_count
        cum_calibrated += step_calibrated
        gather_scatter_ops = 4 * active_count * (cfg.d_model + cfg.d_ff)
        if args.all_backward and not dense_update:
            gather_scatter_ops += 16 * active_count * cfg.d_model
        scale_ops = 0
        if args.backward_scale != 1 and not dense_update:
            scale_ops = cfg.n_layers * (
                2 * cfg.d_model * cfg.d_ff + tokens_per_layer * cfg.d_model
                + cfg.d_ff + cfg.d_model)
        cum_aux += (sum(operations)
                    + (0 if dense_update else gather_scatter_ops) + scale_ops)
        active_history.append(
            1.0 if dense_update else
            active_count / (cfg.n_layers * int((batch[0] != TOKENIZER.pad_id).sum())))
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)

    averaging_method = "ema"
    if args.tail_average:
        reduction_multiplier = 5 if args.tail_method == "median" else 1
        cum_ema_aux += reduction_multiplier * len(tail_states) * parameter_count
        history[-1]["cum_ema_aux_ops"] = cum_ema_aux
        history[-1]["cum_total_flops_with_aux_estimate"] = (
            cum_calibrated + cum_aux + cum_ema_aux)
        install_tail(model, tail_states, args.tail_method)
        averaging_method = f"{args.tail_method}_last_{len(tail_states)}"
    else:
        install_ema(model, ema_state)
    model.eval()
    final_batches = evaluation_batches(task, cfg, held_out, args.eval_batches)
    with FlopCounterMode(display=False) as ema_counter:
        metrics = batch_metrics(model, final_batches, cfg)
    ema_result = {
        "averaging_method": averaging_method,
        "heldout_loss": metrics["loss"],
        "heldout_acc": metrics["acc"],
        "raw_heldout_loss": history[-1]["heldout_loss"],
        "raw_heldout_acc": history[-1]["heldout_acc"],
        "training_total_flops": history[-1]["cum_total_flops_with_aux_estimate"],
        "ema_evaluation_flops": float(ema_counter.get_total_flops()),
        "wallclock_s": time.time() - start,
    }
    ema_result["total_with_calibration_flops"] = ema_result["training_total_flops"]
    if args.calibration_batches:
        calibration_cfg = copy.deepcopy(cfg)
        calibration_cfg.held_out_seed = args.calibration_seed
        calibration_held_out = task.build_held_out(calibration_cfg)
        calibration_batches = evaluation_batches(
            task, calibration_cfg, calibration_held_out, args.calibration_batches)
        calibration_result = calibrate(
            model, calibration_batches, final_batches, calibration_cfg)
        ema_result["calibration"] = calibration_result
        ema_result["total_with_calibration_flops"] = (
            ema_result["training_total_flops"]
            + calibration_result["calibration_total_flops"])
        ema_result["wallclock_s"] = time.time() - start
    save(True)
    label = "TAIL" if args.tail_average else "EMA"
    print(f"[{name}/b{body_seed}] {label} loss {metrics['loss']:.6f} | raw "
          f"{history[-1]['heldout_loss']:.6f} | {label} acc {metrics['acc']:.6f} | "
          f"total {ema_result['training_total_flops'] / 1e12:.3f}T", flush=True)
    if args.calibration_batches:
        print(f"[{name}/b{body_seed}] calibrated "
              f"{ema_result['calibration']['calibrated']['loss']:.6f} | temperature "
              f"{ema_result['calibration']['temperature']:.6f} | total+cal "
              f"{ema_result['total_with_calibration_flops'] / 1e12:.3f}T", flush=True)
    print(f"saved {out}", flush=True)
    return {"arm": name, "path": out, "ema_result": ema_result}


def main():
    parser = argparse.ArgumentParser()
    run = parser.add_subparsers(dest="mode", required=True).add_parser("run")
    run.add_argument("--ckpt-template", default=(
        "results-inverse-ckpt/tagged/predictor_stage1_seed{seed}.pt"))
    run.add_argument("--outdir", required=True)
    run.add_argument("--device", default=None)
    run.add_argument("--size", default="small")
    run.add_argument("--seq-len", type=int, default=256)
    run.add_argument("--task", default="tagged")
    run.add_argument("--body-seed", type=int, required=True)
    run.add_argument("--budget", type=float, default=0.875)
    run.add_argument("--arm", choices=["low_max_abs", "random", "dense"], required=True)
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--body-lr", type=float, default=0.0003)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--all-backward", action="store_true")
    run.add_argument("--backward-scale", type=float, default=1.0)
    run.add_argument("--anchor-every", type=int, default=0)
    run.add_argument("--dense-tail", type=int, default=0)
    run.add_argument("--dense-warmup", type=int, default=0)
    run.add_argument("--tail-average", type=int, default=0)
    run.add_argument("--tail-method", choices=["mean", "median"], default="mean")
    run.add_argument("--ema-decay", type=float, default=0.95)
    run.add_argument("--calibration-batches", type=int, default=0)
    run.add_argument("--calibration-seed", type=int, default=24680)
    run.add_argument("--eval-every", type=int, default=20)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--seed", type=int, default=9400)
    run.add_argument("--held-out-seed", type=int, default=54321)
    run.add_argument("--split", choices=["development", "confirmation"],
                     default="development")
    args = parser.parse_args()
    if args.all_backward and args.backward_scale != 1:
        raise ValueError("backward scaling is implemented for FFN-only sparsity")
    if args.anchor_every < 0 or args.anchor_every == 1:
        raise ValueError("anchor_every must be zero or at least two")
    if args.tail_average < 0 or args.tail_average == 1:
        raise ValueError("tail_average must be zero or at least two")
    if args.dense_tail < 0 or args.dense_tail > args.steps:
        raise ValueError("dense_tail must be between zero and steps")
    if args.dense_warmup < 0 or args.dense_warmup > args.steps:
        raise ValueError("dense_warmup must be between zero and steps")
    program = PROGRAM_BY_NAME.get(args.arm)
    run_arm(args, args.body_seed, args.budget, program)


if __name__ == "__main__":
    main()
