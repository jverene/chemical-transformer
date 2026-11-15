"""Equal-opportunity temperature calibration for dense and sparse-update training."""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from .inverse_oracle import load_model
from .meta_router import batch_metrics, evaluation_batches
from .program_router import PROGRAM_BY_NAME, program_gates, random_gates
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def training_forward(model, batch, cfg, gates, dense, all_backward=False,
                     backward_scale=1.0):
    x, y = batch
    meta = {"diffs": None}
    if not dense:
        if all_backward:
            meta["dense_forward_sparse_all_backward"] = True
        else:
            meta["dense_forward_sparse_backward"] = True
        meta["sparse_backward_scale"] = backward_scale
    logits, info = model(
        x, pad_mask=x != TOKENIZER.pad_id, meta=meta, oracle_gates=gates)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), y.view(-1),
        ignore_index=TOKENIZER.pad_id)
    return loss, info


@torch.no_grad()
def collect_logits(model, batches, cfg):
    logits_values = []
    labels = []
    answer_masks = []
    for batch in batches:
        x, y, _, answer_mask = batch
        logits, _ = model(
            x, pad_mask=x != TOKENIZER.pad_id, meta={"diffs": None})
        logits_values.append(logits.detach())
        labels.append(y.detach())
        answer_masks.append(answer_mask.detach())
    return (torch.cat(logits_values), torch.cat(labels), torch.cat(answer_masks))


def metrics_from_logits(logits, labels, answer_masks, temperature=1.0):
    with torch.no_grad():
        loss = F.cross_entropy(
            (logits / temperature).view(-1, logits.shape[-1]), labels.view(-1),
            ignore_index=TOKENIZER.pad_id)
        pred = logits.argmax(dim=-1)
        valid = labels != TOKENIZER.pad_id
        shifted_answers = torch.roll(answer_masks, -1, dims=1)
        shifted_answers[:, -1] = 0
        selected = valid & (shifted_answers == 1)
        correct = int(((pred == labels) & selected).sum())
        answers = int(selected.sum())
    return {"loss": float(loss.detach()), "acc": correct / max(answers, 1)}


def calibrate(model, calibration_batches, final_batches, cfg):
    model.eval()
    with FlopCounterMode(display=False) as calibration_counter:
        calibration_logits, calibration_labels, _ = collect_logits(
            model, calibration_batches, cfg)
    log_temperature = torch.zeros(
        (), device=calibration_logits.device, dtype=calibration_logits.dtype,
        requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.5, max_iter=20,
        tolerance_grad=1e-10, tolerance_change=1e-12)
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            (calibration_logits * log_temperature.exp()).view(-1, cfg.vocab_size),
            calibration_labels.view(-1), ignore_index=TOKENIZER.pad_id)
        loss.backward()
        evaluations += 1
        return loss

    initial_calibration_loss = float(metrics_from_logits(
        calibration_logits, calibration_labels,
        torch.zeros_like(calibration_labels))["loss"])
    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach())
    if not np.isfinite(temperature) or not 0.05 <= temperature <= 20:
        raise RuntimeError(f"invalid calibrated temperature: {temperature}")
    calibration_aux_ops = 10 * evaluations * calibration_logits.numel()
    with FlopCounterMode(display=False) as evaluation_counter:
        final_logits, final_labels, final_answers = collect_logits(
            model, final_batches, cfg)
    uncalibrated = metrics_from_logits(final_logits, final_labels, final_answers)
    calibrated = metrics_from_logits(
        final_logits, final_labels, final_answers, temperature)
    calibration_loss = float(F.cross_entropy(
        (calibration_logits * log_temperature.exp().detach()).view(-1, cfg.vocab_size),
        calibration_labels.view(-1), ignore_index=TOKENIZER.pad_id))
    return {
        "calibration_seed": cfg.held_out_seed,
        "calibration_batches": len(calibration_batches),
        "temperature": temperature,
        "lbfgs_closure_evaluations": evaluations,
        "calibration_loss_before": initial_calibration_loss,
        "calibration_loss_after": calibration_loss,
        "calibration_forward_flops": float(calibration_counter.get_total_flops()),
        "calibration_aux_ops": calibration_aux_ops,
        "calibration_total_flops": float(calibration_counter.get_total_flops()) + calibration_aux_ops,
        "final_evaluation_flops": float(evaluation_counter.get_total_flops()),
        "uncalibrated": uncalibrated,
        "calibrated": calibrated,
    }


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
    name = f"{args.arm}_calibrated"
    cum_train = 0.0
    cum_calibrated = 0.0
    cum_aux = 0.0
    cum_eval = 0.0
    history = []
    active_history = []
    calibration_result = None
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
            "format": "equal_opportunity_temperature_calibration_v1",
            "split": args.split,
            "completed": completed,
            "arm": name,
            "base_arm": args.arm,
            "program": None if program is None else program.to_dict(),
            "body_seed": body_seed,
            "online_seed": online_seed,
            "checkpoint": os.path.abspath(local.ckpt),
            "execution": "ordinary dense training" if dense else "dense forward with custom active-row-only FFN backward",
            "calibration": calibration_result,
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
            dense_metrics = batch_metrics(model, eval_batches, cfg)
        cum_eval += float(counter.get_total_flops())
        row = {
            "step": step,
            "heldout_loss": dense_metrics["loss"],
            "heldout_acc": dense_metrics["acc"],
            "cum_train_flops": cum_train,
            "cum_full_cpu_calibrated_flops": cum_calibrated,
            "cum_aux_ops": cum_aux,
            "cum_total_flops_with_aux_estimate": cum_calibrated + cum_aux,
            "cum_eval_flops": cum_eval,
            "wallclock_s": time.time() - start,
            "mean_active_fraction": (float(np.mean(active_history))
                                     if active_history else None),
        }
        history.append(row)
        print(f"[{name}/b{body_seed}] step {step} | dense "
              f"{dense_metrics['loss']:.6f} | acc {dense_metrics['acc']:.6f} | total+aux "
              f"{row['cum_total_flops_with_aux_estimate'] / 1e12:.3f}T | "
              f"{row['wallclock_s']:.0f}s", flush=True)
        save(False)

    evaluate(0)
    for index, batch in enumerate(batches):
        step = index + 1
        model.train()
        active = []
        operations = []
        gates = gates_for(batch, active, operations)
        with FlopCounterMode(display=False) as train_counter:
            optimizer.zero_grad(set_to_none=True)
            loss, info = training_forward(model, batch, cfg, gates, dense)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        cum_train += float(train_counter.get_total_flops())
        tokens_per_layer = batch[0].numel()
        active_count = tokens_per_layer * cfg.n_layers if dense else int(sum(active))
        inactive_count = tokens_per_layer * cfg.n_layers - active_count
        dense_step_flops = 580235821056 * args.batch_size // 32
        step_calibrated = dense_step_flops - 8 * inactive_count * cfg.d_model * cfg.d_ff
        cum_calibrated += step_calibrated
        gather_scatter_ops = 4 * active_count * (cfg.d_model + cfg.d_ff)
        cum_aux += sum(operations) + (0 if dense else gather_scatter_ops)
        active_history.append(
            1.0 if dense else
            active_count / (cfg.n_layers * int((batch[0] != TOKENIZER.pad_id).sum())))
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)

    calibration_cfg = copy.deepcopy(cfg)
    calibration_cfg.held_out_seed = args.calibration_seed
    calibration_held_out = task.build_held_out(calibration_cfg)
    calibration_batches = evaluation_batches(
        task, calibration_cfg, calibration_held_out, args.calibration_batches)
    final_batches = evaluation_batches(task, cfg, held_out, args.eval_batches)
    calibration_result = calibrate(model, calibration_batches, final_batches, calibration_cfg)
    calibration_result["training_total_flops"] = history[-1]["cum_total_flops_with_aux_estimate"]
    calibration_result["total_with_calibration_flops"] = (
        history[-1]["cum_total_flops_with_aux_estimate"]
        + calibration_result["calibration_total_flops"])
    calibration_result["wallclock_s"] = time.time() - start
    save(True)
    print(f"[{name}/b{body_seed}] temperature {calibration_result['temperature']:.6f} | "
          f"calibrated loss {calibration_result['calibrated']['loss']:.6f} | "
          f"uncalibrated {calibration_result['uncalibrated']['loss']:.6f} | "
          f"total {calibration_result['total_with_calibration_flops'] / 1e12:.3f}T", flush=True)
    print(f"saved {out}", flush=True)
    return {"arm": name, "path": out, "calibration": calibration_result}


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
    run.add_argument("--eval-every", type=int, default=20)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--calibration-batches", type=int, default=8)
    run.add_argument("--calibration-seed", type=int, default=24680)
    run.add_argument("--seed", type=int, default=9400)
    run.add_argument("--held-out-seed", type=int, default=54321)
    run.add_argument("--split", choices=["development", "confirmation"],
                     default="development")
    args = parser.parse_args()
    program = PROGRAM_BY_NAME.get(args.arm)
    run_arm(args, args.body_seed, args.budget, program)


if __name__ == "__main__":
    main()
