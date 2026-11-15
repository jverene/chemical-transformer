"""Sparse token FFN execution with a shared learned zero-input response."""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from .inverse_oracle import load_model
from .meta_router import evaluation_batches
from .program_router import PROGRAM_BY_NAME, program_gates, random_gates
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def forward_null(model, batch, cfg, gates):
    x, y = batch
    logits, info = model(
        x, pad_mask=x != TOKENIZER.pad_id,
        meta={"diffs": None, "gather_ffn": True, "null_ffn": True},
        oracle_gates=gates)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), y.view(-1),
        ignore_index=TOKENIZER.pad_id)
    return loss, info


def null_metrics(model, batches, cfg, gate_factory=None):
    loss_sum = 0.0
    loss_n = 0
    correct = 0
    answers = 0
    for batch in batches:
        x, y, _, answer_mask = batch
        body_batch = batch[:2]
        gates = gate_factory(body_batch) if gate_factory is not None else None
        meta = {"diffs": None, "gather_ffn": gates is not None,
                "null_ffn": gates is not None}
        logits, _ = model(
            x, pad_mask=x != TOKENIZER.pad_id, meta=meta, oracle_gates=gates)
        loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size), y.view(-1),
            ignore_index=TOKENIZER.pad_id)
        count = int((y != TOKENIZER.pad_id).sum())
        loss_sum += float(loss.detach()) * count
        loss_n += count
        pred = logits.argmax(dim=-1)
        valid = y != TOKENIZER.pad_id
        shifted_answers = torch.roll(answer_mask, -1, dims=1)
        shifted_answers[:, -1] = 0
        selected = valid & (shifted_answers == 1)
        correct += int(((pred == y) & selected).sum())
        answers += int(selected.sum())
    return {"loss": loss_sum / max(loss_n, 1),
            "acc": correct / max(answers, 1)}


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
    eval_generator = torch.Generator().manual_seed(online_seed * 100003 + 813)
    set_seed(online_seed * 100003 + 17)
    name = "random_null" if program is None else f"{program.name}_null"
    cum_train = 0.0
    cum_calibrated = 0.0
    cum_aux = 0.0
    cum_eval = 0.0
    cum_eval_aux = 0.0
    history = []
    active_history = []
    start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{name}_seed{online_seed}.json")

    def gates_for(batch, active, operations, evaluation=False):
        pad = batch[0] != TOKENIZER.pad_id
        if program is not None:
            return program_gates(
                program, cfg.n_layers, pad, budget, active, operations)
        return random_gates(
            cfg.n_layers, batch[0].shape, pad, budget,
            eval_generator if evaluation else mask_generator, active, operations)

    def save(completed=False):
        result = {
            "format": "shared_null_state_ffn_program_v1",
            "split": args.split,
            "completed": completed,
            "arm": name,
            "program": None if program is None else program.to_dict(),
            "body_seed": body_seed,
            "online_seed": online_seed,
            "checkpoint": os.path.abspath(local.ckpt),
            "execution": "gathered active-token FFN; one shared FFN(0) per layer with inactive-token output broadcast and aggregated autograd; ordinary sparse inference uses the same rule",
            "dropout": "active rows use ordinary per-row dropout; the single shared null row uses one dropout mask per layer and is deterministic during evaluation",
            "accounting": "CPU-calibrated dense forward/backward minus realized inactive-row FFN savings plus ten null-row matmul equivalents per nonempty null layer; scoring, gather/scatter, null broadcast, and inactive-gradient reduction estimates added separately",
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
        nonlocal cum_eval, cum_eval_aux
        model.eval()
        eval_batches = evaluation_batches(task, cfg, held_out, args.eval_batches)
        eval_ops = []
        with FlopCounterMode(display=False) as counter:
            dense = null_metrics(model, eval_batches, cfg)
            sparse = null_metrics(
                model, eval_batches, cfg,
                lambda batch: gates_for(batch, [], eval_ops, evaluation=True))
        cum_eval += float(counter.get_total_flops())
        cum_eval_aux += sum(eval_ops)
        row = {
            "step": step,
            "heldout_loss": dense["loss"],
            "heldout_acc": dense["acc"],
            "sparse_heldout_loss": sparse["loss"],
            "sparse_heldout_acc": sparse["acc"],
            "cum_train_flops": cum_train,
            "cum_full_cpu_calibrated_flops": cum_calibrated,
            "cum_aux_ops": cum_aux,
            "cum_total_flops_with_aux_estimate": cum_calibrated + cum_aux,
            "cum_eval_flops": cum_eval,
            "cum_eval_aux_ops": cum_eval_aux,
            "wallclock_s": time.time() - start,
            "mean_active_fraction": (float(np.mean(active_history))
                                     if active_history else None),
        }
        history.append(row)
        print(f"[{name}/b{body_seed}/B{budget:g}] step {step} | dense "
              f"{dense['loss']:.6f} | sparse {sparse['loss']:.6f} | total+aux "
              f"{row['cum_total_flops_with_aux_estimate'] / 1e12:.3f}T | active "
              f"{row['mean_active_fraction']} | {row['wallclock_s']:.0f}s", flush=True)
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
            loss, info = forward_null(model, batch, cfg, gates)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        cum_train += float(train_counter.get_total_flops())
        active_counts = [int(value) for value in active]
        tokens_per_layer = batch[0].numel()
        inactive_counts = [tokens_per_layer - value for value in active_counts]
        dense_step_flops = 580235821056 * args.batch_size // 32
        null_layers = sum(value > 0 for value in inactive_counts)
        step_calibrated = (
            dense_step_flops
            - 12 * sum(inactive_counts) * cfg.d_model * cfg.d_ff
            + 10 * null_layers * cfg.d_model * cfg.d_ff)
        cum_calibrated += step_calibrated
        active_count = sum(active_counts)
        gather_scatter_ops = 4 * active_count * (cfg.d_model + cfg.d_ff)
        null_broadcast_ops = 2 * sum(inactive_counts) * cfg.d_model
        cum_aux += sum(operations) + gather_scatter_ops + null_broadcast_ops
        active_history.append(
            active_count / (cfg.n_layers * int((batch[0] != TOKENIZER.pad_id).sum())))
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
    save(True)
    print(f"saved {out}", flush=True)
    return {"arm": name, "path": out, "history": history, "final": history[-1]}


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
    run.add_argument("--budget", type=float, default=0.5)
    run.add_argument("--arm", choices=["low_max_abs", "random"], required=True)
    run.add_argument("--steps", type=int, default=142)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--body-lr", type=float, default=0.0003)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--eval-every", type=int, default=20)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--seed", type=int, default=9400)
    run.add_argument("--held-out-seed", type=int, default=54321)
    run.add_argument("--split", choices=["development", "confirmation"],
                     default="development")
    args = parser.parse_args()
    program = None if args.arm == "random" else PROGRAM_BY_NAME[args.arm]
    run_arm(args, args.body_seed, args.budget, program)


if __name__ == "__main__":
    main()
