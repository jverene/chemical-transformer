"""One-pass lagged global layer budgets with exact integer compute."""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .inverse_oracle import load_model, scores_to_layer_mask, sequence_budget
from .meta_router import batch_metrics, evaluation_batches, forward_train
from .program_router import base_values
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


@torch.no_grad()
def scores_to_count_mask(scores, pad, counts):
    output = torch.zeros_like(scores)
    for b in range(scores.shape[0]):
        count = int(counts[b])
        if count == 0:
            continue
        valid = pad[b]
        ranking = scores[b].masked_fill(~valid, -torch.inf)
        indices = torch.topk(ranking, count).indices
        output[b, indices] = 1.0
    return output


@torch.no_grad()
def allocate_counts(fractions, pad, budget, n_layers):
    if fractions.ndim != 2 or fractions.shape[0] != n_layers:
        raise ValueError("fractions must have shape [layers,batch] or [layers,1]")
    if fractions.shape[1] == 1:
        fractions = fractions.expand(-1, pad.shape[0])
    if fractions.shape[1] != pad.shape[0]:
        raise ValueError("fraction batch dimension does not match pad")
    output = torch.zeros(
        n_layers, pad.shape[0], dtype=torch.long, device=pad.device)
    for b in range(pad.shape[0]):
        n_valid = int(pad[b].sum())
        target = sequence_budget(pad[b], budget, n_layers)
        if n_valid == 0:
            continue
        quotas = fractions[:, b].clamp(0, 1).float() * n_valid
        quotas = quotas * (target / quotas.sum().clamp_min(1e-12))
        counts = quotas.floor().clamp(max=n_valid).long()
        remainder = target - int(counts.sum())
        order = (quotas - counts.float()).argsort(descending=True)
        while remainder > 0:
            progressed = False
            for index in order:
                if counts[index] < n_valid:
                    counts[index] += 1
                    remainder -= 1
                    progressed = True
                    if remainder == 0:
                        break
            if not progressed:
                raise RuntimeError("could not realize exact global budget")
        output[:, b] = counts
    return output


@torch.no_grad()
def next_global_fractions(scores, pad, budget):
    stacked = torch.stack(scores)
    mask = scores_to_layer_mask(stacked, pad, budget)
    counts = mask.sum(dim=-1)
    valid = pad.sum(dim=-1).clamp_min(1)
    return (counts / valid).mean(dim=1, keepdim=True)


def lagged_gates(fractions, n_layers, pad, budget, active_sink, op_sink,
                 score_sink=None, replay_counts=None, generator=None):
    counts = (allocate_counts(fractions, pad, budget, n_layers)
              if replay_counts is None else replay_counts.to(pad.device))
    if replay_counts is not None:
        gates = []
        for layer in range(n_layers):
            scores = torch.rand(
                pad.shape, generator=generator).to(pad.device)
            mask = scores_to_count_mask(scores, pad, counts[layer])
            gates.append(mask)
            active_sink.append(float(mask.sum()))
            op_sink.append(2 * int(pad.sum()) + pad.shape[0] * pad.shape[1])
        return gates, counts

    def make_gate(layer):
        def gate(x):
            scores = -base_values("max_abs", x, pad)
            if score_sink is not None:
                score_sink[layer] = scores.detach()
            mask = scores_to_count_mask(scores, pad, counts[layer])
            active_sink.append(float(mask.sum()))
            op_sink.append(
                2 * int(pad.sum()) * x.shape[-1] + pad.shape[0] * pad.shape[1])
            return mask
        return gate
    return [make_gate(layer) for layer in range(n_layers)], counts


def run_arm(args, body_seed, budget, replay_path=None):
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
    generator = torch.Generator().manual_seed(online_seed * 100003 + 817)
    set_seed(online_seed * 100003 + 17)
    replay = None
    if replay_path is not None:
        with open(replay_path) as f:
            replay = json.load(f)
    arm = "lagged_global_low_max_abs" if replay is None else "random_replay"
    fractions = torch.full(
        (cfg.n_layers, 1), budget, device=cfg.device)
    train_schedule = {}
    eval_schedule = {}
    if replay is not None:
        train_schedule = {
            row["step"]: torch.tensor(row["counts"]).t()
            for row in replay["budget_history"]}
        eval_schedule = {
            row["step"]: [torch.tensor(counts).t() for counts in row["batches"]]
            for row in replay["eval_budget_history"]}
    cum_train = 0.0
    cum_aux = 0.0
    cum_eval = 0.0
    cum_eval_aux = 0.0
    history = []
    active_history = []
    budget_history = []
    eval_budget_history = []
    fraction_history = []
    start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{arm}_seed{online_seed}.json")

    def gates_for(batch, active, operations, step, evaluation=False,
                  eval_batch_index=None):
        pad = batch[0] != TOKENIZER.pad_id
        replay_counts = None
        if replay is not None:
            replay_counts = (eval_schedule[step][eval_batch_index]
                             if evaluation else train_schedule[step])
        score_sink = None if replay is not None or evaluation else [None] * cfg.n_layers
        gates, counts = lagged_gates(
            fractions, cfg.n_layers, pad, budget, active, operations,
            score_sink=score_sink, replay_counts=replay_counts,
            generator=generator)
        return gates, counts, score_sink

    def save(completed=False):
        result = {
            "format": "lagged_global_low_max_abs_v1",
            "split": args.split,
            "completed": completed,
            "arm": arm,
            "body_seed": body_seed,
            "online_seed": online_seed,
            "checkpoint": os.path.abspath(local.ckpt),
            "execution": "previous-step global raw-score layer budgets; current on-policy low-max-abs token selection; exact largest-remainder integer counts; one forward and one ordinary backward",
            "causality": "uniform budgets on the first step; thereafter one batch-averaged layer-budget vector from the just-completed step; no planning pass and no additional gradient computation",
            "accounting": "executed model FLOPs plus separately estimated score, global selection, replay, and integer-rounding operations",
            "config": {k: v for k, v in vars(args).items() if k != "handler"},
            "budget": budget,
            "replay_path": replay_path,
            "history": history,
            "active_history": active_history,
            "budget_history": budget_history,
            "eval_budget_history": eval_budget_history,
            "fraction_history": fraction_history,
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
        eval_counts = []
        eval_index = [0]
        fractions_before = fractions.detach().clone()

        def factory(batch):
            index = eval_index[0]
            eval_index[0] += 1
            gates, counts, _ = gates_for(
                batch, [], eval_ops, step, evaluation=True, eval_batch_index=index)
            eval_counts.append(counts.t().detach().cpu().tolist())
            return gates

        with FlopCounterMode(display=False) as counter:
            dense = batch_metrics(model, eval_batches, cfg)
            sparse = batch_metrics(model, eval_batches, cfg, factory)
        if not torch.equal(fractions, fractions_before):
            raise RuntimeError("evaluation changed lagged global budgets")
        cum_eval += float(counter.get_total_flops())
        cum_eval_aux += sum(eval_ops)
        if replay is None:
            eval_budget_history.append({"step": step, "batches": eval_counts})
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
        operations = []
        gates, counts, score_sink = gates_for(batch, active, operations, step)
        with FlopCounterMode(display=False) as train_counter:
            optimizer.zero_grad(set_to_none=True)
            loss, info = forward_train(model, batch, cfg, gates, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        cum_train += float(train_counter.get_total_flops())
        cum_aux += sum(operations)
        active_history.append(
            sum(active) / (cfg.n_layers * int((batch[0] != TOKENIZER.pad_id).sum())))
        budget_history.append({"step": step, "counts": counts.t().detach().cpu().tolist()})
        if replay is None:
            fractions = next_global_fractions(
                score_sink, batch[0] != TOKENIZER.pad_id, budget)
            planning_ops = (cfg.n_layers * batch[0].numel()
                            + 10 * cfg.n_layers * batch[0].shape[0])
            cum_aux += planning_ops
            fraction_history.append({
                "step": step,
                "next_layer_fractions": fractions.squeeze(1).detach().cpu().tolist(),
            })
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
    save(True)
    print(f"saved {out}", flush=True)
    return {"arm": arm, "path": out, "history": history, "final": history[-1]}


def self_test():
    torch.manual_seed(7)
    for budget in (0.375, 0.5, 0.75):
        fractions = torch.rand(2, 1)
        pad = torch.tensor([[True] * 5,
                            [True, True, True, False, False]])
        counts = allocate_counts(fractions, pad, budget, 2)
        for b in range(2):
            assert int(counts[:, b].sum()) == sequence_budget(pad[b], budget, 2)
            assert int(counts[:, b].max()) <= int(pad[b].sum())
    x = torch.randn(2, 5, 8)
    pad = torch.tensor([[True] * 5, [True, True, True, False, False]])
    fractions = torch.full((2, 1), 0.75)
    active = []
    operations = []
    scores = [None, None]
    gates, counts = lagged_gates(
        fractions, 2, pad, 0.75, active, operations, score_sink=scores)
    masks = [gate(x) for gate in gates]
    for b in range(2):
        assert sum(int(mask[b].sum()) for mask in masks) == sequence_budget(pad[b], 0.75, 2)
    next_fractions = next_global_fractions(scores, pad, 0.75)
    assert next_fractions.shape == (2, 1)
    replay_active = []
    replay_ops = []
    replay_masks, replay_counts = lagged_gates(
        None, 2, pad, 0.75, replay_active, replay_ops,
        replay_counts=counts, generator=torch.Generator().manual_seed(8))
    for b in range(2):
        assert torch.equal(replay_counts[:, b], counts[:, b])
        assert sum(int(mask[b].sum()) for mask in replay_masks) == sequence_budget(pad[b], 0.75, 2)
        for mask in replay_masks:
            assert float(mask[b, ~pad[b]].sum()) == 0
    result = {
        "largest_remainder_exact_budget": "passed",
        "current_token_score_capture": "passed",
        "lagged_fraction_shape": "passed",
        "random_replay_budget_identity": "passed",
        "padding_error": 0.0,
        "planning_forward_passes": 0,
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
    run.add_argument("--body-seed", type=int, required=True)
    run.add_argument("--budget", type=float, default=0.75)
    run.add_argument("--steps", type=int, default=116)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--body-lr", type=float, default=0.0003)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--eval-every", type=int, default=29)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--seed", type=int, default=9400)
    run.add_argument("--held-out-seed", type=int, default=54321)
    run.add_argument("--split", choices=["development", "confirmation"],
                     default="development")
    run.add_argument("--replay-path", default=None)
    run.set_defaults(handler=lambda args: run_arm(
        args, args.body_seed, args.budget, args.replay_path))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
