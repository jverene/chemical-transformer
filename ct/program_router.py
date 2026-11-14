"""Inspectable fixed allocation programs with exact executable budgets."""
import argparse
from dataclasses import dataclass
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .inverse_oracle import load_model, scores_to_layer_mask, sequence_budget
from .meta_router import batch_metrics, forward_train
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


@dataclass(frozen=True)
class Program:
    name: str
    terms: tuple

    @property
    def bases(self):
        return tuple(dict.fromkeys(name for name, _ in self.terms))

    @property
    def expression(self):
        pieces = []
        for feature, coefficient in self.terms:
            sign = "+" if coefficient >= 0 else "-"
            magnitude = abs(coefficient)
            term = feature if magnitude == 1 else f"{magnitude:g}*{feature}"
            pieces.append((sign, term))
        first_sign, first_term = pieces[0]
        text = ("" if first_sign == "+" else "-") + first_term
        return text + "".join(f" {sign} {term}" for sign, term in pieces[1:])

    def to_dict(self):
        return {
            "name": self.name,
            "expression": self.expression,
            "terms": [[name, coefficient] for name, coefficient in self.terms],
        }


PROGRAMS = (
    Program("high_rms", (("rms", 1.0),)),
    Program("low_rms", (("rms", -1.0),)),
    Program("high_abs_mean", (("abs_mean", 1.0),)),
    Program("low_abs_mean", (("abs_mean", -1.0),)),
    Program("high_max_abs", (("max_abs", 1.0),)),
    Program("low_max_abs", (("max_abs", -1.0),)),
    Program("high_position", (("position", 1.0),)),
    Program("low_position", (("position", -1.0),)),
    Program("high_outlier", (("max_abs", 1.0), ("rms", -1.0))),
    Program("low_outlier", (("max_abs", -1.0), ("rms", 1.0))),
)
PROGRAM_BY_NAME = {program.name: program for program in PROGRAMS}


@torch.no_grad()
def base_values(name, x, pad):
    if name == "rms":
        return torch.sqrt(x.detach().square().mean(dim=-1) + 1e-6)
    if name == "abs_mean":
        return x.detach().mean(dim=-1).abs()
    if name == "max_abs":
        return x.detach().abs().amax(dim=-1)
    if name == "position":
        positions = torch.arange(
            x.shape[1], device=x.device, dtype=x.dtype) / max(x.shape[1] - 1, 1)
        return positions.unsqueeze(0).expand(x.shape[0], -1)
    raise ValueError(f"unknown program feature: {name}")


@torch.no_grad()
def normalized_values(values, pad):
    output = torch.zeros_like(values)
    for b in range(values.shape[0]):
        valid = pad[b]
        current = values[b, valid]
        if current.numel() > 1:
            output[b, valid] = ((current - current.mean()) /
                                (current.std(unbiased=False) + 1e-6))
    return output


@torch.no_grad()
def program_scores(program, x, pad):
    if x.ndim != 3 or pad.shape != x.shape[:2]:
        raise ValueError("program expects x[B,S,D] and pad[B,S]")
    values = {name: base_values(name, x, pad) for name in program.bases}
    if len(program.terms) > 1:
        values = {name: normalized_values(value, pad)
                  for name, value in values.items()}
    scores = sum(coefficient * values[name] for name, coefficient in program.terms)
    return scores.detach() * pad.to(scores.dtype)


def feature_ops_per_token(program, d_model):
    costs = {"rms": d_model + 3, "abs_mean": 2 * d_model + 2,
             "max_abs": 2 * d_model, "position": 1}
    operations = sum(costs[name] for name in program.bases)
    if len(program.terms) > 1:
        operations += 5 * len(program.bases) + 2 * len(program.terms)
    elif program.terms[0][1] != 1:
        operations += 1
    return operations


def program_gates(program, n_layers, pad, budget, active_sink=None, op_sink=None):
    def make_gate():
        def gate(x):
            scores = program_scores(program, x, pad)
            hard = scores_to_layer_mask(scores.unsqueeze(0), pad, budget)[0]
            if active_sink is not None:
                active_sink.append(float(hard.sum()))
            if op_sink is not None:
                op_sink.append(
                    int(pad.sum()) * feature_ops_per_token(program, x.shape[-1])
                    + x.shape[0] * x.shape[1])
            return hard
        return gate
    return [make_gate() for _ in range(n_layers)]


def random_gates(n_layers, shape, pad, budget, generator,
                 active_sink=None, op_sink=None):
    gates = []
    for _ in range(n_layers):
        mask = scores_to_layer_mask(
            torch.empty(1, *shape, device=pad.device), pad, budget,
            generator=generator, random_mask=True)[0]
        gates.append(mask)
        if active_sink is not None:
            active_sink.append(float(mask.sum()))
        if op_sink is not None:
            op_sink.append(shape[0] * shape[1])
    return gates


def non_anchor_budget(budget, anchor_every):
    if anchor_every < 2:
        raise ValueError("anchor_every must be at least 2")
    anchor_fraction = 1.0 / anchor_every
    result = (budget - anchor_fraction) / (1 - anchor_fraction)
    if not 0 <= result <= 1:
        raise ValueError("average budget is incompatible with dense anchors")
    return result


def run_arm(args, body_seed, budget, program, anchor_every=None):
    online_seed = args.seed + body_seed
    base_name = "random" if program is None else program.name
    name = base_name if anchor_every is None else f"{base_name}_anchor{anchor_every}"
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{name}_seed{online_seed}.json")
    config = {k: v for k, v in vars(args).items() if k != "handler"}
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        expected_program = None if program is None else program.to_dict()
        if (existing.get("completed") is True
                and existing.get("config") == config
                and existing.get("arm") == name
                and existing.get("program") == expected_program
                and existing.get("body_seed") == body_seed
                and existing.get("online_seed") == online_seed
                and existing.get("checkpoint") == os.path.abspath(
                    args.ckpt_template.format(seed=body_seed))
                and existing.get("budget") == budget
                and existing.get("anchor_every") == anchor_every
                and existing.get("final", {}).get("step") == args.steps):
            print(f"reusing completed {out}", flush=True)
            return {**existing, "path": out}
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
               for _ in range(args.steps * args.grad_accum)]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.body_lr, weight_decay=args.weight_decay)
    mask_generator = torch.Generator().manual_seed(online_seed * 100003 + 811)
    eval_generator = torch.Generator().manual_seed(online_seed * 100003 + 813)
    set_seed(online_seed * 100003 + 17)
    sparse_budget = (budget if anchor_every is None else
                     non_anchor_budget(budget, anchor_every))
    cum_train = 0.0
    cum_aux = 0.0
    cum_eval = 0.0
    cum_eval_aux = 0.0
    history = []
    active_history = []
    start = time.time()

    def gates_for(batch, active_sink, op_sink, evaluation=False, step=None):
        if (not evaluation and anchor_every is not None
                and step % anchor_every == 0):
            return None
        pad = batch[0] != TOKENIZER.pad_id
        active_budget = budget if evaluation else sparse_budget
        if program is not None:
            return program_gates(
                program, cfg.n_layers, pad, active_budget, active_sink, op_sink)
        return random_gates(
            cfg.n_layers, batch[0].shape, pad, active_budget,
            eval_generator if evaluation else mask_generator, active_sink, op_sink)

    def save(completed=False):
        result = {
            "format": "fixed_allocation_program_screen_v1",
            "split": args.split,
            "completed": completed,
            "arm": name,
            "program": None if program is None else program.to_dict(),
            "body_seed": body_seed,
            "online_seed": online_seed,
            "checkpoint": os.path.abspath(local.ckpt),
            "execution": "on-policy pre-FFN features; exact per-layer Top-B; gather/scatter; no router learning",
            "accounting": "executed matmul/attention FLOPs plus separately estimated feature and selection operations",
            "config": {k: v for k, v in vars(args).items() if k != "handler"},
            "budget": budget,
            "anchor_every": anchor_every,
            "non_anchor_budget": sparse_budget,
            "sparse_evaluation_budget": budget,
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
        eval_rng = random.Random(999)
        eval_batch_size = args.eval_batch_size or args.batch_size
        eval_batches = [get_batch(task, cfg, eval_rng, eval_batch_size, held_out)
                        for _ in range(args.eval_batches)]
        eval_ops = []
        with FlopCounterMode(display=False) as counter:
            dense = batch_metrics(model, eval_batches, cfg)
            sparse = batch_metrics(
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
            "cum_aux_ops": cum_aux,
            "cum_total_flops_with_aux_estimate": cum_train + cum_aux,
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
    for index in range(args.steps):
        step = index + 1
        micro_batches = batches[index * args.grad_accum:(index + 1) * args.grad_accum]
        model.train()
        active = []
        step_ops = []
        all_sparse = True
        with FlopCounterMode(display=False) as train_counter:
            optimizer.zero_grad(set_to_none=True)
            token_counts = [int((batch[1] != TOKENIZER.pad_id).sum())
                            for batch in micro_batches]
            total_tokens = sum(token_counts)
            for batch, token_count in zip(micro_batches, token_counts):
                gates = gates_for(batch, active, step_ops, step=step)
                all_sparse = all_sparse and gates is not None
                loss, info = forward_train(model, batch, cfg, gates, gates is not None)
                (loss * (token_count / total_tokens)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        cum_train += float(train_counter.get_total_flops())
        cum_aux += sum(step_ops)
        if not all_sparse:
            active_history.append(1.0)
        else:
            pad_total = sum(int((batch[0] != TOKENIZER.pad_id).sum())
                            for batch in micro_batches)
            active_history.append(sum(active) / (cfg.n_layers * pad_total))
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
    save(True)
    print(f"saved {out}", flush=True)
    return {
        "arm": name,
        "path": out,
        "program": None if program is None else program.to_dict(),
        "body_seed": body_seed,
        "budget": budget,
        "anchor_every": anchor_every,
        "non_anchor_budget": sparse_budget,
        "history": history,
        "final": history[-1],
    }


def summarize_regime(results):
    random_result = next(result for result in results if result["arm"] == "random")
    arms = {}
    for result in results:
        row = result["final"]
        paired = {}
        for index, history_row in enumerate(result["history"]):
            step = history_row["step"]
            if step == 0:
                continue
            control = random_result["history"][index]
            assert control["step"] == step
            paired[str(step)] = {
                "minus_random_loss": (history_row["sparse_heldout_loss"]
                                      - control["sparse_heldout_loss"]),
                "minus_random_accuracy": (history_row["sparse_heldout_acc"]
                                          - control["sparse_heldout_acc"]),
            }
        arms[result["arm"]] = {
            "final_sparse_loss": row["sparse_heldout_loss"],
            "final_sparse_accuracy": row["sparse_heldout_acc"],
            "cum_train_flops": row["cum_train_flops"],
            "cum_aux_ops": row["cum_aux_ops"],
            "cum_total_flops_with_aux_estimate": row["cum_total_flops_with_aux_estimate"],
            "mean_active_fraction": row["mean_active_fraction"],
            "paired_by_step": paired,
        }
    return {"format": "fixed_allocation_program_regime_summary_v1", "arms": arms}


def run_screen(args):
    selected_programs = [PROGRAM_BY_NAME[name] for name in args.programs.split(",")]
    regimes = []
    for body_seed in [int(value) for value in args.body_seeds.split(",")]:
        for budget in [float(value) for value in args.budgets.split(",")]:
            regime_args = argparse.Namespace(**vars(args))
            regime_args.outdir = os.path.join(
                args.outdir, f"body{body_seed}_budget{budget:g}")
            results = [run_arm(regime_args, body_seed, budget, None)]
            results.extend(run_arm(regime_args, body_seed, budget, program)
                           for program in selected_programs)
            summary = summarize_regime(results)
            with open(os.path.join(regime_args.outdir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            regimes.append({
                "body_seed": body_seed,
                "budget": budget,
                "summary": summary,
            })
    programs = {}
    for program in selected_programs:
        comparisons = []
        for regime in regimes:
            row = regime["summary"]["arms"][program.name]
            comparisons.append({
                "body_seed": regime["body_seed"],
                "budget": regime["budget"],
                "minus_random_loss": row["paired_by_step"][str(args.steps)]["minus_random_loss"],
                "minus_random_accuracy": row["paired_by_step"][str(args.steps)]["minus_random_accuracy"],
            })
        losses = [row["minus_random_loss"] for row in comparisons]
        programs[program.name] = {
            "program": program.to_dict(),
            "regimes": comparisons,
            "wins": sum(value < 0 for value in losses),
            "mean_minus_random_loss": float(np.mean(losses)),
            "worst_minus_random_loss": max(losses),
            "eligible_all_regimes_win": all(value < 0 for value in losses),
        }
    eligible = [name for name, row in programs.items() if row["eligible_all_regimes_win"]]
    selected = (min(eligible, key=lambda name: programs[name]["mean_minus_random_loss"])
                if eligible else None)
    output = {
        "format": "fixed_allocation_program_screen_summary_v1",
        "split": args.split,
        "steps": args.steps,
        "programs": programs,
        "selected_for_longer_development": selected,
        "decision": ("advance one frozen program to longer development" if selected
                     else "kill fixed-program route: no program beat random in every regime"),
        "note": "short development screen only; no Level 2 or Level 3 claim and no dense-fewer frontier claim",
    }
    with open(os.path.join(args.outdir, "screen_summary.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2), flush=True)


def self_test():
    torch.manual_seed(20261113)
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    pad = torch.tensor([[True, True, True, True, False],
                        [True, True, True, False, False]])
    budget = 0.5
    active = []
    operations = []
    gates = program_gates(
        PROGRAM_BY_NAME["high_position"], 2, pad, budget, active, operations)
    expected_counts = []
    for gate in gates:
        mask = gate(x)
        assert mask.shape == (2, 5) and not mask.requires_grad
        for b in range(2):
            count = sequence_budget(pad[b], budget, 1)
            expected_counts.append(count)
            assert int(mask[b].sum()) == count
            expected = torch.arange(5)[pad[b]][-count:]
            assert torch.equal(mask[b].nonzero().squeeze(-1), expected)
        assert float(mask[~pad].sum()) == 0
    scores = program_scores(PROGRAM_BY_NAME["high_outlier"], x, pad)
    assert scores.shape == (2, 5) and torch.isfinite(scores).all()
    assert not scores.requires_grad
    random_active = []
    random_ops = []
    random_masks = random_gates(
        2, (2, 5), pad, budget, torch.Generator().manual_seed(7),
        random_active, random_ops)
    for mask in random_masks:
        for b in range(2):
            assert int(mask[b].sum()) == sequence_budget(pad[b], budget, 1)
        assert float(mask[~pad].sum()) == 0
    result = {
        "programs": len(PROGRAMS),
        "exact_budget_checks": len(expected_counts),
        "high_position_selection": "passed",
        "padding_error": 0.0,
        "program_ops_positive": all(value > 0 for value in operations),
        "random_ops_positive": all(value > 0 for value in random_ops),
        "outlier_scores_finite": True,
        "passed": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(handler=lambda args: self_test())
    screen = sub.add_parser("screen")
    screen.add_argument("--ckpt-template", default=(
        "results-inverse-ckpt/tagged/predictor_stage1_seed{seed}.pt"))
    screen.add_argument("--outdir", required=True)
    screen.add_argument("--device", default=None)
    screen.add_argument("--size", default="small")
    screen.add_argument("--seq-len", type=int, default=256)
    screen.add_argument("--task", default="tagged")
    screen.add_argument("--body-seeds", default="4,8")
    screen.add_argument("--budgets", default="0.375,0.5")
    screen.add_argument("--programs", default=",".join(PROGRAM_BY_NAME))
    screen.add_argument("--steps", type=int, default=50)
    screen.add_argument("--batch-size", type=int, default=32)
    screen.add_argument("--grad-accum", type=int, default=1)
    screen.add_argument("--body-lr", type=float, default=0.0003)
    screen.add_argument("--weight-decay", type=float, default=0.01)
    screen.add_argument("--dropout", type=float, default=0.1)
    screen.add_argument("--eval-every", type=int, default=25)
    screen.add_argument("--eval-batches", type=int, default=4)
    screen.add_argument("--eval-batch-size", type=int, default=None)
    screen.add_argument("--held-out-seed", type=int, default=None)
    screen.add_argument("--seed", type=int, default=9400)
    screen.add_argument("--split", choices=["development", "confirmation"],
                        default="development")
    screen.set_defaults(handler=run_screen)
    args = parser.parse_args()
    if hasattr(args, "grad_accum") and args.grad_accum < 1:
        parser.error("--grad-accum must be positive")
    if getattr(args, "eval_batch_size", None) is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
