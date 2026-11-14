"""Low-overhead online policy routing using only ordinary training losses."""
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
from .meta_router import (
    batch_metrics, evaluation_batches, forward_train, interpolate_cost,
    permute_valid_scores, router_flops_per_token)
from .router_pilot import AllocationRouter
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def load_warm_router(path, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != "online_meta_allocation_router_v1":
        raise ValueError("unrecognized warm-router checkpoint")
    router = AllocationRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["model_state"])
    return router, checkpoint


@torch.no_grad()
def expected_budget_threshold(scores, pad, budget, temperature=1.0):
    values = scores[pad]
    if values.numel() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    lo = values.min() - 10 * temperature
    hi = values.max() + 10 * temperature
    for _ in range(40):
        midpoint = (lo + hi) / 2
        mean = torch.sigmoid((values - midpoint) / temperature).mean()
        if mean > budget:
            lo = midpoint
        else:
            hi = midpoint
    return ((lo + hi) / 2).detach()


def policy_gates(router, pad, policy_state, args, shuffle=False,
                 generator=None, evaluation=False, logprob_sink=None,
                 active_sink=None):
    def make_gate(layer_index):
        def gate(x):
            scores = router.forward_layer(x.detach(), layer_index)
            if shuffle:
                scores = permute_valid_scores(scores, pad, generator)
            threshold = policy_state["thresholds"][layer_index].detach()
            logits = (scores - threshold) / args.temperature
            probabilities = torch.sigmoid(logits)
            if evaluation:
                hard = (probabilities.detach() >= 0.5).float() * pad.float()
            else:
                noise = torch.rand(
                    scores.shape, generator=generator).to(scores.device)
                hard = ((noise < probabilities.detach()) & pad).float()
                if logprob_sink is not None:
                    log_probability = -F.binary_cross_entropy_with_logits(
                        logits, hard, reduction="none")
                    logprob_sink.append(log_probability[pad].sum() / int(pad.sum()))
                new_threshold = expected_budget_threshold(
                    scores.detach(), pad, args.budget, args.temperature)
                policy_state["thresholds"][layer_index] = new_threshold
            if active_sink is not None:
                active_sink.append(float(hard.sum()))
            return hard.detach()
        return gate
    return [make_gate(layer) for layer in range(router.n_layers)]


def random_gates(n_layers, shape, pad, budget, generator, active_sink=None):
    gates = []
    for _ in range(n_layers):
        noise = torch.rand(shape, generator=generator).to(pad.device)
        mask = ((noise < budget) & pad).float()
        gates.append(mask)
        if active_sink is not None:
            active_sink.append(float(mask.sum()))
    return gates


def policy_update(router, optimizer, pending_rewards):
    optimizer.zero_grad(set_to_none=True)
    advantages = torch.tensor(
        [value[1] for value in pending_rewards], device=router.layer.weight.device)
    objective = -torch.stack([
        advantage * log_probability
        for advantage, (log_probability, _, _) in zip(advantages, pending_rewards)
    ]).mean()
    objective.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("non-finite policy-router gradient")
    optimizer.step()
    analytical_backward = (
        2 * router_flops_per_token(router, router.norm.normalized_shape[0])
        * router.n_layers * sum(value[2] for value in pending_rewards))
    return {
        "router_objective": float(objective.detach()),
        "router_grad_norm": float(gradient_norm.detach()),
        "advantages": [float(value) for value in advantages],
        "analytical_backward_flops": analytical_backward,
    }


def run_arm(args, arm):
    set_seed(args.seed)
    local = argparse.Namespace(**vars(args))
    local.allow_legacy_extra_keys = True
    model, cfg = load_model(local)
    cfg.batch_size = args.batch_size
    cfg.dropout = args.dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.dropout
    task = get_task(args.task)
    held_out = task.build_held_out(cfg)
    rng = random.Random(args.seed * 100003 + 7)
    n_steps = args.steps // 2 if arm == "dense-fewer" else args.steps
    batches = [get_batch(task, cfg, rng, args.batch_size)[:2]
               for _ in range(n_steps)]
    uses_policy = arm in ("warm", "scratch", "shuffled")
    router = None
    router_optimizer = None
    if uses_policy:
        if arm == "scratch":
            router = AllocationRouter(
                cfg.d_model, cfg.n_layers, cfg.max_len,
                hidden=args.router_hidden).to(cfg.device)
            source = None
        else:
            router, source = load_warm_router(args.warm_router, cfg.device)
            source = {key: value for key, value in source.items()
                      if key != "model_state"}
        router_optimizer = torch.optim.AdamW(
            router.parameters(), lr=args.router_lr, weight_decay=0.0)
    policy_state = {
        "thresholds": torch.zeros(cfg.n_layers, device=cfg.device),
        "baseline": None,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.body_lr, weight_decay=args.weight_decay)
    mask_generator = torch.Generator().manual_seed(args.seed * 100003 + 811)
    shuffle_generator = torch.Generator().manual_seed(args.seed * 100003 + 812)
    eval_generator = torch.Generator().manual_seed(args.seed * 100003 + 813)
    set_seed(args.seed * 100003 + 17)
    cum_train = 0.0
    cum_policy = 0.0
    cum_eval = 0.0
    history = []
    update_history = []
    active_history = []
    pending_action = None
    pending_rewards = []
    start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{arm}_seed{args.seed}.json")

    def gates_for(batch, evaluation=False, logprob_sink=None, active_sink=None):
        pad = batch[0] != TOKENIZER.pad_id
        if uses_policy:
            return policy_gates(
                router, pad, policy_state, args, shuffle=arm == "shuffled",
                generator=eval_generator if evaluation else shuffle_generator,
                evaluation=evaluation, logprob_sink=logprob_sink,
                active_sink=active_sink)
        if arm == "random":
            return random_gates(
                cfg.n_layers, batch[0].shape, pad, args.budget,
                eval_generator if evaluation else mask_generator, active_sink)
        return None

    def save(completed=False):
        result = {
            "format": "ordinary_stream_policy_router_v1",
            "split": args.split,
            "completed": completed,
            "arm": arm,
            "seed": args.seed,
            "checkpoint": os.path.abspath(args.ckpt),
            "warm_router": None if arm == "scratch" else (
                os.path.abspath(args.warm_router) if uses_policy else None),
            "warm_source": source if uses_policy and source is not None else None,
            "optimizer": "ordinary AdamW every body step; delayed score-function router updates",
            "reward": "next ordinary training loss with EMA baseline; no additional reward forward pass",
            "execution": "causal per-token Bernoulli hard gates with gather/scatter; deterministic threshold evaluation",
            "accounting": "FlopCounterMode executed matmul/attention FLOPs; train/policy/eval separated; zero extra reward forwards",
            "config": {k: v for k, v in vars(args).items() if k != "handler"},
            "history": history,
            "update_history": update_history,
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
            dense = batch_metrics(model, eval_batches, cfg)
            sparse = None
            if arm in ("warm", "scratch", "shuffled", "random"):
                sparse = batch_metrics(
                    model, eval_batches, cfg,
                    lambda batch: gates_for(batch, evaluation=True))
        cum_eval += float(counter.get_total_flops())
        row = {
            "step": step,
            "heldout_loss": dense["loss"],
            "heldout_acc": dense["acc"],
            "sparse_heldout_loss": None if sparse is None else sparse["loss"],
            "sparse_heldout_acc": None if sparse is None else sparse["acc"],
            "cum_train_flops": cum_train,
            "cum_policy_flops": cum_policy,
            "cum_reward_eval_flops": 0.0,
            "cum_eval_flops": cum_eval,
            "cum_total_flops": cum_train + cum_policy,
            "wallclock_s": time.time() - start,
            "mean_active_fraction": (float(np.mean(active_history))
                                     if active_history else None),
        }
        history.append(row)
        print(f"[{arm}/s{args.seed}] step {step} | dense {dense['loss']:.6f} | "
              f"sparse {row['sparse_heldout_loss']} | total "
              f"{row['cum_total_flops'] / 1e12:.3f}T | policy "
              f"{cum_policy / 1e12:.3f}T | active {row['mean_active_fraction']} | "
              f"{row['wallclock_s']:.0f}s", flush=True)
        save(False)

    evaluate(0)
    for index, batch in enumerate(batches):
        step = index + 1
        model.train()
        logprob_parts = []
        active = []
        completes_window = (uses_policy and pending_action is not None
                            and len(pending_rewards) == args.reward_window - 1)
        gates = gates_for(
            batch, logprob_sink=None if completes_window else logprob_parts,
            active_sink=active)
        with FlopCounterMode(display=False) as train_counter:
            optimizer.zero_grad(set_to_none=True)
            loss, info = forward_train(model, batch, cfg, gates, gates is not None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        cum_train += float(train_counter.get_total_flops())
        current_loss = float(loss.detach())
        if gates is not None:
            active_history.append(sum(active) / (cfg.n_layers * int((batch[0] != TOKENIZER.pad_id).sum())))
        if uses_policy:
            previous_action = pending_action
            pending_action = (None if completes_window else
                              (torch.stack(logprob_parts).sum(),
                               int((batch[0] != TOKENIZER.pad_id).sum())))
            if previous_action is not None:
                if policy_state["baseline"] is None:
                    policy_state["baseline"] = current_loss
                advantage = policy_state["baseline"] - current_loss
                pending_rewards.append((previous_action[0], advantage,
                                        previous_action[1]))
                policy_state["baseline"] = (
                    (1 - args.baseline_beta) * policy_state["baseline"]
                    + args.baseline_beta * current_loss)
            if len(pending_rewards) == args.reward_window:
                with FlopCounterMode(display=False) as policy_counter:
                    update = policy_update(router, router_optimizer, pending_rewards)
                update["counter_flops"] = float(policy_counter.get_total_flops())
                cum_policy += max(
                    update["counter_flops"], update["analytical_backward_flops"])
                update["step"] = step
                update_history.append(update)
                pending_rewards = []
                print(f"  [{arm}] policy step {step} | loss {current_loss:.6f} | "
                      f"grad {update['router_grad_norm']:.3e}", flush=True)
        if step % args.eval_every == 0 or step == n_steps:
            evaluate(step)
    if uses_policy:
        router_out = os.path.join(args.outdir, f"{arm}_router_seed{args.seed}.pt")
        torch.save({
            "format": "ordinary_stream_policy_router_checkpoint_v1",
            "model_state": router.state_dict(),
            "router_config": {
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "max_len": cfg.max_len, "hidden": args.router_hidden,
            },
            "policy_state": policy_state,
            "source_checkpoint": os.path.abspath(args.ckpt),
            "source_online_seed": args.seed,
        }, router_out)
    save(True)
    print(f"saved {out}", flush=True)
    return {
        "arm": arm, "path": out, "final": history[-1], "history": history,
        "update_history": update_history, "active_history": active_history,
    }


def summarize(results, args):
    arms = {}
    for result in results:
        final = result["final"]
        arms[result["arm"]] = {
            "step": final["step"],
            "heldout_loss": final["heldout_loss"],
            "sparse_heldout_loss": final["sparse_heldout_loss"],
            "heldout_acc": final["heldout_acc"],
            "sparse_heldout_acc": final["sparse_heldout_acc"],
            "cum_train_flops": final["cum_train_flops"],
            "cum_policy_flops": final["cum_policy_flops"],
            "cum_reward_eval_flops": final["cum_reward_eval_flops"],
            "cum_total_flops": final["cum_total_flops"],
            "wallclock_s": final["wallclock_s"],
            "mean_active_fraction": final["mean_active_fraction"],
        }
    dense_fewer_target = arms["dense-fewer"]["cum_total_flops"]
    random_target = arms["random"]["cum_total_flops"]
    matched = {}
    for target_name, target in (("dense_fewer", dense_fewer_target),
                                ("random_cost", random_target)):
        matched[target_name] = {"total_flops": target, "loss": {}}
        for result in results:
            arm = result["arm"]
            metric = ("heldout_loss" if arm in ("dense", "dense-fewer")
                      else "sparse_heldout_loss")
            matched[target_name]["loss"][arm] = interpolate_cost(
                result["history"], target, metric)
    paired = {}
    for arm in ("warm", "scratch", "shuffled"):
        if arm not in arms:
            continue
        final = arms[arm]["sparse_heldout_loss"]
        paired[arm] = {}
        if "random" in arms:
            paired[arm]["minus_random"] = final - arms["random"]["sparse_heldout_loss"]
        if "shuffled" in arms:
            paired[arm]["minus_shuffled"] = final - arms["shuffled"]["sparse_heldout_loss"]
        if "dense-fewer" in arms:
            paired[arm]["minus_dense_fewer"] = final - arms["dense-fewer"]["heldout_loss"]
    return {
        "format": "ordinary_stream_policy_router_summary_v1",
        "split": args.split,
        "arms": arms,
        "matched_cost": matched,
        "paired_final_sparse_loss": paired,
        "note": f"{args.split} run; expected rather than exact per-step Bernoulli budget; one online seed",
    }


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
        model = build_model("predictor-supervised", cfg).double().eval()
        router = AllocationRouter(16, 2, 16, hidden=8).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        router_optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
        args = argparse.Namespace(temperature=1.0, budget=0.5)
        state = {"thresholds": torch.zeros(2), "baseline": None}
        generator = torch.Generator().manual_seed(8129)
        x = torch.randint(0, 19, (2, 6))
        x[0, -1] = TOKENIZER.pad_id
        y = torch.roll(x, -1, dims=1)
        y[:, -1] = TOKENIZER.pad_id
        pad = x != TOKENIZER.pad_id
        scores = torch.randn(2, 6, dtype=torch.float64)
        threshold = expected_budget_threshold(scores, pad, 0.5)
        expected_error = abs(float(torch.sigmoid(scores[pad] - threshold).mean()) - 0.5)
        logprob_parts = []
        active = []
        gates = policy_gates(
            router, pad, state, args, generator=generator,
            logprob_sink=logprob_parts, active_sink=active)
        loss, _ = forward_train(model, (x, y), cfg, gates, True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        body_step = int(optimizer.state[model.head.weight]["step"])
        action = torch.stack(logprob_parts).sum()
        x2 = torch.randint(0, 19, (2, 6))
        y2 = torch.roll(x2, -1, dims=1)
        y2[:, -1] = TOKENIZER.pad_id
        loss2, _ = forward_train(model, (x2, y2), cfg)
        loss2.backward()
        optimizer.step()
        body_step_after = int(optimizer.state[model.head.weight]["step"])
        body_before_policy = {
            name: value.detach().clone() for name, value in model.state_dict().items()}
        with FlopCounterMode(display=False) as counter:
            update = policy_update(
                router, router_optimizer,
                [(action, -float(loss2.detach()), int(pad.sum()))])
        grad_norm = update["router_grad_norm"]
        body_unchanged = all(
            torch.equal(value, model.state_dict()[name])
            for name, value in body_before_policy.items())
        body_step_after_policy = int(optimizer.state[model.head.weight]["step"])
        padding_error = 0.0
        for gate in gates:
            probe = torch.randn(2, 6, 16)
            mask = gate(probe)
            padding_error += float(mask[~pad].abs().sum())
        result = {
            "expected_budget_error": expected_error,
            "next_loss_policy_grad_norm": grad_norm,
            "body_step_before_policy_update": body_step,
            "body_step_after_policy_update": body_step_after_policy,
            "body_unchanged_during_policy_update": body_unchanged,
            "policy_update_flops": float(counter.get_total_flops()),
            "analytical_backward_flops": update["analytical_backward_flops"],
            "padding_error": padding_error,
        }
        result["passed"] = bool(
            expected_error < 1e-10 and np.isfinite(grad_norm) and grad_norm > 0
            and update["analytical_backward_flops"] > 0
            and body_step == 1 and body_step_after == 2
            and body_step_after_policy == 2 and body_unchanged
            and padding_error == 0)
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError("policy-router self-test failed")
        return result
    finally:
        torch.set_default_dtype(previous)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(handler=lambda args: self_test())
    run = sub.add_parser("run")
    run.add_argument("--ckpt", required=True)
    run.add_argument("--warm-router", required=True)
    run.add_argument("--outdir", required=True)
    run.add_argument("--device", default=None)
    run.add_argument("--size", default="small")
    run.add_argument("--seq-len", type=int, default=256)
    run.add_argument("--task", default="tagged")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--budget", type=float, default=0.5)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--reward-window", type=int, default=4)
    run.add_argument("--baseline-beta", type=float, default=0.2)
    run.add_argument("--body-lr", type=float, default=0.0003)
    run.add_argument("--router-lr", type=float, default=0.001)
    run.add_argument("--router-hidden", type=int, default=128)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--eval-every", type=int, default=50)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--seed", type=int, default=9308)
    run.add_argument("--arms", default="warm,scratch,shuffled,random,dense,dense-fewer")
    run.add_argument("--split", choices=["development", "confirmation"],
                     default="development")

    def run_all(args):
        results = [run_arm(args, arm) for arm in args.arms.split(",")]
        summary = summarize(results, args)
        out = os.path.join(args.outdir, "summary.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"saved {out}", flush=True)
    run.set_defaults(handler=run_all)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
