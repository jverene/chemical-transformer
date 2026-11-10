"""Short online polarity-persistence pilot with real AdamW updates."""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .inverse_oracle import first_order_scores, load_model, scores_to_layer_mask
from .on_policy import on_policy_gates
from .router_pilot import load_router
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import eval_loop, set_seed


def forward_train(model, batch, cfg, gates=None, gather=False):
    x, y = batch
    meta = {"diffs": None}
    if gather:
        meta["gather_ffn"] = True
    logits, info = model(x, pad_mask=x != TOKENIZER.pad_id, meta=meta,
                         oracle_gates=gates)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                           ignore_index=TOKENIZER.pad_id)
    return loss, info


@torch.no_grad()
def dense_eval(model, batches, cfg):
    values = []
    for batch in batches:
        loss, info = forward_train(model, batch, cfg)
        values.append((float(loss), float(info["flops"])))
    return values


def router_flops_per_token(router, d_model):
    hidden = router.feature.out_features
    embedding = router.layer.embedding_dim
    return 2 * (d_model * hidden + (hidden + 2 * embedding) * hidden + hidden)


def calibration_reward(model, optimizer, router, orientation, batch_a,
                       batch_c_opt, cfg, pad, budget, body_lr, before_opt):
    cal_model = copy.deepcopy(model).to(cfg.device).eval()
    cal_optimizer = torch.optim.AdamW(cal_model.parameters(), lr=body_lr)
    cal_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    gates = on_policy_gates(router, orientation, pad, budget)
    loss, info = forward_train(cal_model, batch_a, cfg, gates, gather=True)
    cal_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(cal_model.parameters(), 1.0)
    cal_optimizer.step()
    after_values = dense_eval(cal_model, batch_c_opt, cfg)
    after_opt = float(np.mean([value[0] for value in after_values]))
    eval_flops = sum(value[1] for value in after_values)
    return {
        "reward": before_opt - after_opt,
        "train_flops": 3 * float(info["flops"]),
        "eval_flops": eval_flops,
        "train_loss": float(loss.detach()),
    }


def make_static_gates(mask, cfg):
    return [mask[i] for i in range(cfg.n_layers)]


def run_arm(args, arm):
    set_seed(args.seed)
    local = argparse.Namespace(**vars(args))
    local.allow_legacy_extra_keys = True
    model, cfg = load_model(local)
    cfg.batch_size = args.batch_size
    cfg.dropout = 0.1
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.1
    model.to(cfg.device)
    router, router_checkpoint = load_router(args.router, cfg.device)
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    task = get_task(args.task)
    held_out = task.build_held_out(cfg)
    rng = random.Random(args.seed * 100003 + 7)
    calibration_rng = random.Random(args.seed * 100003 + 1701)
    mask_generator = torch.Generator().manual_seed(args.seed * 100003 + 811)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.body_lr)
    n_steps = args.steps // 2 if arm == "dense-fewer" else args.steps
    cum_train = 0.0
    cum_calibration = 0.0
    cum_router = 0.0
    orientation = None
    history = []
    calibration_history = []
    router_unit = router_flops_per_token(router, cfg.d_model)
    start = time.time()

    def evaluate(step):
        metrics = eval_loop(model, held_out, task, cfg, args.eval_batches)
        row = {
            "step": step,
            "heldout_loss": metrics["loss"],
            "heldout_acc": metrics["acc"],
            "cum_train_flops": cum_train,
            "cum_calibration_flops": cum_calibration,
            "cum_router_flops": cum_router,
            "cum_total_flops": cum_train + cum_calibration + cum_router,
            "wallclock_s": time.time() - start,
        }
        history.append(row)
        print(f"[{arm}/s{args.seed}] step {step} | loss {metrics['loss']:.6f} | "
              f"acc {metrics['acc']:.4f} | total {row['cum_total_flops'] / 1e12:.3f}T | "
              f"orientation {orientation} | {row['wallclock_s']:.0f}s", flush=True)

    evaluate(0)
    for step in range(1, n_steps + 1):
        batch_a = get_batch(task, cfg, rng, cfg.batch_size)[:2]
        pad = batch_a[0] != TOKENIZER.pad_id
        valid_count = int(pad.sum())
        gates = None
        gather = False
        model.train()
        if arm in ("top", "bottom", "calibrated"):
            if arm in ("top", "bottom"):
                orientation = arm
            if arm == "calibrated" and (step == 1 or step % args.calibrate_every == 0):
                model.eval()
                batch_c_opt = [get_batch(task, cfg, calibration_rng, args.calibration_batch_size)[:2]
                               for _ in range(args.c_opt_batches)]
                before_values = dense_eval(model, batch_c_opt, cfg)
                before_opt = float(np.mean([value[0] for value in before_values]))
                before_eval_flops = sum(value[1] for value in before_values)
                rewards = {}
                calibration_flops = before_eval_flops
                for candidate in ["top", "bottom"]:
                    result = calibration_reward(
                        model, optimizer, router, candidate, batch_a, batch_c_opt,
                        cfg, pad, args.budget, args.body_lr, before_opt)
                    rewards[candidate] = result["reward"]
                    calibration_flops += result["train_flops"] + result["eval_flops"]
                orientation = max(["top", "bottom"], key=lambda name: rewards[name])
                calibration_history.append({
                    "step": step,
                    "rewards": rewards,
                    "selected": orientation,
                    "calibration_flops": calibration_flops,
                    "calibration_a_tokens": 2 * valid_count,
                    "calibration_c_opt_tokens": (
                        2 * args.c_opt_batches * args.calibration_batch_size
                        * cfg.seq_len),
                })
                cum_calibration += calibration_flops
                cum_router += 2 * router_unit * cfg.n_layers * valid_count
                model.train()
            gates = on_policy_gates(router, orientation, pad, args.budget)
            gather = True
        elif arm == "random":
            mask = scores_to_layer_mask(
                torch.empty(cfg.n_layers, *batch_a[0].shape, device=cfg.device),
                pad, args.budget, generator=mask_generator, random_mask=True)
            gates = make_static_gates(mask, cfg)
            gather = True
        elif arm == "layer":
            if step == 1 or step % args.calibrate_every == 0:
                model.eval()
                layer_scores = first_order_scores(
                    model, {**dict(model.named_parameters()), **dict(model.named_buffers())},
                    batch_a, cfg)
                current_mask = scores_to_layer_mask(layer_scores, pad, args.budget)
                cum_calibration += 3 * float(forward_train(model, batch_a, cfg)[1]["flops"])
                model.train()
            gates = make_static_gates(current_mask, cfg)
            gather = True
        elif arm == "top" or arm == "bottom":
            gates = on_policy_gates(router, arm, pad, args.budget)
            gather = True

        optimizer.zero_grad()
        loss, info = forward_train(model, batch_a, cfg, gates, gather=gather)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cum_train += 3 * float(info["flops"])
        if arm in ("top", "bottom", "calibrated"):
            cum_router += router_unit * cfg.n_layers * valid_count
        if step % args.eval_every == 0 or step == n_steps:
            evaluate(step)
    result = {
        "format": "online_polarity_pilot_v1",
        "split": "development",
        "arm": arm,
        "seed": args.seed,
        "checkpoint": os.path.abspath(args.ckpt),
        "optimizer": "AdamW initialized fresh at shared checkpoint",
        "body_lr": args.body_lr,
        "dropout": cfg.dropout,
        "feature_trajectory": "on_policy_hard_gather" if arm in ("top", "bottom", "calibrated") else "static_hard_gather_or_dense",
        "execution": "active_row_gather_scatter_for_all_non_dense_arms",
        "router": os.path.abspath(args.router),
        "router_teacher_states": router_checkpoint["teacher_states"],
        "config": {k: v for k, v in vars(args).items() if k != "handler"},
        "history": history,
        "calibration_history": calibration_history,
        "final": history[-1],
    }
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{arm}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out}", flush=True)


def add_body_arguments(parser):
    parser.add_argument("--device", default=None)
    parser.add_argument("--size", default="small")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--allow-legacy-extra-keys", action="store_true")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    add_body_arguments(run)
    run.add_argument("--ckpt", required=True)
    run.add_argument("--router", required=True)
    run.add_argument("--outdir", required=True)
    run.add_argument("--task", default="tagged")
    run.add_argument("--arms", default="calibrated,bottom,top,random,layer,dense,dense-fewer")
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--budget", type=float, default=0.5)
    run.add_argument("--calibrate-every", type=int, default=10)
    run.add_argument("--c-opt-batches", type=int, default=4)
    run.add_argument("--calibration-batch-size", type=int, default=8)
    run.add_argument("--body-lr", type=float, default=0.0003)
    run.add_argument("--eval-every", type=int, default=50)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--seed", type=int, default=9004)
    run.set_defaults(handler=lambda args: [
        run_arm(args, arm) for arm in args.arms.split(",")])
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
