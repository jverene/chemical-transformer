"""Persistent-router distillation from true-model inverse allocation labels."""
import argparse
import glob
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.func import functional_call

from .config import Config, get_device
from .inverse_oracle import (
    body_gradients, budget_mask, dense_batch_loss, evaluate_candidate,
    first_order_scores, flat_gradient, harden_gates, load_model, mean_se,
    optimize_gates, scores_to_layer_mask, virtual_update)
from .models import build_model
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


@torch.no_grad()
def capture_features(model, state, batch, cfg):
    x, _ = batch
    sink = []
    logits, _ = functional_call(
        model, state, (x,),
        {"pad_mask": x != TOKENIZER.pad_id,
         "meta": {"diffs": None, "pre_ffn_sink": sink}})
    return torch.stack(sink), logits


class AllocationRouter(nn.Module):
    def __init__(self, d_model, n_layers, max_len, hidden=128,
                 embedding_dim=16):
        super().__init__()
        self.n_layers = n_layers
        self.max_len = max_len
        self.norm = nn.LayerNorm(d_model)
        self.feature = nn.Linear(d_model, hidden)
        self.layer = nn.Embedding(n_layers, embedding_dim)
        self.position = nn.Embedding(max_len, embedding_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden + 2 * embedding_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, features, layer_index=None):
        if features.ndim == 3:
            features = features.unsqueeze(0).unsqueeze(0)
        elif features.ndim == 4:
            features = features.unsqueeze(0)
        n, n_layers, batch, seq_len, _ = features.shape
        if seq_len > self.max_len or (layer_index is None and n_layers != self.n_layers):
            raise ValueError("router feature shape is incompatible with router")
        flat = features.reshape(-1, features.shape[-1])
        if layer_index is None:
            layer_ids = torch.arange(n_layers, device=features.device).view(
                1, n_layers, 1, 1).expand(n, n_layers, batch, seq_len).reshape(-1)
        else:
            layer_ids = torch.full(
                (n, n_layers, batch, seq_len), int(layer_index),
                dtype=torch.long, device=features.device).reshape(-1)
        position_ids = torch.arange(seq_len, device=features.device).view(
            1, 1, 1, seq_len).expand(n, n_layers, batch, seq_len).reshape(-1)
        hidden = self.feature(self.norm(flat))
        hidden = torch.cat(
            [hidden, self.layer(layer_ids), self.position(position_ids)], dim=-1)
        return self.output(hidden).squeeze(-1).reshape(
            n, n_layers, batch, seq_len)

    def forward_layer(self, x, layer_index):
        return self.forward(x, layer_index=layer_index)[0, 0]


def router_config(cfg, hidden, embedding_dim=16):
    return {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "max_len": cfg.max_len,
        "hidden": hidden,
        "embedding_dim": embedding_dim,
    }


def cpu_batch(batch):
    return tuple(value.detach().cpu() for value in batch)


def move_batch(batch, device):
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    return tuple(value.to(device) for value in batch)


def checkpoint_tag(path):
    match = re.search(r"seed(\d+)", os.path.basename(path))
    if match:
        return f"seed{match.group(1)}"
    return os.path.basename(path).replace(".pt", "")


def generate_teachers(args):
    os.makedirs(args.teacher_dir, exist_ok=True)
    for checkpoint_index, ckpt in enumerate(args.ckpts):
        local = argparse.Namespace(**vars(args))
        local.ckpt = ckpt
        model, cfg = load_model(local)
        state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        task = get_task(args.task)
        tag = checkpoint_tag(ckpt)
        print(f"Teacher generation | {tag} | {args.states} states", flush=True)
        for index in range(args.states):
            out = os.path.join(args.teacher_dir, tag, f"state_{index:03d}.pt")
            if os.path.exists(out):
                print(f"  exists {out}", flush=True)
                continue
            rng = random.Random(
                args.seed + checkpoint_index * 100003 + index * 1009)
            generator = torch.Generator().manual_seed(
                args.seed + checkpoint_index * 99991 + index * 7919)
            start = time.time()
            batch_a = get_batch(task, cfg, rng, args.batch_size)[:2]
            batch_c_opt = [get_batch(task, cfg, rng, args.batch_size)[:2]
                           for _ in range(args.c_opt_batches)]
            batch_c_hold = [get_batch(task, cfg, rng, args.batch_size)[:2]
                            for _ in range(args.c_hold_batches)]
            features, _ = capture_features(model, state, batch_a, cfg)
            pad = batch_a[0] != TOKENIZER.pad_id
            soft, history = optimize_gates(
                model, state, batch_a, batch_c_opt, pad, cfg, args.lr,
                args.gate_lr, args.gate_steps, args.budget, "uniform", generator)
            hard = harden_gates(soft, pad, args.budget)
            record = {
                "format": "inverse_router_teacher_v1",
                "checkpoint": os.path.abspath(ckpt),
                "checkpoint_tag": tag,
                "state_index": index,
                "features": features.detach().cpu().half(),
                "soft_gates": soft.detach().cpu().float(),
                "hard_mask": hard.detach().cpu().to(torch.uint8),
                "pad": pad.detach().cpu(),
                "batch_a": cpu_batch(batch_a),
                "c_opt": [cpu_batch(batch) for batch in batch_c_opt],
                "c_hold": [cpu_batch(batch) for batch in batch_c_hold],
                "q_history": history,
                "budget": args.budget,
                "teacher_config": {
                    "body_lr": args.lr,
                    "gate_lr": args.gate_lr,
                    "gate_steps": args.gate_steps,
                    "c_opt_batches": args.c_opt_batches,
                    "c_hold_batches": args.c_hold_batches,
                },
                "teacher_wallclock_s": time.time() - start,
            }
            os.makedirs(os.path.dirname(out), exist_ok=True)
            temporary = out + ".tmp"
            torch.save(record, temporary)
            os.replace(temporary, out)
            print(f"  {tag} state {index} | Q {history[0]['q']:.6f} -> "
                  f"{history[-1]['q']:.6f} | {record['teacher_wallclock_s']:.1f}s",
                  flush=True)
        del model


def normalize_tags(tags):
    return [f"seed{tag}" if tag.isdigit() else tag for tag in tags]


def teacher_files(teacher_dir, tags):
    groups = []
    for tag in normalize_tags(tags):
        files = sorted(glob.glob(os.path.join(teacher_dir, tag, "state_*.pt")))
        if not files:
            raise FileNotFoundError(f"no teacher states for {tag}")
        groups.append(files)
    return groups


def balanced_teacher_files(teacher_dir, tags, size):
    groups = teacher_files(teacher_dir, tags)
    if size % len(groups):
        raise ValueError("teacher-set size must be divisible by checkpoint count")
    per_group = size // len(groups)
    if any(len(files) < per_group for files in groups):
        raise ValueError("requested teacher-set size exceeds generated states")
    return [path for files in groups for path in files[:per_group]]


def load_teacher(path):
    record = torch.load(path, map_location="cpu")
    if record.get("format") != "inverse_router_teacher_v1":
        raise ValueError(f"unrecognized teacher record: {path}")
    return record


def teacher_targets(record, loss_name):
    targets = record["soft_gates"].float()
    if loss_name == "bce":
        return targets
    normalized = torch.zeros_like(targets)
    for b in range(targets.shape[1]):
        valid = record["pad"][b].unsqueeze(0).expand(targets.shape[0], -1)
        values = targets[:, b, :][valid]
        normalized[:, b, :][valid] = (
            values - values.mean()) / values.std().clamp_min(1e-6)
    return normalized


def train_one_router(args, files, size):
    records = [load_teacher(path) for path in files]
    targets = [teacher_targets(record, args.loss) for record in records]
    set_seed(args.router_seed)
    first = records[0]
    config = {
        "d_model": first["features"].shape[-1],
        "n_layers": first["features"].shape[0],
        "max_len": 512,
        "hidden": args.hidden,
        "embedding_dim": args.embedding_dim,
    }
    router = AllocationRouter(**config).to(args.device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.router_lr)
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(records)))
        random.Random(args.router_seed + epoch * 1009).shuffle(order)
        losses = []
        for offset in range(0, len(order), args.router_batch_size):
            indices = order[offset:offset + args.router_batch_size]
            features = torch.stack([records[i]["features"] for i in indices]).to(
                args.device).float()
            target = torch.stack([targets[i] for i in indices]).to(args.device)
            pads = torch.stack([records[i]["pad"] for i in indices]).to(args.device)
            logits = router(features)
            valid = pads.unsqueeze(1).expand_as(target)
            if args.loss == "bce":
                loss = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
            else:
                loss = F.mse_loss(logits[valid], target[valid])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "teacher_loss": mean_loss,
                        "loss_name": args.loss})
        print(f"  N={size} epoch={epoch} | teacher {args.loss} "
              f"{mean_loss:.6f}", flush=True)
    result = {
        "format": "inverse_allocation_router_v1",
        "model_state": router.state_dict(),
        "router_config": config,
        "teacher_files": [os.path.abspath(path) for path in files],
        "teacher_states": size,
        "teacher_wallclock_s": float(sum(r["teacher_wallclock_s"] for r in records)),
        "router_config_training": {
            "loss": args.loss,
            "epochs": args.epochs,
            "router_lr": args.router_lr,
            "router_batch_size": args.router_batch_size,
            "router_seed": args.router_seed,
        },
        "history": history,
        "router_train_wallclock_s": time.time() - start,
    }
    os.makedirs(args.router_dir, exist_ok=True)
    out = os.path.join(args.router_dir, f"router_n{size:03d}.pt")
    torch.save(result, out)
    print(f"  saved {out}", flush=True)
    return {
        "states": size,
        "path": out,
        "history": history,
        "teacher_wallclock_s": result["teacher_wallclock_s"],
        "router_train_wallclock_s": result["router_train_wallclock_s"],
    }


def train_routers(args):
    tags = args.train_tags.split(",")
    summaries = []
    for size in [int(value) for value in args.sizes.split(",")]:
        files = balanced_teacher_files(args.teacher_dir, tags, size)
        summaries.append(train_one_router(args, files, size))
    result = {
        "train_tags": normalize_tags(tags),
        "config": {k: v for k, v in vars(args).items() if k != "handler"},
        "learning_curve": summaries,
    }
    os.makedirs(args.router_dir, exist_ok=True)
    with open(os.path.join(args.router_dir, "training_summary.json"), "w") as f:
        json.dump(result, f, indent=2)


def load_router(path, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != "inverse_allocation_router_v1":
        raise ValueError(f"unrecognized router checkpoint: {path}")
    router = AllocationRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["model_state"])
    router.eval()
    return router, checkpoint


def evaluate_summary(states):
    def values(kind):
        return [next(r["c_hold_improvement"] for r in state["records"]
                     if r["kind"] == kind) for state in states]
    result = {}
    for kind in ["router", "inverse_soft", "inverse_hard", "gategrad_shared",
                 "gategrad_layer", "dense"]:
        result[f"{kind}_c_hold"] = mean_se(values(kind))
        result[f"{kind}_c_hold_per_state"] = values(kind)
    random_means = []
    best_random = []
    for state in states:
        random_records = [r for r in state["records"] if r["kind"] == "random"]
        random_means.append(float(np.mean(
            [r["c_hold_improvement"] for r in random_records])))
        best_random.append(next(
            r["c_hold_improvement"] for r in random_records
            if r.get("best_of_k_selected")))
    result["random_mean_c_hold"] = mean_se(random_means)
    result["best_of_k_random_c_hold"] = mean_se(best_random)
    router_values = values("router")
    for name, controls in [
        ("router_minus_random", random_means),
        ("router_minus_layer", values("gategrad_layer")),
        ("router_minus_shared", values("gategrad_shared")),
        ("router_minus_best_of_k", best_random),
        ("router_minus_teacher", values("inverse_hard")),
    ]:
        result[name] = mean_se([a - b for a, b in zip(router_values, controls)])
        result[f"{name}_per_state"] = [a - b for a, b in zip(router_values, controls)]
    result["router_teacher_mask_iou"] = mean_se([state["router_teacher_mask_iou"]
                                                  for state in states])
    if "router_teacher_spearman" in states[0]:
        result["router_teacher_spearman"] = mean_se([
            state["router_teacher_spearman"] for state in states])
    if "router_orientation" in states[0]:
        orientations = [state["router_orientation"] for state in states]
        result["router_orientation"] = {
            "top": int(sum(value == "top" for value in orientations)),
            "bottom": int(sum(value == "bottom" for value in orientations)),
        }
    return result


def evaluate_routers(args):
    files = teacher_files(args.teacher_dir, [args.eval_tag])[0][:args.states]
    if len(files) < args.states:
        raise ValueError("not enough evaluation teacher states")
    records = [load_teacher(path) for path in files]
    local = argparse.Namespace(**vars(args))
    local.ckpt = records[0]["checkpoint"]
    model, cfg = load_model(local)
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    routers = [(path, *load_router(path, args.device)) for path in args.routers]
    results = {path: [] for path in args.routers}
    for index, teacher in enumerate(records):
        batch_a = move_batch(teacher["batch_a"], args.device)
        batch_c_opt = move_batch(teacher["c_opt"], args.device)
        batch_c_hold = move_batch(teacher["c_hold"], args.device)
        pad = batch_a[0] != TOKENIZER.pad_id
        before_opt = dense_batch_loss(model, state, batch_c_opt, cfg)
        before_hold = dense_batch_loss(model, state, batch_c_hold, cfg)
        _, dense_grads = body_gradients(model, state, batch_a, None, cfg)
        dense_grad = flat_gradient(dense_grads)
        layer_scores = first_order_scores(model, state, batch_a, cfg)
        shared_scores = layer_scores.sum(dim=0)
        shared = budget_mask(shared_scores, pad, args.budget).unsqueeze(0).expand(
            cfg.n_layers, -1, -1).contiguous()
        layer = scores_to_layer_mask(layer_scores, pad, args.budget)
        teacher_soft = teacher["soft_gates"].to(args.device).float()
        teacher_hard = teacher["hard_mask"].to(args.device).float()
        candidates = [
            ("gategrad_shared", -1, shared),
            ("gategrad_layer", -1, layer),
            ("dense", -1, None),
            ("inverse_soft", -1, teacher_soft),
            ("inverse_hard", -1, teacher_hard),
        ]
        generator = torch.Generator().manual_seed(args.seed + index * 7919)
        for candidate in range(args.candidates):
            candidates.append(("random", candidate, scores_to_layer_mask(
                layer_scores, pad, args.budget, generator=generator,
                random_mask=True)))
        base_records = []
        for kind, candidate, mask in candidates:
            record = evaluate_candidate(
                model, state, mask, batch_a, batch_c_opt, batch_c_hold,
                dense_grad, cfg, args.lr, before_opt, before_hold)
            record.update({"kind": kind, "index": candidate})
            base_records.append(record)
        random_records = [r for r in base_records if r["kind"] == "random"]
        max(random_records, key=lambda r: r["cosine"])["core_selected"] = True
        max(random_records, key=lambda r: r["c_opt_improvement"])[
            "best_of_k_selected"] = True
        features = teacher["features"].to(args.device).float()
        for path, router, router_checkpoint in routers:
            with torch.no_grad():
                scores = router(features)[0]
            polarity = {}
            if args.router_direction == "copt":
                for orientation, signed_scores in [("top", scores), ("bottom", -scores)]:
                    candidate_mask = scores_to_layer_mask(
                        signed_scores, pad, args.budget)
                    updated, _ = virtual_update(
                        model, state, batch_a, [candidate_mask[i] for i in range(cfg.n_layers)],
                        cfg, args.lr, create_graph=False)
                    after_opt = dense_batch_loss(model, updated, batch_c_opt, cfg)
                    polarity[orientation] = {
                        "mask": candidate_mask,
                        "c_opt_improvement": before_opt - after_opt,
                    }
                orientation = max(
                    ["top", "bottom"],
                    key=lambda name: polarity[name]["c_opt_improvement"])
                mask = polarity[orientation]["mask"]
            else:
                orientation = args.router_direction
                mask = scores_to_layer_mask(
                    scores if orientation == "top" else -scores, pad, args.budget)
            router_record = evaluate_candidate(
                model, state, mask, batch_a, batch_c_opt, batch_c_hold,
                dense_grad, cfg, args.lr, before_opt, before_hold)
            router_record.update({"kind": "router", "index": -1})
            intersection = (mask * teacher_hard).sum()
            union = ((mask + teacher_hard) > 0).float().sum().clamp_min(1)
            valid = pad.unsqueeze(0).expand_as(scores)
            rank_correlation = float(spearmanr(
                scores[valid].detach().cpu().numpy(),
                teacher_soft[valid].detach().cpu().numpy())[0])
            state_record = {
                "state_index": teacher["state_index"],
                "teacher_file": files[index],
                "records": [*base_records, router_record],
                "router_teacher_mask_iou": float(intersection / union),
                "router_teacher_spearman": rank_correlation,
                "router_orientation": orientation,
            }
            if polarity:
                state_record["polarity_c_opt"] = {
                    name: values["c_opt_improvement"]
                    for name, values in polarity.items()
                }
            results[path].append(state_record)
            print(f"  state {index} N={router_checkpoint['teacher_states']} | "
                  f"orientation {orientation} | "
                  f"router {router_record['c_hold_improvement']:+.6f} | "
                  f"teacher {next(r['c_hold_improvement'] for r in base_records if r['kind'] == 'inverse_hard'):+.6f}",
                  flush=True)
    output = {
        "format": "inverse_router_evaluation_v1",
        "split": args.split,
        "evaluation_checkpoint": records[0]["checkpoint"],
        "evaluation_tag": normalize_tags([args.eval_tag])[0],
        "feature_trajectory": "dense_pre_ffn_planning_features",
        "config": {k: v for k, v in vars(args).items() if k != "handler"},
        "routers": {},
    }
    for path, _, router_checkpoint in routers:
        output["routers"][path] = {
            "teacher_states": router_checkpoint["teacher_states"],
            "teacher_wallclock_s": router_checkpoint["teacher_wallclock_s"],
            "router_train_wallclock_s": router_checkpoint["router_train_wallclock_s"],
            "summary": evaluate_summary(results[path]),
            "states": results[path],
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps({path: value["summary"] for path, value in
                      output["routers"].items()}, indent=2), flush=True)
    print(f"saved {args.out}", flush=True)


def self_test():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(991)
    cfg = Config.for_size(
        "small", device="cpu", d_model=16, n_layers=2, n_heads=4, d_ff=32,
        max_len=16, seq_len=6, dropout=0.0)
    model = build_model("predictor-supervised", cfg).double().eval()
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    x = torch.randint(0, 19, (2, cfg.seq_len))
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = TOKENIZER.pad_id
    with torch.no_grad():
        direct, _ = model(x, pad_mask=x != TOKENIZER.pad_id,
                          meta={"diffs": None})
    features, captured = capture_features(model, state, (x, y), cfg)
    logits_error = float((direct - captured).abs().max())
    router = AllocationRouter(16, 2, 16, hidden=8).double()
    scores = router(features)
    target = torch.rand_like(scores)
    valid = (x != TOKENIZER.pad_id).unsqueeze(1).expand_as(target)
    loss = F.binary_cross_entropy_with_logits(scores[valid], target[valid])
    loss.backward()
    gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all()
                           for p in router.parameters())
    mask = scores_to_layer_mask(scores[0].detach(), x != TOKENIZER.pad_id, 0.5)
    expected = 2 * sum(min(int(p.sum()), max(1, int(round(0.5 * int(p.sum())))))
                       for p in (x != TOKENIZER.pad_id))
    result = {
        "feature_shape": list(features.shape),
        "capture_logits_max_error": logits_error,
        "router_output_shape": list(scores.shape),
        "router_backward_finite": bool(gradients_finite),
        "hard_budget_error": abs(int(mask.sum()) - expected),
    }
    result["passed"] = bool(
        features.shape == (2, 2, cfg.seq_len, 16) and logits_error == 0
        and scores.shape == (1, 2, 2, cfg.seq_len) and gradients_finite
        and result["hard_budget_error"] == 0)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("router-pilot self-test failed")


def add_body_arguments(parser):
    parser.add_argument("--device", default=None)
    parser.add_argument("--size", default="small")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--allow-legacy-extra-keys", action="store_true")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(handler=lambda args: self_test())

    generate = sub.add_parser("generate")
    add_body_arguments(generate)
    generate.add_argument("--ckpts", nargs="+", required=True)
    generate.add_argument("--teacher-dir", required=True)
    generate.add_argument("--task", default="tagged")
    generate.add_argument("--states", type=int, default=16)
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--budget", type=float, default=0.5)
    generate.add_argument("--c-opt-batches", type=int, default=4)
    generate.add_argument("--c-hold-batches", type=int, default=4)
    generate.add_argument("--lr", type=float, default=0.001)
    generate.add_argument("--gate-lr", type=float, default=10000)
    generate.add_argument("--gate-steps", type=int, default=10)
    generate.add_argument("--seed", type=int, default=6000)
    generate.set_defaults(handler=generate_teachers)

    train = sub.add_parser("train")
    train.add_argument("--teacher-dir", required=True)
    train.add_argument("--router-dir", required=True)
    train.add_argument("--train-tags", default="0,1,2")
    train.add_argument("--sizes", default="6,12,24,48")
    train.add_argument("--device", default=get_device())
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument("--embedding-dim", type=int, default=16)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--router-lr", type=float, default=0.001)
    train.add_argument("--router-batch-size", type=int, default=2)
    train.add_argument("--loss", choices=["bce", "mse-z"], default="bce")
    train.add_argument("--router-seed", type=int, default=7000)
    train.set_defaults(handler=train_routers)

    evaluate = sub.add_parser("evaluate")
    add_body_arguments(evaluate)
    evaluate.add_argument("--teacher-dir", required=True)
    evaluate.add_argument("--eval-tag", required=True)
    evaluate.add_argument("--states", type=int, required=True)
    evaluate.add_argument("--routers", nargs="+", required=True)
    evaluate.add_argument("--split", choices=["smoke", "development", "confirmation"],
                          required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--budget", type=float, default=0.5)
    evaluate.add_argument("--router-direction", choices=["top", "bottom", "copt"],
                          default="top")
    evaluate.add_argument("--candidates", type=int, default=8)
    evaluate.add_argument("--lr", type=float, default=0.001)
    evaluate.add_argument("--seed", type=int, default=8000)
    evaluate.set_defaults(handler=evaluate_routers)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
