"""On-policy hard-router evaluation with executable gather/scatter FFNs."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .config import Config
from .inverse_oracle import (
    body_gradients, budget_mask, dense_batch_loss, evaluate_candidate,
    first_order_scores, flat_gradient, load_model, mean_se,
    scores_to_layer_mask, virtual_update)
from .models import build_model
from .router_pilot import (
    AllocationRouter, evaluate_summary, load_router, load_teacher, move_batch,
    normalize_tags, teacher_files)
from .tokenizer import TOKENIZER


def on_policy_gates(router, orientation, pad, budget, score_sink=None,
                    mask_sink=None):
    def make_gate(layer_index):
        def gate(x):
            with torch.no_grad():
                scores = router.forward_layer(x, layer_index)
            signed = scores if orientation == "top" else -scores
            mask = scores_to_layer_mask(
                signed.unsqueeze(0), pad, budget)[0]
            if score_sink is not None:
                score_sink.append(scores)
            if mask_sink is not None:
                mask_sink.append(mask)
            return mask
        return gate
    return [make_gate(layer) for layer in range(router.n_layers)]


def on_policy_reward(model, state, batch_a, batch_c_opt, gates, cfg, lr,
                     before_opt):
    updated, _ = virtual_update(
        model, state, batch_a, gates, cfg, lr, create_graph=False)
    after_opt = dense_batch_loss(model, updated, batch_c_opt, cfg)
    return before_opt - after_opt


def evaluate_on_policy(model, state, batch_a, batch_c_opt, batch_c_hold,
                       gates, scores, masks, cfg, lr, before_opt, before_hold):
    updated, train_loss = virtual_update(
        model, state, batch_a, gates, cfg, lr, create_graph=False)
    after_opt = dense_batch_loss(model, updated, batch_c_opt, cfg)
    after_hold = dense_batch_loss(model, updated, batch_c_hold, cfg)
    mask = torch.stack(masks)
    pad = batch_a[0] != TOKENIZER.pad_id
    return {
        "train_loss": float(train_loss.detach()),
        "c_opt_loss_after": after_opt,
        "c_opt_improvement": before_opt - after_opt,
        "c_hold_loss_after": after_hold,
        "c_hold_improvement": before_hold - after_hold,
        "active_fraction": float(mask.sum() / (mask.shape[0] * pad.sum())),
    }, torch.stack(scores), mask


def run(args):
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
        for path, router, router_checkpoint in routers:
            polarity = {}
            if args.router_direction == "copt":
                for orientation in ["top", "bottom"]:
                    gates = on_policy_gates(
                        router, orientation, pad, args.budget)
                    polarity[orientation] = on_policy_reward(
                        model, state, batch_a, batch_c_opt, gates, cfg,
                        args.lr, before_opt)
                orientation = max(["top", "bottom"], key=lambda name: polarity[name])
            else:
                orientation = args.router_direction
            scores, masks = [], []
            gates = on_policy_gates(
                router, orientation, pad, args.budget,
                score_sink=scores, mask_sink=masks)
            router_record, scores, mask = evaluate_on_policy(
                model, state, batch_a, batch_c_opt, batch_c_hold, gates,
                scores, masks, cfg, args.lr, before_opt, before_hold)
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
                state_record["polarity_c_opt"] = polarity
            results[path].append(state_record)
            print(f"  state {index} N={router_checkpoint['teacher_states']} | "
                  f"orientation {orientation} | "
                  f"router {router_record['c_hold_improvement']:+.6f} | "
                  f"teacher {next(r['c_hold_improvement'] for r in base_records if r['kind'] == 'inverse_hard'):+.6f}",
                  flush=True)
    output = {
        "format": "on_policy_router_evaluation_v1",
        "split": args.split,
        "evaluation_checkpoint": records[0]["checkpoint"],
        "evaluation_tag": normalize_tags([args.eval_tag])[0],
        "feature_trajectory": "on_policy_hard_gather",
        "execution": "active_row_gather_scatter",
        "config": {k: v for k, v in vars(args).items() if k != "handler"},
        "routers": {},
    }
    for path, _, router_checkpoint in routers:
        output["routers"][path] = {
            "teacher_states": router_checkpoint["teacher_states"],
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
    torch.manual_seed(1201)
    cfg = Config.for_size(
        "small", device="cpu", d_model=16, n_layers=2, n_heads=4, d_ff=32,
        max_len=16, seq_len=5, dropout=0.0)
    model = build_model("predictor-supervised", cfg).double().eval()
    x = torch.randint(0, 19, (2, cfg.seq_len))
    pad = x != TOKENIZER.pad_id
    masks = [scores_to_layer_mask(
        torch.randn(1, 2, cfg.seq_len), pad, 0.5)[0]
        for _ in range(cfg.n_layers)]
    kwargs = {"pad_mask": pad, "meta": {"diffs": None}}
    logits_mask, _ = model(x, oracle_gates=masks, **kwargs)
    logits_gather, _ = model(x, oracle_gates=[lambda _: masks[0], lambda _: masks[1]],
                             **kwargs)
    output_error = float((logits_mask - logits_gather).abs().max().detach())
    loss_mask = F.cross_entropy(logits_mask.view(-1, cfg.vocab_size), x.view(-1),
                                ignore_index=TOKENIZER.pad_id)
    loss_mask.backward()
    grads_mask = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    loss_gather = F.cross_entropy(logits_gather.view(-1, cfg.vocab_size), x.view(-1),
                                  ignore_index=TOKENIZER.pad_id)
    loss_gather.backward()
    grads_gather = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    gradient_error = max(float((a - b).abs().max())
                         for a, b in zip(grads_mask, grads_gather))
    router = AllocationRouter(16, 2, 16, hidden=8).double()
    features = torch.randn(2, 2, cfg.seq_len, 16)
    full = router(features)[0]
    layer_errors = [float((full[l] - router.forward_layer(features[l], l)).abs().max().detach())
                    for l in range(2)]
    result = {
        "gather_output_max_error": output_error,
        "gather_gradient_max_error": gradient_error,
        "single_layer_router_max_errors": layer_errors,
        "passed": output_error < 1e-12 and gradient_error < 1e-10
                  and max(layer_errors) == 0,
    }
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("on-policy self-test failed")


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
                          default="copt")
    evaluate.add_argument("--candidates", type=int, default=8)
    evaluate.add_argument("--lr", type=float, default=0.001)
    evaluate.add_argument("--seed", type=int, default=8000)
    evaluate.set_defaults(handler=run)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
