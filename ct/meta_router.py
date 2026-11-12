"""Online meta-routing through realized functional AdamW body updates."""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.flop_counter import FlopCounterMode

from .inverse_oracle import load_model, scores_to_layer_mask
from .router_pilot import AllocationRouter
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import set_seed


def loss_from_state(model, state, batch, cfg, gates=None, gather=False):
    x, y = batch
    meta = {"diffs": None}
    if gather:
        meta["gather_ffn"] = True
    logits, info = functional_call(
        model, state, (x,),
        {"pad_mask": x != TOKENIZER.pad_id, "meta": meta,
         "oracle_gates": gates})
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), y.view(-1),
        ignore_index=TOKENIZER.pad_id)
    return loss, info


def forward_train(model, batch, cfg, gates=None, gather=False):
    x, y = batch
    logits, info = model(
        x, pad_mask=x != TOKENIZER.pad_id,
        meta={"diffs": None, "gather_ffn": gather}, oracle_gates=gates)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), y.view(-1),
        ignore_index=TOKENIZER.pad_id)
    return loss, info


def functional_clip(grads, max_norm):
    present = [grad for grad in grads.values() if grad is not None]
    if not present:
        raise RuntimeError("body update produced no gradients")
    total = torch.norm(torch.stack([torch.norm(grad, 2) for grad in present]), 2)
    coefficient = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return {
        name: None if grad is None else grad * coefficient
        for name, grad in grads.items()
    }, total


def functional_adamw(params, grads, optimizer):
    defaults = optimizer.defaults
    if defaults.get("amsgrad") or defaults.get("maximize"):
        raise ValueError("functional AdamW supports neither AMSGrad nor maximize")
    beta1, beta2 = defaults["betas"]
    lr = defaults["lr"]
    eps = defaults["eps"]
    weight_decay = defaults["weight_decay"]
    updated = dict(params)
    updated_state = {}
    for name, param in params.items():
        grad = grads[name]
        previous = optimizer.state.get(param)
        if grad is None:
            if previous is not None:
                updated_state[name] = dict(previous)
            continue
        if previous is None:
            step = 1
            exp_avg = torch.zeros_like(param)
            exp_avg_sq = torch.zeros_like(param)
            step_device = "cpu"
        else:
            step = int(previous["step"]) + 1
            exp_avg = previous["exp_avg"]
            exp_avg_sq = previous["exp_avg_sq"]
            step_device = previous["step"].device
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()
        bias1 = 1 - beta1 ** step
        bias2 = 1 - beta2 ** step
        variance_floor = torch.finfo(exp_avg_sq.dtype).tiny
        denom = exp_avg_sq.clamp_min(variance_floor).sqrt() / np.sqrt(bias2) + eps
        decayed = param * (1 - lr * weight_decay)
        updated[name] = decayed - (lr / bias1) * exp_avg / denom
        updated_state[name] = {
            "step": torch.tensor(float(step), device=step_device),
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }
    return updated, updated_state


def functional_body_update(model, batch, cfg, gates, optimizer, max_norm=1.0):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    x, y = batch
    logits, info = model(
        x, pad_mask=x != TOKENIZER.pad_id,
        meta={"diffs": None, "gather_ffn": False}, oracle_gates=gates)
    train_loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size), y.view(-1),
        ignore_index=TOKENIZER.pad_id)
    values = torch.autograd.grad(
        train_loss, tuple(params.values()), create_graph=True, allow_unused=True)
    grads = dict(zip(params, values))
    grads, grad_norm = functional_clip(grads, max_norm)
    updated, optimizer_state = functional_adamw(params, grads, optimizer)
    return ({
        **updated, **buffers}, optimizer_state, train_loss, info, grad_norm)


def install_functional_update(model, optimizer, state, optimizer_state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(state[name].detach())
        for name, values in optimizer_state.items():
            param = dict(model.named_parameters())[name]
            target = optimizer.state[param]
            for key, value in values.items():
                value = value.detach()
                if key in target and torch.is_tensor(target[key]):
                    target[key].copy_(value)
                else:
                    target[key] = value.clone()
    optimizer.zero_grad(set_to_none=True)


def permute_valid_scores(scores, pad, generator):
    output = torch.empty_like(scores)
    for b in range(scores.shape[0]):
        valid = pad[b]
        count = int(valid.sum())
        permutation = torch.randperm(count, generator=generator).to(scores.device)
        output[b, valid] = scores[b, valid][permutation]
        output[b, ~valid] = scores[b, ~valid]
    return output


def router_gates(router, pad, budget, differentiable, shuffle=False,
                 generator=None, mask_sink=None):
    def make_gate(layer_index):
        def gate(x):
            if differentiable:
                scores = router.forward_layer(x, layer_index)
                if shuffle:
                    scores = permute_valid_scores(scores, pad, generator)
                hard = scores_to_layer_mask(
                    scores.detach().unsqueeze(0), pad, budget)[0]
                soft = torch.sigmoid(scores)
                result = hard + soft - soft.detach()
            else:
                scores = router.forward_layer(x.detach(), layer_index).detach()
                if shuffle:
                    scores = permute_valid_scores(scores, pad, generator)
                result = scores_to_layer_mask(
                    scores.unsqueeze(0), pad, budget)[0]
            if mask_sink is not None:
                mask_sink.append(result.detach())
            return result
        return gate
    return [make_gate(layer) for layer in range(router.n_layers)]


def random_gates(n_layers, shape, pad, budget, generator):
    return [scores_to_layer_mask(
        torch.empty(1, *shape, device=pad.device), pad, budget,
        generator=generator, random_mask=True)[0]
        for _ in range(n_layers)]


def router_flops_per_token(router, d_model):
    hidden = router.feature.out_features
    embedding = router.layer.embedding_dim
    return 2 * (d_model * hidden + (hidden + 2 * embedding) * hidden + hidden)


def meta_step(model, optimizer, router, router_optimizer, batch, reward_batches,
              cfg, args, shuffle, generator):
    masks = []
    gates = router_gates(
        router, batch[0] != TOKENIZER.pad_id, args.budget, True,
        shuffle=shuffle, generator=generator, mask_sink=masks)
    with FlopCounterMode(display=False) as train_counter:
        state, optimizer_state, train_loss, info, body_grad_norm = \
            functional_body_update(model, batch, cfg, gates, optimizer)
    train_flops = float(train_counter.get_total_flops())
    params = tuple(router.parameters())
    accumulated = [torch.zeros_like(param) for param in params]
    reward_losses = []
    reward_forward_flops = 0.0
    was_training = model.training
    model.eval()
    with FlopCounterMode(display=False) as meta_counter:
        for index, reward_batch in enumerate(reward_batches):
            reward_loss, reward_info = loss_from_state(
                model, state, reward_batch, cfg)
            reward_losses.append(float(reward_loss.detach()))
            reward_forward_flops += float(reward_info["flops"])
            gradients = torch.autograd.grad(
                reward_loss, params, retain_graph=index + 1 < len(reward_batches))
            for target, gradient in zip(accumulated, gradients):
                target.add_(gradient / len(reward_batches))
        router_optimizer.zero_grad(set_to_none=True)
        for param, gradient in zip(params, accumulated):
            param.grad = gradient
        router_grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        router_optimizer.step()
    meta_flops = float(meta_counter.get_total_flops())
    model.train(was_training)
    install_functional_update(model, optimizer, state, optimizer_state)
    mask = torch.stack(masks)
    pad = batch[0] != TOKENIZER.pad_id
    return {
        "train_loss": float(train_loss.detach()),
        "body_grad_norm": float(body_grad_norm.detach()),
        "router_grad_norm": float(router_grad_norm.detach()),
        "reward_losses": reward_losses,
        "reward_loss": float(np.mean(reward_losses)),
        "train_flops": train_flops,
        "meta_flops": meta_flops,
        "reward_forward_flops_analytical": reward_forward_flops,
        "active_fraction": float(mask.sum() / (mask.shape[0] * pad.sum())),
        "state_detached": all(
            not value.requires_grad and value.grad_fn is None
            for values in optimizer.state.values() for value in values.values()
            if torch.is_tensor(value)),
    }


def ordinary_step(model, optimizer, batch, cfg, gates):
    with FlopCounterMode(display=False) as counter:
        optimizer.zero_grad(set_to_none=True)
        loss, info = forward_train(model, batch, cfg, gates, gates is not None)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {
        "train_loss": float(loss.detach()),
        "body_grad_norm": float(grad_norm.detach()),
        "train_flops": float(counter.get_total_flops()),
        "forward_flops_analytical": float(info["flops"]),
    }


def evaluation_batches(task, cfg, held_out, count):
    rng = random.Random(999)
    return [get_batch(task, cfg, rng, cfg.batch_size, held_out)
            for _ in range(count)]


@torch.no_grad()
def batch_metrics(model, batches, cfg, gate_factory=None):
    loss_sum = 0.0
    loss_n = 0
    correct = 0
    answers = 0
    for batch in batches:
        x, y, _, answer_mask = batch
        body_batch = batch[:2]
        gates = gate_factory(body_batch) if gate_factory is not None else None
        meta = {"diffs": None, "gather_ffn": gates is not None}
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
    uses_router = arm in ("meta", "shuffled")
    router = None
    router_optimizer = None
    if uses_router:
        router = AllocationRouter(
            cfg.d_model, cfg.n_layers, cfg.max_len, hidden=args.router_hidden).to(
                cfg.device)
        router_optimizer = torch.optim.AdamW(
            router.parameters(), lr=args.router_lr, weight_decay=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.body_lr, weight_decay=args.weight_decay)
    mask_generator = torch.Generator().manual_seed(args.seed * 100003 + 811)
    shuffle_generator = torch.Generator().manual_seed(args.seed * 100003 + 812)
    eval_generator = torch.Generator().manual_seed(args.seed * 100003 + 813)
    set_seed(args.seed * 100003 + 17)
    cum_train = 0.0
    cum_meta = 0.0
    cum_eval = 0.0
    history = []
    meta_history = []
    start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{arm}_seed{args.seed}.json")

    def gates_for(batch, differentiable=False, evaluation=False):
        pad = batch[0] != TOKENIZER.pad_id
        if arm in ("meta", "shuffled"):
            return router_gates(
                router, pad, args.budget, differentiable,
                shuffle=arm == "shuffled",
                generator=eval_generator if evaluation else shuffle_generator)
        if arm == "random":
            return random_gates(
                cfg.n_layers, batch[0].shape, pad, args.budget,
                eval_generator if evaluation else mask_generator)
        return None

    def save(completed=False):
        result = {
            "format": "online_training_stream_meta_router_v1",
            "split": args.split,
            "completed": completed,
            "arm": arm,
            "seed": args.seed,
            "checkpoint": os.path.abspath(args.ckpt),
            "optimizer": "realized differentiable AdamW on meta-steps; ordinary AdamW otherwise",
            "data_stream": "one pregenerated ordinary stream; reward indices subsequently body-trained",
            "execution": "hard per-layer top-b; gather/scatter ordinarily, equivalent dense output masking on differentiable meta-steps",
            "accounting": "FlopCounterMode executed matmul/attention FLOPs; train/meta/eval separated; wallclock includes all work",
            "config": {k: v for k, v in vars(args).items() if k != "handler"},
            "history": history,
            "meta_history": meta_history,
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
            if arm in ("meta", "shuffled", "random"):
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
            "cum_meta_flops": cum_meta,
            "cum_eval_flops": cum_eval,
            "cum_total_flops": cum_train + cum_meta,
            "cum_all_flops": cum_train + cum_meta + cum_eval,
            "wallclock_s": time.time() - start,
        }
        history.append(row)
        print(f"[{arm}/s{args.seed}] step {step} | dense {dense['loss']:.6f} | "
              f"sparse {row['sparse_heldout_loss']} | total "
              f"{row['cum_total_flops'] / 1e12:.3f}T | meta "
              f"{cum_meta / 1e12:.3f}T | {row['wallclock_s']:.0f}s", flush=True)
        save(False)

    evaluate(0)
    for index, batch in enumerate(batches):
        step = index + 1
        model.train()
        is_meta = (uses_router and (step - 1) % args.meta_every == 0
                   and step + args.reward_batches <= n_steps)
        if is_meta:
            reward_indices = list(range(step + 1, step + args.reward_batches + 1))
            result = meta_step(
                model, optimizer, router, router_optimizer, batch,
                [batches[position - 1] for position in reward_indices], cfg, args,
                arm == "shuffled", shuffle_generator)
            result.update({"step": step, "reward_batch_indices": reward_indices})
            meta_history.append(result)
            cum_train += result["train_flops"]
            cum_meta += result["meta_flops"]
            print(f"  [{arm}] meta step {step} | reward {result['reward_loss']:.6f} | "
                  f"router grad {result['router_grad_norm']:.3e} | "
                  f"train {result['train_flops'] / 1e9:.1f}G | "
                  f"meta {result['meta_flops'] / 1e9:.1f}G", flush=True)
        else:
            gates = gates_for(batch)
            result = ordinary_step(model, optimizer, batch, cfg, gates)
            cum_train += result["train_flops"]
        if step % args.eval_every == 0 or step == n_steps:
            evaluate(step)
    if any(index > n_steps or index <= record["step"]
           for record in meta_history
           for index in record["reward_batch_indices"]):
        raise RuntimeError("a reward batch was not subsequently body-trained")
    if uses_router:
        router_out = os.path.join(args.outdir, f"{arm}_router_seed{args.seed}.pt")
        torch.save({
            "format": "online_meta_allocation_router_v1",
            "model_state": router.state_dict(),
            "router_config": {
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "max_len": cfg.max_len, "hidden": args.router_hidden,
            },
            "source_checkpoint": os.path.abspath(args.ckpt),
            "source_online_seed": args.seed,
            "source_train_flops": cum_train,
            "source_meta_flops": cum_meta,
        }, router_out)
    save(True)
    print(f"saved {out}", flush=True)
    return {"arm": arm, "path": out, "final": history[-1],
            "history": history, "meta_history": meta_history}


def interpolate_cost(history, target, metric):
    available = [row for row in history if row[metric] is not None]
    return float(np.interp(
        target, [row["cum_total_flops"] for row in available],
        [row[metric] for row in available]))


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
            "cum_meta_flops": final["cum_meta_flops"],
            "cum_eval_flops": final["cum_eval_flops"],
            "cum_total_flops": final["cum_total_flops"],
            "wallclock_s": final["wallclock_s"],
        }
    target = arms["dense-fewer"]["cum_total_flops"]
    matched = {}
    for result in results:
        arm = result["arm"]
        metric = ("heldout_loss" if arm in ("dense", "dense-fewer")
                  else "sparse_heldout_loss")
        matched[arm] = interpolate_cost(result["history"], target, metric)
    learned = next(result for result in results if result["arm"] == "meta")
    random_result = next(result for result in results if result["arm"] == "random")
    shuffled = next(result for result in results if result["arm"] == "shuffled")
    level2 = {
        "meta_minus_random_sparse_loss": (
            learned["final"]["sparse_heldout_loss"]
            - random_result["final"]["sparse_heldout_loss"]),
        "meta_minus_shuffled_sparse_loss": (
            learned["final"]["sparse_heldout_loss"]
            - shuffled["final"]["sparse_heldout_loss"]),
        "single_seed_passes_descriptive": bool(
            learned["final"]["sparse_heldout_loss"]
            < random_result["final"]["sparse_heldout_loss"]
            and learned["final"]["sparse_heldout_loss"]
            < shuffled["final"]["sparse_heldout_loss"]),
    }
    return {
        "format": "online_training_stream_meta_router_summary_v1",
        "split": args.split,
        "arms": arms,
        "dense_fewer_matched_total_cost": {
            "total_flops": target,
            "primary_sparse_deployment_loss_by_arm": matched,
            "meta_minus_dense_fewer": matched["meta"] - matched["dense-fewer"],
        },
        "level2": level2,
        "level3_single_seed_passes_descriptive": bool(
            matched["meta"] < matched["dense-fewer"]),
        "meta_updates": len(learned["meta_history"]),
        "note": "one seed; multi-seed uncertainty requires the combined analysis",
    }


def self_test():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        from .config import Config
        from .models import build_model
        torch.manual_seed(12012)
        cfg = Config.for_size(
            "small", device="cpu", d_model=16, n_layers=2, n_heads=4,
            d_ff=32, max_len=16, seq_len=6, dropout=0.0)
        base = build_model("predictor-supervised", cfg).double().eval()
        ordinary = copy.deepcopy(base)
        x = torch.randint(0, 19, (2, cfg.seq_len))
        x[0, -1] = TOKENIZER.pad_id
        y = torch.roll(x, -1, dims=1)
        y[:, -1] = TOKENIZER.pad_id
        reward_x = torch.randint(0, 19, (2, cfg.seq_len))
        reward_y = torch.roll(reward_x, -1, dims=1)
        reward_y[:, -1] = TOKENIZER.pad_id
        batch = (x, y)
        reward_batch = (reward_x, reward_y)
        pad = x != TOKENIZER.pad_id
        gates = [torch.full(x.shape, 0.7) for _ in range(cfg.n_layers)]
        reference_optimizer = torch.optim.AdamW(ordinary.parameters(), lr=1e-3)
        reference_optimizer.zero_grad()
        reference_loss, _ = forward_train(ordinary, batch, cfg, gates, False)
        reference_loss.backward()
        torch.nn.utils.clip_grad_norm_(ordinary.parameters(), 1.0)
        reference_optimizer.step()
        model = copy.deepcopy(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        state, opt_state, _, _, _ = functional_body_update(
            model, batch, cfg, gates, optimizer)
        parameter_error = max(
            float((state[name].detach()
                   - dict(ordinary.named_parameters())[name].detach()).abs().max())
            for name in dict(model.named_parameters()))
        expected_state_names = {
            name for name, param in ordinary.named_parameters()
            if param in reference_optimizer.state}
        state_keys_match = set(opt_state) == expected_state_names
        state_error = 0.0
        for name in expected_state_names:
            reference_state = reference_optimizer.state[dict(ordinary.named_parameters())[name]]
            for key in ("exp_avg", "exp_avg_sq"):
                state_error = max(state_error, float(
                    (opt_state[name][key].detach() - reference_state[key]).abs().max()))
        unchanged = all(torch.equal(model.state_dict()[name], value)
                        for name, value in before.items())
        optimizer_untouched = len(optimizer.state) == 0
        install_functional_update(model, optimizer, state, opt_state)
        detached = all(
            not value.requires_grad and value.grad_fn is None
            for values in optimizer.state.values() for value in values.values()
            if torch.is_tensor(value))
        installed_error = max(
            float((state[name].detach() - param.detach()).abs().max())
            for name, param in model.named_parameters())

        hyper_model = copy.deepcopy(base)
        hyper_optimizer = torch.optim.AdamW(hyper_model.parameters(), lr=1e-3)
        soft = [torch.full(x.shape, 0.5, requires_grad=True)
                for _ in range(cfg.n_layers)]
        updated, _, _, _, _ = functional_body_update(
            hyper_model, batch, cfg, soft, hyper_optimizer)
        reward, _ = loss_from_state(hyper_model, updated, reward_batch, cfg)
        analytic = torch.autograd.grad(reward, soft)[0][0, 0]
        epsilon = 1e-6
        numeric_values = []
        for sign in (1, -1):
            shifted = [gate.detach().clone() for gate in soft]
            shifted[0][0, 0] += sign * epsilon
            shifted_state, _, _, _, _ = functional_body_update(
                hyper_model, batch, cfg, shifted, hyper_optimizer)
            value, _ = loss_from_state(
                hyper_model, shifted_state, reward_batch, cfg)
            numeric_values.append(float(value.detach()))
        numeric = (numeric_values[0] - numeric_values[1]) / (2 * epsilon)
        relative_error = abs(float(analytic) - numeric) / max(abs(numeric), 1e-12)

        router = AllocationRouter(16, 2, 16, hidden=8).double()
        router_optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
        ste_model = copy.deepcopy(base)
        ste_optimizer = torch.optim.AdamW(ste_model.parameters(), lr=1e-3)
        ste_gates = router_gates(router, pad, 0.5, True)
        ste_state, _, _, _, _ = functional_body_update(
            ste_model, batch, cfg, ste_gates, ste_optimizer)
        ste_reward, _ = loss_from_state(ste_model, ste_state, reward_batch, cfg)
        router_grads = torch.autograd.grad(ste_reward, tuple(router.parameters()))
        router_grad_norm = float(torch.norm(torch.stack([
            torch.norm(grad) for grad in router_grads])))
        probe = torch.randn(2, cfg.seq_len, 16)
        hard = ste_gates[0](probe)
        expected = [min(int(row.sum()), max(1, int(round(0.5 * int(row.sum())))))
                    for row in pad]
        budget_error = max(abs(int(hard[b].sum()) - expected[b])
                           for b in range(pad.shape[0]))
        padding_error = float(hard.masked_select(~pad).abs().sum())
        with FlopCounterMode(display=False) as counter:
            forward_train(base, batch, cfg)
        counted_flops = float(counter.get_total_flops())
        result = {
            "functional_adamw_parameter_max_error": parameter_error,
            "functional_adamw_state_max_error": state_error,
            "functional_adamw_state_keys_match": state_keys_match,
            "original_body_unchanged": unchanged,
            "original_optimizer_unchanged": optimizer_untouched,
            "installed_parameter_max_error": installed_error,
            "installed_optimizer_state_detached": detached,
            "adamw_hypergradient": float(analytic),
            "central_finite_difference": numeric,
            "hypergradient_relative_error": relative_error,
            "ste_next_batch_router_grad_norm": router_grad_norm,
            "ste_hard_budget_error": budget_error,
            "ste_padding_error": padding_error,
            "executed_flop_counter_positive": counted_flops > 0,
        }
        result["passed"] = bool(
            parameter_error < 1e-9 and state_error < 1e-9 and state_keys_match
            and unchanged and optimizer_untouched and installed_error == 0 and detached
            and relative_error < 2e-4 and np.isfinite(router_grad_norm)
            and router_grad_norm > 0 and budget_error == 0 and padding_error == 0
            and counted_flops > 0)
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError("meta-router self-test failed")
        return result
    finally:
        torch.set_default_dtype(previous_dtype)


def checkpoint_smoke(args):
    set_seed(args.seed)
    args.allow_legacy_extra_keys = True
    model, cfg = load_model(args)
    cfg.batch_size = args.batch_size
    cfg.dropout = args.dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.dropout
    model.train()
    router = AllocationRouter(
        cfg.d_model, cfg.n_layers, cfg.max_len, hidden=args.router_hidden).to(
            cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.body_lr)
    router_optimizer = torch.optim.AdamW(router.parameters(), lr=args.router_lr)
    task = get_task(args.task)
    rng = random.Random(args.seed)
    batches = [get_batch(task, cfg, rng, args.batch_size)[:2]
               for _ in range(args.reward_batches + 1)]
    generator = torch.Generator().manual_seed(args.seed)
    pad = batches[0][0] != TOKENIZER.pad_id
    probe_gates = router_gates(router, pad, args.budget, False)
    model.eval()
    masked_loss, _ = forward_train(
        model, batches[0], cfg, probe_gates, gather=False)
    gathered_loss, _ = forward_train(
        model, batches[0], cfg, probe_gates, gather=True)
    model.train()
    hard_forward_error = abs(float((masked_loss - gathered_loss).detach()))
    result = meta_step(
        model, optimizer, router, router_optimizer, batches[0], batches[1:],
        cfg, args, False, generator)
    result["hard_masked_minus_gathered_loss"] = hard_forward_error
    expected_active = sum(
        min(int(row.sum()), max(1, int(round(args.budget * int(row.sum())))))
        for row in pad) * cfg.n_layers
    expected_fraction = expected_active / (cfg.n_layers * int(pad.sum()))
    result["expected_active_fraction"] = expected_fraction
    result["passed"] = bool(
        np.isfinite(result["router_grad_norm"])
        and result["router_grad_norm"] > 0 and result["state_detached"]
        and hard_forward_error < 1e-6
        and abs(result["active_fraction"] - expected_fraction) < 1e-7)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("checkpoint meta-step smoke test failed")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(handler=lambda args: self_test())
    smoke = sub.add_parser("smoke")
    run = sub.add_parser("run")
    for target in (smoke, run):
        target.add_argument("--ckpt", required=True)
        target.add_argument("--device", default=None)
        target.add_argument("--size", default="small")
        target.add_argument("--seq-len", type=int, default=256)
        target.add_argument("--task", default="tagged")
        target.add_argument("--batch-size", type=int, default=32)
        target.add_argument("--budget", type=float, default=0.5)
        target.add_argument("--reward-batches", type=int, default=4)
        target.add_argument("--body-lr", type=float, default=0.0003)
        target.add_argument("--router-lr", type=float, default=0.001)
        target.add_argument("--router-hidden", type=int, default=128)
        target.add_argument("--dropout", type=float, default=0.1)
        target.add_argument("--seed", type=int, default=9104)
    run.add_argument("--outdir", required=True)
    run.add_argument("--arms", default="meta,shuffled,random,dense,dense-fewer")
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--meta-every", type=int, default=10)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument("--eval-every", type=int, default=50)
    run.add_argument("--eval-batches", type=int, default=4)
    run.add_argument("--split", choices=["development", "confirmation"],
                     default="development")
    smoke.set_defaults(handler=checkpoint_smoke)

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
    if hasattr(args, "reward_batches") and args.reward_batches < 1:
        parser.error("--reward-batches must be positive")
    if hasattr(args, "batch_size") and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
