"""Training and evaluation loops shared by every method/task/seed."""
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

from .tokenizer import TOKENIZER
from .tasks import get_task
from .tasks.base import get_batch
from .models import build_model
from .flops import FLOPsCounter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def eval_loop(model, held_out, task, cfg, n_batches: int, collect: bool = False) -> dict:
    """Held-out evaluation. Accuracy is computed on answer tokens only."""
    model.eval()
    rng = random.Random(999)  # fixed: identical eval batches at every call
    loss_sum, loss_n = 0.0, 0
    correct_n, ans_n = 0, 0
    bin_correct = [0] * task.n_bins
    bin_total = [0] * task.n_bins
    gate_sum = [0.0] * task.n_bins
    gate_n = [0] * task.n_bins
    flops_sum, tok_sum = 0.0, 0
    ent_list, corr_list, compute_list = [], [], []
    ffn_tok = FLOPsCounter.ffn_per_token(cfg.d_model, cfg.d_ff) * cfg.n_layers

    for _ in range(n_batches):
        x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size, held_out=held_out)
        pad = x != TOKENIZER.pad_id
        logits, info = model(x, pad_mask=pad, meta={"diffs": diffs})
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                               ignore_index=TOKENIZER.pad_id)
        loss_sum += loss.item() * int(pad.sum())
        loss_n += int(pad.sum())

        preds = logits.argmax(dim=-1)
        tgt_valid = y != TOKENIZER.pad_id
        ans_t = torch.roll(ans, -1, dims=1)
        ans_t[:, -1] = 0
        diffs_t = torch.roll(diffs, -1, dims=1)
        diffs_t[:, -1] = -1
        m = tgt_valid & (ans_t == 1)
        corr = (preds == y) & m
        correct_n += int(corr.sum())
        ans_n += int(m.sum())
        for b in range(task.n_bins):
            mb = m & (diffs_t == b)
            bin_correct[b] += int((corr & mb).sum())
            bin_total[b] += int(mb.sum())

        n_real = int(pad.sum())
        flops_sum += info["flops"]
        tok_sum += n_real

        g = info.get("gate_mean")
        if g is not None:
            for b in range(task.n_bins):
                gb = pad & (diffs == b)
                if gb.any():
                    gate_sum[b] += float(g[gb].sum())
                    gate_n[b] += int(gb.sum())

        if collect:
            if info.get("entropy_mean") is not None:
                ent_list.append(info["entropy_mean"][m].float().cpu().numpy())
                corr_list.append(corr[m].float().cpu().numpy())
            if g is not None:
                attn_per_tok = FLOPsCounter.attention(
                    x.shape[0], x.shape[1], cfg.d_model, cfg.n_heads) / x.numel()
                per_tok = attn_per_tok + g * cfg.n_layers * \
                    FLOPsCounter.ffn_per_token(cfg.d_model, cfg.d_ff)
                compute_list.append(per_tok[pad].float().cpu().numpy())

    bin_acc = [bc / bt if bt else 0.0 for bc, bt in zip(bin_correct, bin_total)]
    total_gate_n = sum(gate_n)
    out = {
        "loss": loss_sum / max(loss_n, 1),
        "acc": correct_n / max(ans_n, 1),
        "balanced_acc": float(np.mean(bin_acc)),
        "bin_acc": bin_acc,
        "bin_total": bin_total,
        "flops_per_token": flops_sum / max(tok_sum, 1),
        "gate_by_bin": [gs / gn if gn else None for gs, gn in zip(gate_sum, gate_n)],
        "mean_gate": (sum(gate_sum) / total_gate_n) if total_gate_n else None,
    }
    if collect and ent_list:
        ent = np.concatenate(ent_list)
        cor = np.concatenate(corr_list)
        out["entropy_pearson"] = float(scipy_stats.pearsonr(ent, cor).statistic)
        out["entropy_spearman"] = float(scipy_stats.spearmanr(ent, cor).statistic)
        out["entropy_n"] = int(len(ent))
        idx = np.random.RandomState(0).choice(len(ent), size=min(20000, len(ent)),
                                              replace=False)
        out["entropy_samples"] = [round(float(v), 4) for v in ent[idx]]
        out["correct_samples"] = [int(v) for v in cor[idx]]
    if collect and compute_list:
        comp = np.concatenate(compute_list)
        idx = np.random.RandomState(0).choice(len(comp), size=min(20000, len(comp)),
                                              replace=False)
        out["compute_samples"] = [round(float(v), 1) for v in comp[idx]]
    return out


@torch.no_grad()
def wallclock_eval(model, held_out, task, cfg) -> float:
    """Seconds per real (non-pad) token for a full forward pass, eval mode."""
    model.eval()
    rng = random.Random(4242)
    _sync(cfg.device)
    t0 = time.perf_counter()
    tok = 0
    for _ in range(cfg.wallclock_batches):
        x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size, held_out=held_out)
        pad = x != TOKENIZER.pad_id
        model(x, pad_mask=pad, meta={"diffs": diffs})
        tok += int(pad.sum())
    _sync(cfg.device)
    return (time.perf_counter() - t0) / max(tok, 1)


def train_run(method: str, task_name: str, seed: int, cfg,
              out_path: str, ckpt: bool = True, verbose: bool = True) -> dict:
    set_seed(seed)
    task = get_task(task_name)
    held_out = task.build_held_out(cfg)
    model = build_model(method, cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Stage detection
    is_predictor_stage1 = (method in ("predictor", "predictor-supervised")
                           and getattr(cfg, "stage", 1) == 1
                           and getattr(cfg, "predictor_collect_targets", False))
    is_predictor_stage2 = (method in ("predictor", "predictor-supervised")
                           and getattr(cfg, "stage", 1) == 2)

    # Option 3b specific
    is_supervised = (method == "predictor-supervised")

    # Target accumulation for Stage 1 (Option 3b: supervised from difficulty tags)
    target_accum = None
    target_count = 0
    target_start_step = None

    if is_predictor_stage2:
        # Load Stage 1 checkpoint
        if is_supervised:
            # Option 3b: predictor-supervised
            stage1_path = out_path.replace("predictor-supervised_seed", "predictor-supervised_stage1_seed").replace(".json", ".pt")
        else:
            # Option 2: predictor
            stage1_path = out_path.replace("predictor_seed", "predictor_stage1_seed").replace(".json", ".pt")
        if os.path.exists(stage1_path):
            print(f"Loading Stage 1 checkpoint from {stage1_path}")
            model.load_state_dict(torch.load(stage1_path, map_location=cfg.device))
        else:
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found at {stage1_path}. "
                f"Must run Stage 1 first — Stage 2 starts from random init without it."
            )

    # Optimizer setup
    if is_predictor_stage2 and is_supervised:
        # Option 3b: supervised difficulty with budget
        # Unfreeze body with low LR, full LR for DifficultyHead
        body_params = []
        head_params = []
        for name, p in model.named_parameters():
            if "diff_head" in name:
                head_params.append(p)
            else:
                body_params.append(p)
        opt = torch.optim.AdamW([
            {"params": body_params, "lr": cfg.lr * 0.01},
            {"params": head_params, "lr": cfg.lr},
        ])
        sparsity_lambda = cfg.sparsity_lambda  # from config
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        sparsity_lambda = cfg.sparsity_lambda

    rng = random.Random(seed * 100003 + 7)

    history = defaultdict(list)
    t0 = time.time()
    if verbose:
        print(f"\n=== {method} / {task_name} / seed {seed} "
              f"({n_params:,} params, device {cfg.device}) ===", flush=True)

    for step in range(cfg.n_steps):
        model.train()
        x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size)
        pad = x != TOKENIZER.pad_id
        logits, info = model(x, pad_mask=pad, meta={"diffs": diffs})

        # Cross-entropy
        ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                             ignore_index=TOKENIZER.pad_id)

        # Option 3b Stage 1: no target accumulation needed (targets from tags)
        # Just compute CE
        ce_loss = ce

        # Loss computation
        loss = ce_loss + sparsity_lambda * info.get("sparsity_penalty", 0)

        # Option 3b Stage 2: supervised difficulty + budget
        if is_predictor_stage2 and "predictor-supervised" in method:
            if "diff_score_mean" in info:
                pred_score = info["diff_score_mean"]  # (B, S) raw score before sigmoid
                # Per-position difficulty targets from tags
                target_gate_e = getattr(cfg, "predictor_target_gate_e", 0.2)
                target_gate_m = getattr(cfg, "predictor_target_gate_m", 0.5)
                target_gate_h = getattr(cfg, "predictor_target_gate_h", 0.8)

                # Create per-position targets from diffs
                if getattr(cfg, "predictor_digit_targets", False):
                    # Option B (mixed): digit-count bins 0..3 -> 0.2/0.4/0.6/0.8
                    target = torch.full(diffs.shape, 0.5, device=diffs.device)
                    valid = diffs >= 0
                    target[valid] = 0.2 + 0.2 * diffs[valid].float()
                else:
                    # E (0) -> 0.2, M (1) -> 0.5, H (2) -> 0.8, none (-1) -> 0.5
                    target = torch.where(diffs == 0, target_gate_e,
                                 torch.where(diffs == 1, target_gate_m,
                                 torch.where(diffs == 2, target_gate_h, 0.5)))

                # MSE loss on sigmoid(score) vs target (both in [0,1] space)
                pred_gate = torch.sigmoid(pred_score)
                mse_loss = F.mse_loss(pred_gate, target.float().detach())

                # Budget loss: (mean_gate - target_budget)^2
                mean_gate = info["gate_mean"].mean()
                budget_loss = (mean_gate - getattr(cfg, "predictor_budget_target", 0.5)) ** 2

                mse_weight = getattr(cfg, "predictor_mse_weight", 5.0)
                budget_weight = getattr(cfg, "predictor_budget_weight", 0.5)

                loss = loss + mse_weight * mse_loss + budget_weight * budget_loss

                # Debug logging (first step only)
                if step == 0:
                    t = target.float().detach()
                    print(f"  [debug] Target unique: {t.unique().tolist()}")
                    print(f"  [debug] Target samples: {t[0, :10].tolist()}")
                    print(f"  [debug] Pred gate samples: {pred_gate[0, :10].tolist()}")
                    print(f"  [debug] MSE: {mse_loss.item():.4f}, Budget: {budget_loss.item():.4f}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.n_steps - 1:
            m = eval_loop(model, held_out, task, cfg, cfg.eval_batches)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["eval_loss"].append(m["loss"])
            history["acc"].append(m["acc"])
            history["balanced_acc"].append(m["balanced_acc"])
            history["bin_acc"].append(m["bin_acc"])
            history["flops_per_token"].append(m["flops_per_token"])
            history["gate_by_bin"].append(m["gate_by_bin"])
            if verbose:
                bins = " ".join(f"{a:.3f}" for a in m["bin_acc"])
                print(f"[{method}/{task_name}/s{seed}] step {step:5d} | "
                      f"loss {loss.item():.4f} | acc {m['acc']:.3f} | "
                      f"bal {m['balanced_acc']:.3f} | bins {bins} | "
                      f"MFLOPs/tok {m['flops_per_token'] / 1e6:.1f} | "
                      f"{time.time() - t0:.0f}s", flush=True)

    final = eval_loop(model, held_out, task, cfg, cfg.final_eval_batches, collect=True)
    final["wallclock_per_token"] = wallclock_eval(model, held_out, task, cfg)
    final["train_wallclock_s"] = time.time() - t0

    # Debug summary for predictor-supervised
    if is_supervised and is_predictor_stage2:
        gb = final.get("gate_by_bin", [None, None, None])
        ge, gm, gh = gb[0] or 0, gb[1] or 0, gb[2] or 0
        spread = gh - ge
        print(f"\n  === Option 3b Summary ===")
        print(f"  Gate E/M/H: {ge:.3f} / {gm:.3f} / {gh:.3f}")
        print(f"  Gate spread (H-E): {spread:.3f}")
        print(f"  Mean gate: {final.get('mean_gate', 'N/A')}")
        print(f"  Accuracy: {final['acc']:.3f}")
        if spread > 0.15 and ge < gm < gh:
            print(f"  >>> GO: correlation looks positive, spread > 0.15")
        elif spread < 0.05:
            print(f"  >>> STOP: spread < 0.05, hidden states don't encode difficulty")
        else:
            print(f"  >>> TWEAK: spread {spread:.3f} in [0.05, 0.15], try mse=10.0 budget=0.1")

    result = {
        "method": method,
        "task": task_name,
        "seed": seed,
        "params": n_params,
        "trainable_params": n_trainable,
        "config": cfg.to_dict(),
        "history": dict(history),
        "final": final,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    if ckpt:
        torch.save(model.state_dict(), out_path.replace(".json", ".pt"))
    if verbose:
        print(f"saved {out_path}", flush=True)
    return result