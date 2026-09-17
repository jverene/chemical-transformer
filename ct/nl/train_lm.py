"""NL training driver: Stage-1 continue-training, oracle-supervised forks,
and the fixed/MoD single-stage arms. Mirrors the arithmetic JSON schema.

Stage 1 (--stage 1 --method baseline): dense continue-train on the train
split; its checkpoint is shared by every fork (paired-by-seed design).

Stage 2 (--stage 2 --method ours-gategrad|ours-loss|shuffled): attaches
DifficultyHeads, trains on the oracle-covered pool with MSE targets + budget;
body at lr*0.01 unless --full-unfreeze. Targets come from the sidecar:
  ours-gategrad -> bins_grad (0 = irreducible floor 0.1, then quartiles)
  ours-loss     -> bins_loss (loss quartiles, the proxy-difficulty ablation)
  shuffled      -> gategrad targets permuted within sequence (control)

Single-stage arms (--stage 1 --method fixed|mod): end-to-end with gates on,
same data/budget as baseline.

Billed training FLOPs use the shared 3x-forward accounting; the oracle
prepass cost is a shared one-off and is NOT in per-run cum_train_flops
(it is reported in oracle_diag.json).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from .lm import GatedLM, GATE_TARGETS

GRAD_TARGETS = GATE_TARGETS
LOSS_TARGETS = [0.2, 0.4, 0.6, 0.8]


def targets_from_bins(bins, kind, device):
    table = torch.tensor(GRAD_TARGETS if kind == "grad" else LOSS_TARGETS,
                         dtype=torch.float32, device=device)
    return table[bins.long()]


def targets_from_bins_lay(bins_lay, device):
    """Per-(token,layer) quartile bins (L,B,S) -> per-layer gate targets."""
    table = torch.tensor(GRAD_TARGETS, dtype=torch.float32, device=device)
    return table[bins_lay.long()]


def shuffled_targets(target):
    """Per-sequence within-sequence permutation of the target tensor (B,S)."""
    B, S = target.shape
    perm = torch.rand(B, S, device=target.device).argsort(dim=1)
    return target.gather(1, perm)


@torch.no_grad()
def evaluate(lm, held_x, held_y, held_bins, batch_seqs=4, max_batches=None):
    lm.model.eval()
    n = held_x.shape[0]
    nb = (n + batch_seqs - 1) // batch_seqs
    if max_batches is not None:
        nb = min(nb, max_batches)
    ce_sum, tok_n = 0.0, 0
    bin_ce, bin_n = {}, {}
    gate_bins = defaultdict(float)
    gate_n = defaultdict(int)
    flops_sum, flops_tok = 0.0, 0
    nbins = int(held_bins.max()) + 1
    bin_ce = np.zeros(nbins)
    bin_n = np.zeros(nbins)
    for i in range(0, nb * batch_seqs, batch_seqs):
        x = held_x[i:i + batch_seqs].to(lm.device)
        y = held_y[i:i + batch_seqs].to(lm.device)
        bins = held_bins[i:i + x.shape[0]]
        logits = lm.model(x).logits
        ce = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), y.reshape(-1),
            reduction="none").view_as(y)
        ce_sum += float(ce.sum())
        tok_n += y.numel()
        for b in range(nbins):
            m = bins == b
            bin_ce[b] += float(ce[m].sum())
            bin_n[b] += int(m.sum())
        g = lm.gates_per_layer().mean(0)  # (B, S)
        flops_sum += float(lm.billed_flops(x.shape, g))
        flops_tok += y.numel()
        for b in range(nbins):
            m = bins == b
            gate_bins[b] += float(g[m].sum())
            gate_n[b] += int(m.sum())
    out = {
        "held_loss": ce_sum / max(tok_n, 1),
        "bin_loss": (bin_ce / np.maximum(bin_n, 1)).tolist(),
        "flops_per_token": flops_sum / max(flops_tok, 1),
        "gate_by_bin": ([gate_bins[b] / gate_n[b] for b in range(nbins)]
                        if gate_n[0] else None),
    }
    lm.model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--method", required=True,
                   choices=["baseline", "fixed", "mod", "rotation", "ours-gategrad",
                            "ours-loss", "shuffled"])
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-seqs", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--body-lr-frac", type=float, default=0.01)
    p.add_argument("--full-unfreeze", action="store_true")
    p.add_argument("--mse-weight", type=float, default=5.0)
    p.add_argument("--budget-weight", type=float, default=0.5)
    p.add_argument("--budget-target", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8,
                   help="held batches per mid-train eval (full set at final)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-batches per optimizer step (loss/accum each)")
    p.add_argument("--max-skip", type=int, default=8,
                   help="abort after this many consecutive nonfinite steps")
    p.add_argument("--per-layer-targets", action="store_true",
                   help="stage-2: use per-(token,layer) oracle bins "
                        "(requires oracle.npz with pool_bins_grad_lay)")
    p.add_argument("--suffix", default="",
                   help="output-filename suffix, e.g. '-full-unfreeze'")
    p.add_argument("--data-root", default="data/nl")
    p.add_argument("--out-root", default="results-nl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.out_root, args.domain), exist_ok=True)
    tag = f"{args.method}{args.suffix}_seed{args.seed}"
    out_json = os.path.join(args.out_root, args.domain, f"{tag}.json")
    if os.path.exists(out_json):
        print(f"skip {out_json}: exists")
        return

    data_dir = os.path.join(args.data_root, args.domain)
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    train = np.load(os.path.join(data_dir, "train_tokens.npy"), mmap_mode="r")
    held = np.load(os.path.join(data_dir, "heldout_tokens.npy"), mmap_mode="r")
    # data sanity: an out-of-vocab id (corrupt packing, uint32 garbage) shows
    # up as a NaN/garbage loss many steps later; fail here instead.
    vocab = None
    try:
        from transformers import AutoConfig
        vocab = AutoConfig.from_pretrained(args.model).vocab_size
    except Exception:
        pass
    if vocab is not None:
        tmax, hmax = int(train[:].max()), int(held[:].max())
        if tmax >= vocab or hmax >= vocab:
            raise ValueError(
                f"token id out of vocab ({max(tmax, hmax)} >= {vocab}) — "
                f"data pipeline corrupt")
    held_x = torch.from_numpy(np.asarray(held)[:, :-1].astype(np.int64))
    held_y = torch.from_numpy(np.asarray(held)[:, 1:].astype(np.int64))
    oracle = np.load(os.path.join(data_dir, "oracle.npz")) \
        if os.path.exists(os.path.join(data_dir, "oracle.npz")) else None
    held_bins = (torch.from_numpy(oracle["held_bins_loss"].astype(np.int64))
                 if oracle is not None else torch.zeros(held_x.shape[0],
                                                        held_x.shape[1],
                                                        dtype=torch.int64))

    stage1_ckpt = os.path.join(args.out_root, args.domain,
                               f"baseline_seed{args.seed}.pt")
    two_stage = args.method in ("ours-gategrad", "ours-loss", "shuffled")

    lm = GatedLM(args.model, args.device, method=args.method)
    if two_stage or args.method in ("fixed", "mod"):
        # fixed/mod could train from scratch-pretrained; we fork the same
        # Stage-1 body so every method shares its starting point.
        lm.model.load_state_dict(torch.load(stage1_ckpt,
                                            map_location=args.device))
    lm.model.train()

    body_lr_frac = 1.0 if (args.full_unfreeze or not two_stage) \
        else args.body_lr_frac
    opt = torch.optim.AdamW(lm.param_groups(args.lr, body_lr_frac))
    sched = None
    warmup = max(1, int(0.02 * args.steps))

    def lr_fn(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    rng = np.random.RandomState(args.seed * 100003 + 7)
    kind = {"ours-gategrad": "grad", "ours-loss": "loss",
            "shuffled": "grad"}.get(args.method)
    pool_n = oracle["pool_bins_grad"].shape[0] if oracle is not None else 0

    history = defaultdict(list)
    cum_flops = 0.0
    n_skipped = 0
    consecutive_bad = 0
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        loss_sum, flops_step = 0.0, 0.0
        step_finite = True
        for _ in range(args.grad_accum):
            idx = rng.choice(pool_n if two_stage else train.shape[0],
                             size=args.batch_seqs, replace=False)
            chunk = torch.from_numpy(np.asarray(
                train[idx] if not two_stage else train[:pool_n][idx],
                dtype=np.int64)).to(args.device)
            x, y = chunk[:, :-1], chunk[:, 1:]
            target = None
            target_lay = None
            if two_stage:
                if args.per_layer_targets:
                    raw = oracle["pool_bins_grad_lay"][:, idx]
                    b = torch.from_numpy(np.asarray(raw, dtype=np.int64)).to(
                        args.device)                      # (L, B, S)
                    target = None
                    target_lay = targets_from_bins_lay(b, args.device)
                else:
                    raw = (oracle["pool_bins_grad"] if kind == "grad"
                           else oracle["pool_bins_loss"])[idx]
                    b = torch.from_numpy(raw.astype(np.int64)).to(args.device)
                    target = targets_from_bins(b, kind, args.device)
                    target_lay = None
                if args.method == "shuffled":
                    target = shuffled_targets(target)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.amp and args.device == "cuda"):
                loss, info = lm.loss_and_gates(
                    x, y, target=target, target_lay=target_lay,
                    mse_weight=args.mse_weight,
                    budget_weight=args.budget_weight,
                    budget_target=args.budget_target)
                if not torch.isfinite(loss):
                    step_finite = False
                    break
                (loss / args.grad_accum).backward()
            loss_sum += float(loss)
            flops_step += float(info["flops"])
        if not step_finite:
            # discard partial grads; a lone spike should not kill a 10h run
            opt.zero_grad()
            n_skipped += 1
            consecutive_bad += 1
            print(f"[{args.domain}/{args.method}/s{args.seed}] step {step} | "
                  f"NONFINITE loss — skipped ({n_skipped} total)", flush=True)
            if consecutive_bad >= args.max_skip:
                print(f"aborting: {consecutive_bad} consecutive nonfinite "
                      f"steps — training is diverging, not spiking", flush=True)
                sys.exit(3)
            continue
        consecutive_bad = 0
        torch.nn.utils.clip_grad_norm_(lm.model.parameters(), 1.0)
        opt.step()
        sched.step()
        cum_flops += 3.0 * flops_step
        loss = loss_sum / args.grad_accum

        if step % args.eval_every == 0 or step == args.steps - 1:
            m = evaluate(lm, held_x, held_y, held_bins,
                         max_batches=args.eval_batches)
            history["step"].append(step)
            history["train_loss"].append(float(loss))
            history["held_loss"].append(m["held_loss"])
            history["bin_loss"].append(m["bin_loss"])
            history["flops_per_token"].append(m["flops_per_token"])
            history["gate_by_bin"].append(m["gate_by_bin"])
            history["cum_train_flops"].append(cum_flops)
            gb = m["gate_by_bin"]
            print(f"[{args.domain}/{args.method}/s{args.seed}] step {step} | "
                  f"loss {float(loss):.4f} | held {m['held_loss']:.4f} | "
                  f"MFLOPs/tok {m['flops_per_token'] / 1e6:.1f} | "
                  f"gate bins "
                  f"{[round(v, 2) for v in gb] if gb else '-'} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    final = evaluate(lm, held_x, held_y, held_bins)
    if args.method == "rotation":
        # dense-inference eval: gates temporarily off — the parity comparison
        for w in lm.wrappers:
            w.gate_mode = "none"
        m_dense = evaluate(lm, held_x, held_y, held_bins)
        final["dense_inference_held_loss"] = m_dense["held_loss"]
        for w in lm.wrappers:
            w.gate_mode = "rotation"
    final["cum_train_flops"] = cum_flops
    final["train_wallclock_s"] = time.time() - t0
    final["skipped_nonfinite_steps"] = n_skipped
    result = {
        "method": args.method, "domain": args.domain, "seed": args.seed,
        "stage": args.stage, "model": args.model,
        "config": vars(args), "history": dict(history), "final": final,
        "oracle_flops_shared": "see oracle_diag.json (prepass, one-off)",
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    if args.stage == 1 and args.method == "baseline":
        torch.save(lm.model.state_dict(), stage1_ckpt)
    elif two_stage:
        torch.save({f"wrapper_{i}.head": w.head.state_dict()
                    for i, w in enumerate(lm.wrappers)},
                   out_json.replace(".json", "_heads.pt"))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
