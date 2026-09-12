"""Gate-gradient difficulty oracle and the P0 experiment.

The marginal value of FFN compute for token t is measured on the frozen
Stage-1 body by evaluating dL/dg_{t,l} at g=1 in a single forward+backward
(gates enter as leaf tensors). The token score is

    score_t = -sum_l dL/dg_{t,l}      (marginal loss reduction per unit FFN)

Scores <= 0 mean extra compute does not reduce loss (irreducible hardness);
they are floored to the lowest target (0.1). Positive scores are
quartile-binned per batch to 0.2/0.4/0.6/0.8, mirroring the tag targets.

P0 pairs two Stage-2 arms against the SAME Stage-1 checkpoint per seed:
  - predictor-supervised  (tag targets, the workshop recipe)
  - gategrad              (gate-gradient targets, identical hyperparameters)
Success gate: paired accuracy at matched billed training FLOPs; bin spread is
explicitly NOT the metric (gate-grad targets vary within problems, so a
better allocation can show lower E/M/H spread).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, get_device
from .models import build_model
from .tasks import get_task
from .tasks.base import get_batch
from .tokenizer import TOKENIZER
from .train import train_run

QUARTILE_TARGETS = (0.2, 0.4, 0.6, 0.8)
FLOOR_TARGET = 0.1


def build_oracle(stage1_path: str, cfg):
    """Frozen Stage-1 body used to measure per-token compute value."""
    cfg1 = Config(**cfg.to_dict())
    cfg1.stage = 1
    cfg1.grad_checkpoint = False
    model = build_model("predictor-supervised", cfg1).to(cfg.device)
    model.load_state_dict(torch.load(stage1_path, map_location=cfg.device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def gategrad_targets(oracle, x, y, diffs, cfg):
    """One fwd+bwd on the frozen body -> (target (B,S), diagnostics dict)."""
    B, S = x.shape
    pad = x != TOKENIZER.pad_id
    leaves = [torch.ones(B, S, device=x.device, requires_grad=True)
              for _ in range(cfg.n_layers)]
    logits, oinfo = oracle(x, pad_mask=pad, meta={"diffs": diffs},
                           oracle_gates=leaves)
    ce_per_tok = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1),
                                 ignore_index=TOKENIZER.pad_id,
                                 reduction="none").view(B, S)
    loss = ce_per_tok[pad].mean() if pad.any() else ce_per_tok.mean()
    loss.backward()
    grads = torch.stack([g.grad for g in leaves], dim=0)        # (L, B, S)
    score = -grads.sum(dim=0).detach()                          # (B, S)

    problem = pad & (diffs >= 0)
    target = torch.full((B, S), 0.5, device=x.device)  # pads/gaps, as tag arm
    diag = {"flops": float(oinfo["flops"])}
    if problem.any():
        s_np = score[problem].cpu().numpy()
        frac_nonpos = float((s_np <= 0).mean())
        tgt_np = np.full(s_np.shape, FLOOR_TARGET, dtype=np.float32)
        pos = s_np > 0
        if pos.any():
            edges = np.quantile(s_np[pos], [0.25, 0.5, 0.75])
            tgt_np[pos] = np.array(QUARTILE_TARGETS,
                                   dtype=np.float32)[np.digitize(s_np[pos], edges)]
        target[problem] = torch.from_numpy(tgt_np).to(x.device)
        ce = ce_per_tok[problem].detach().cpu().numpy()
        diag.update({
            "frac_nonpos": frac_nonpos,
            "mean_score": float(s_np.mean()),
            "mean_target_problem": float(tgt_np.mean()),
            "mean_target_all": float(target[pad].mean()),
            "score_ce_pearson": float(np.corrcoef(s_np, ce)[0, 1]),
            "score_ce_spearman": float(_spearman(s_np, ce)),
            "layer_grad_mean": [float(grads[l][problem].mean())
                                for l in range(cfg.n_layers)],
        })
        # Falsifiable scale prediction: as the model grows, reducible mass
        # migrates toward harder bins -> per-bin mean score and frac_nonpos
        # on H should approach M's across sizes.
        s_full = score.cpu().numpy()
        d_np = diffs.cpu().numpy()
        bins_present = sorted(set(d_np[d_np >= 0].tolist()))
        bin_score, bin_nonpos, bin_n = [], [], []
        for b in bins_present:
            m = (d_np == b) & pad.cpu().numpy()
            bin_score.append(float(s_full[m].mean()))
            bin_nonpos.append(float((s_full[m] <= 0).mean()))
            bin_n.append(int(m.sum()))
        diag["bin_ids"] = bins_present
        diag["bin_score_mean"] = bin_score
        diag["bin_frac_nonpos"] = bin_nonpos
        diag["bin_n"] = bin_n
    return target.detach(), diag


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def run_p0(seeds, outdir, device=None, stage1_steps=5000, stage2_steps=3000,
           verbose=True, task_name="tagged", size="small",
           arms=("stage1", "tag", "gategrad", "shuffled"), lr=None,
           batch_size=None, seq_len=None):
    device = device or get_device()
    for seed in seeds:
        # Size-qualify output names so multi-size probes don't clobber each
        # other; the 11M testbed keeps its legacy names.
        sfx = "" if size == "small" else f"_{size}"
        overrides = {"device": device, "eval_every": 500,
                     "stage2_mode": "supervised_budget"}
        if size != "small":
            # RoPE+SDPA at >=12 layers diverges with constant LR and no
            # warmup (Sep 7 P1 incident); use the validated window-probe
            # recipe for every non-testbed size.
            overrides["lr_schedule"] = "cosine"
            overrides["weight_decay"] = 0.1
        if batch_size is not None:
            overrides["batch_size"] = batch_size
        if seq_len is not None:
            overrides["seq_len"] = seq_len
        def mk(**extra):
            cfg = Config.for_size(size, **overrides)
            if lr is not None:
                cfg.lr = lr
            for k, v in extra.items():
                setattr(cfg, k, v)
            return cfg

        stage1_path = os.path.join(
            outdir, task_name, f"predictor-supervised_stage1{sfx}_seed{seed}.pt")
        stage1_json = stage1_path.replace(".pt", ".json")
        tag_json = os.path.join(
            outdir, task_name, f"predictor-supervised{sfx}_seed{seed}.json")
        gg_json = os.path.join(
            outdir, task_name, f"gategrad{sfx}_seed{seed}.json")

        # 1. Shared Stage-1 body
        if not os.path.exists(stage1_path):
            if "stage1" in arms:
                cfg1 = mk(stage=1, n_steps=stage1_steps)
                train_run("predictor-supervised", task_name, seed, cfg1,
                          stage1_json, ckpt=True, verbose=verbose)
            else:
                raise FileNotFoundError(
                    f"{stage1_path} missing and 'stage1' not in arms "
                    f"({arms}) — stage the checkpoint or enable the arm")
        else:
            print(f"Stage 1 exists: {stage1_path}")

        # 1b. Fixed-two-stage control: constant targets on the two-stage
        # protocol (closes the protocol-vs-content 2x2; no oracle pass).
        fx_json = os.path.join(
            outdir, task_name, f"fixed2stage{sfx}_seed{seed}.json")
        if "fixed2stage" in arms and not os.path.exists(fx_json):
            cfg_fx = mk(stage=2, n_steps=stage2_steps)
            cfg_fx.predictor_constant_target = 0.5
            train_run("predictor-supervised", task_name, seed, cfg_fx,
                      fx_json, ckpt=False, verbose=verbose,
                      stage1_ckpt=stage1_path)

        # 2. Tag-target arm (workshop recipe, hyperparameter-identical)
        if "tag" in arms and not os.path.exists(tag_json):
            cfg_tag = mk(stage=2, n_steps=stage2_steps)
            train_run("predictor-supervised", task_name, seed, cfg_tag,
                      tag_json, ckpt=False, verbose=verbose,
                      stage1_ckpt=stage1_path)

        # 3. Gate-grad arm (only the target source differs)
        if "gategrad" in arms and not os.path.exists(gg_json):
            cfg_gg = mk(stage=2, n_steps=stage2_steps)
            cfg_gg.predictor_target_source = "gategrad"
            train_run("predictor-supervised", task_name, seed, cfg_gg,
                      gg_json, ckpt=True, verbose=verbose,
                      stage1_ckpt=stage1_path)

        # 4. Shuffled-target control: gategrad targets permuted within
        # sequence — closes the "any varying targets would do" attack.
        sh_json = os.path.join(outdir, task_name, f"shuffled{sfx}_seed{seed}.json")
        if "shuffled" in arms and not os.path.exists(sh_json):
            cfg_sh = mk(stage=2, n_steps=stage2_steps)
            cfg_sh.predictor_target_source = "gategrad"
            cfg_sh.predictor_shuffle_targets = True
            train_run("predictor-supervised", task_name, seed, cfg_sh,
                      sh_json, ckpt=False, verbose=verbose,
                      stage1_ckpt=stage1_path)

    if "tag" in arms or "gategrad" in arms or "shuffled" in arms:
        compare_p0(outdir, seeds, task_name)


def compare_p0(outdir, seeds, task_name="tagged"):
    """Paired comparison: accuracy at matched billed training FLOPs."""
    rows = {}
    for arm, fname in (("tag", "predictor-supervised_seed{s}.json"),
                       ("gategrad", "gategrad_seed{s}.json"),
                       ("shuffled", "shuffled_seed{s}.json")):
        rows[arm] = {}
        for s in seeds:
            p = os.path.join(outdir, task_name, fname.format(s=s))
            if not os.path.exists(p):
                continue
            with open(p) as f:
                r = json.load(f)
            rows[arm][s] = _interp_at_budget(r)

    print("\n=== P0 paired comparison (tagged) ===")
    print(f"{'seed':>4} | {'arm':>8} | {'acc':>6} | {'bal':>6} | "
          f"{'MFLOPs/tok':>10} | {'cumTrainGFLOPs':>14} | gate E/M/H")
    for s in seeds:
        for arm in ("tag", "gategrad", "shuffled"):
            if s not in rows[arm]:
                continue
            r = rows[arm][s]
            gb = r.get("gate_by_bin") or [float("nan")] * 3
            print(f"{s:>4} | {arm:>8} | {r['acc']:6.3f} | {r['balanced_acc']:6.3f} | "
                  f"{r['flops_per_token'] / 1e6:10.1f} | "
                  f"{r['cum_train_flops'] / 1e9:14.1f} | "
                  f"{gb[0]:.2f}/{gb[1]:.2f}/{gb[2]:.2f}")
    for arm in ("tag", "gategrad", "shuffled"):
        accs = [rows[arm][s]["acc"] for s in seeds if s in rows[arm]]
        if accs:
            print(f"{arm}: mean acc {np.mean(accs):.3f} ± {np.std(accs):.3f} "
                  f"({len(accs)} seeds)")


def _interp_at_budget(r):
    """Final metrics evaluated at the smaller of the two arms' budgets is
    handled by the caller via history interpolation; here we report the
    endpoint plus the curve needed for interpolation."""
    final = dict(r["final"])
    final["history_steps"] = r["history"]["step"]
    final["history_acc"] = r["history"]["acc"]
    final["history_cum_flops"] = r["history"].get("cum_train_flops", [])
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--outdir", default="results-p0")
    p.add_argument("--device", default=None)
    p.add_argument("--task", default="tagged")
    p.add_argument("--size", default="small")
    p.add_argument("--arms", default="stage1,tag,gategrad,shuffled",
                   help="comma subset of: stage1,tag,gategrad,shuffled")
    p.add_argument("--lr", type=float, default=None,
                   help="Override the preset learning rate")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--stage1-steps", type=int, default=5000)
    p.add_argument("--stage2-steps", type=int, default=3000)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    run_p0([int(s) for s in args.seeds.split(",")], args.outdir,
           device=args.device, stage1_steps=args.stage1_steps,
           stage2_steps=args.stage2_steps, task_name=args.task,
           size=args.size, arms=tuple(a.strip() for a in args.arms.split(",")),
           lr=args.lr, batch_size=args.batch_size, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
