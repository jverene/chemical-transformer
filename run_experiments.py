"""Experiment runner: tasks x methods x seeds -> results/{task}/{method}_seed{s}.json

Skips runs whose JSON already exists (resumable). MoD capacity can be set to
"auto": it is then matched to the mean gate the Chemical model achieved on the
same task (averaged over available chemical seeds), falling back to 0.5.

For predictor (Option 2): runs two-stage training:
  Stage 1: full FFN (gate=1.0), n_steps = stage1_steps
  Stage 2: DifficultyHead gate, n_steps = stage2_steps, loads Stage 1 checkpoint

For predictor-v3 (Option 3): runs two-stage training with supervised difficulty targets:
  Stage 1: full FFN + collect per-token CE targets in last N steps
  Stage 2: freeze body, train DifficultyHead with MSE loss against per-token CE targets
"""
import argparse
import glob
import json
import os

from ct.config import Config
from ct.models import METHODS
from ct.train import train_run

TASK_METHODS = {
    "tagged": METHODS,
    # tag-only requires explicit difficulty tags -> not applicable to mixed
    "mixed": [m for m in METHODS if m != "tag-only"],
}


def resolve_mod_capacity(task: str, outdir: str) -> float:
    gates = []
    for path in glob.glob(os.path.join(outdir, task, "chemical_seed*.json")):
        with open(path) as f:
            r = json.load(f)
        mg = r["final"].get("mean_gate")
        if mg is not None:
            gates.append(mg)
    if gates:
        cap = sum(gates) / len(gates)
        print(f"MoD capacity auto-matched to Chemical mean gate: {cap:.3f}")
        return cap
    print("MoD capacity: no chemical results found, falling back to 0.5")
    return 0.5


def run_predictor_two_stage(task: str, seed: int, cfg: Config, outdir: str,
                            stage1_steps: int, stage2_steps: int,
                            eval_every: int, no_ckpt: bool, verbose: bool = True):
    """Run two-stage training for predictor mode (Option 2).

    Stage 1: stage=1 (full FFN), stage1_steps steps
    Stage 2: stage=2 (DifficultyHead), stage2_steps steps, loads Stage 1 ckpt
    """
    stage1_out = os.path.join(outdir, task, f"predictor_stage1_seed{seed}.json")
    stage2_out = os.path.join(outdir, task, f"predictor_seed{seed}.json")

    # Check if Stage 2 already complete
    if os.path.exists(stage2_out):
        print(f"skip {stage2_out}: exists")
        return

    # ---- STAGE 1 ----
    cfg1 = Config(**cfg.to_dict())
    cfg1.stage = 1
    cfg1.n_steps = stage1_steps

    if verbose:
        print(f"\n=== predictor Stage 1 / {task} / seed {seed} "
              f"({stage1_steps} steps) ===")

    if not os.path.exists(stage1_out):
        train_run("predictor", task, seed, cfg1, stage1_out,
                  ckpt=not no_ckpt, verbose=verbose)
    else:
        print(f"Stage 1 exists: {stage1_out}")

    # ---- STAGE 2 ----
    cfg2 = Config(**cfg.to_dict())
    cfg2.stage = 2
    cfg2.n_steps = stage2_steps

    if verbose:
        print(f"\n=== predictor Stage 2 / {task} / seed {seed} "
              f"({stage2_steps} steps) ===")

    # Stage 2 loads Stage 1 checkpoint automatically via train_run
    train_run("predictor", task, seed, cfg2, stage2_out,
              ckpt=not no_ckpt, verbose=verbose)


def run_predictor_v3_two_stage(task: str, seed: int, cfg: Config, outdir: str,
                               stage1_steps: int, stage2_steps: int,
                               target_steps: int, mse_weight: float,
                               eval_every: int, no_ckpt: bool, verbose: bool = True):
    """Run two-stage training for predictor-v3 (Option 3: supervised difficulty).

    Stage 1: stage=1, predictor_collect_targets=True, predictor_target_steps=target_steps
    Stage 2: stage=2, predictor_mse_weight=mse_weight, predictor_freeze_body=True
    """
    stage1_out = os.path.join(outdir, task, f"predictor-v3_stage1_seed{seed}.json")
    stage2_out = os.path.join(outdir, task, f"predictor-v3_seed{seed}.json")

    # Check if Stage 2 already complete
    if os.path.exists(stage2_out):
        print(f"skip {stage2_out}: exists")
        return

    # ---- STAGE 1 ----
    cfg1 = Config(**cfg.to_dict())
    cfg1.stage = 1
    cfg1.n_steps = stage1_steps
    cfg1.predictor_collect_targets = True
    cfg1.predictor_target_steps = target_steps

    if verbose:
        print(f"\n=== predictor-v3 Stage 1 / {task} / seed {seed} "
              f"({stage1_steps} steps, collect targets in last {target_steps}) ===")

    if not os.path.exists(stage1_out):
        train_run("predictor", task, seed, cfg1, stage1_out,
                  ckpt=not no_ckpt, verbose=verbose)
    else:
        print(f"Stage 1 exists: {stage1_out}")

    # ---- STAGE 2 ----
    cfg2 = Config(**cfg.to_dict())
    cfg2.stage = 2
    cfg2.n_steps = stage2_steps
    cfg2.predictor_mse_weight = 1.0
    cfg2.predictor_freeze_body = True

    if verbose:
        print(f"\n=== predictor-v3 Stage 2 / {task} / seed {seed} "
              f"({stage2_steps} steps, MSE supervision) ===")

    train_run("predictor", task, seed, cfg2, stage2_out,
              ckpt=not no_ckpt, verbose=verbose)


def run_predictor_supervised_two_stage(task: str, seed: int, cfg: Config, outdir: str,
                                       stage1_steps: int, stage2_steps: int,
                                       eval_every: int, no_ckpt: bool, verbose: bool = True):
    """Run two-stage training for predictor-supervised (Option 3b: supervised difficulty + budget).

    Stage 1: stage=1 (full FFN), stage1_steps steps
    Stage 2: stage=2 (predictor-supervised), stage2_steps steps, loads Stage 1 ckpt
    Uses supervised difficulty targets from tags + budget constraint.
    """
    stage1_out = os.path.join(outdir, task, f"predictor-supervised_stage1_seed{seed}.json")
    stage2_out = os.path.join(outdir, task, f"predictor-supervised_seed{seed}.json")

    # Check if Stage 2 already complete
    if os.path.exists(stage2_out):
        print(f"skip {stage2_out}: exists")
        return

    # ---- STAGE 1 ----
    cfg1 = Config(**cfg.to_dict())
    cfg1.stage = 1
    cfg1.n_steps = stage1_steps

    if verbose:
        print(f"\n=== predictor-supervised Stage 1 / {task} / seed {seed} "
              f"({stage1_steps} steps) ===")

    if not os.path.exists(stage1_out):
        # Always save Stage 1 checkpoint - Stage 2 needs it to load
        train_run("predictor-supervised", task, seed, cfg1, stage1_out,
                  ckpt=True, verbose=verbose)
    else:
        print(f"Stage 1 exists: {stage1_out}")

    # ---- STAGE 2 ----
    cfg2 = Config(**cfg.to_dict())
    cfg2.stage = 2
    cfg2.n_steps = stage2_steps
    cfg2.stage2_mode = "supervised_budget"

    if verbose:
        print(f"\n=== predictor-supervised Stage 2 / {task} / seed {seed} "
              f"({stage2_steps} steps, supervised + budget) ===")

    train_run("predictor-supervised", task, seed, cfg2, stage2_out,
              ckpt=not no_ckpt, verbose=verbose)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="tagged,mixed")
    p.add_argument("--methods", default="all")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--size", default="small", choices=["small", "medium"])
    p.add_argument("--steps", type=int, default=None,
                   help="Total steps for single-stage methods (default: 6000)")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--test-seqs", type=int, default=None)
    p.add_argument("--mod-capacity", default="auto")
    p.add_argument("--sparsity-lambda", type=float, default=None)
    p.add_argument("--chem-tag-init", action="store_true")
    p.add_argument("--chem-hidden", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--no-ckpt", action="store_true")
    # Two-stage predictor options (Option 2)
    p.add_argument("--stage1-steps", type=int, default=3000,
                   help="Steps for Stage 1 (full FFN) of predictor")
    p.add_argument("--stage2-steps", type=int, default=3000,
                   help="Steps for Stage 2 (DifficultyHead) of predictor")
    # Option 3 (predictor-v3) options
    p.add_argument("--predictor-target-steps", type=int, default=500,
                   help="Number of steps in Stage 1 to collect per-token CE targets")
    p.add_argument("--predictor-mse-weight", type=float, default=1.0,
                   help="MSE loss weight for Option 3 Stage 2")
    # Option 3b (predictor-supervised) options
    p.add_argument("--predictor-sup-stage1-steps", type=int, default=3000,
                   help="Steps for Stage 1 (full FFN) of predictor-supervised")
    p.add_argument("--predictor-sup-stage2-steps", type=int, default=3000,
                   help="Steps for Stage 2 (supervised DifficultyHead) of predictor-supervised")
    p.add_argument("--predictor-target-gate-e", type=float, default=0.2,
                   help="Target gate for Easy tokens")
    p.add_argument("--predictor-target-gate-m", type=float, default=0.5,
                   help="Target gate for Medium tokens")
    p.add_argument("--predictor-target-gate-h", type=float, default=0.8,
                   help="Target gate for Hard tokens")
    p.add_argument("--predictor-budget-target", type=float, default=0.5,
                   help="Target mean gate (budget)")
    p.add_argument("--predictor-budget-weight", type=float, default=1.0,
                   help="Budget loss weight for Option 3b")
    p.add_argument("--predictor-digit-targets", action="store_true",
                   help="Option B (mixed): difficulty targets from digit-count bins")

    args = p.parse_args()

    tasks = args.tasks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    for task in tasks:
        methods = TASK_METHODS[task] if args.methods == "all" else args.methods.split(",")
        for method in methods:
            if method not in TASK_METHODS[task]:
                print(f"skip {method}/{task}: not applicable")
                continue

            overrides = {}
            if args.steps is not None:
                overrides["n_steps"] = args.steps
            if args.eval_every is not None:
                overrides["eval_every"] = args.eval_every
            if args.batch_size is not None:
                overrides["batch_size"] = args.batch_size
            if args.seq_len is not None:
                overrides["seq_len"] = args.seq_len
            if args.test_seqs is not None:
                overrides["test_seqs"] = args.test_seqs
            if method == "mod":
                if args.mod_capacity == "auto":
                    overrides["mod_capacity"] = resolve_mod_capacity(task, args.outdir)
                else:
                    overrides["mod_capacity"] = float(args.mod_capacity)
            if args.sparsity_lambda is not None:
                overrides["sparsity_lambda"] = args.sparsity_lambda
            if args.chem_tag_init:
                overrides["chem_tag_init"] = True
            if args.chem_hidden:
                overrides["chem_hidden_input"] = True
            if args.predictor_digit_targets:
                overrides["predictor_digit_targets"] = True

            cfg = Config.for_size(args.size, **overrides)

            if method == "predictor":
                # Option 2: two-stage without supervision
                for seed in seeds:
                    run_predictor_two_stage(
                        task, seed, cfg, args.outdir,
                        args.stage1_steps, args.stage2_steps,
                        args.eval_every or 250, args.no_ckpt
                    )
            elif method == "predictor-v3":
                # Option 3: two-stage with supervised difficulty targets
                for seed in seeds:
                    run_predictor_v3_two_stage(
                        task, seed, cfg, args.outdir,
                        args.stage1_steps, args.stage2_steps,
                        args.predictor_target_steps, args.predictor_mse_weight,
                        args.eval_every or 250, args.no_ckpt
                    )
            elif method == "predictor-supervised":
                # Option 3b: two-stage with supervised difficulty + budget
                for seed in seeds:
                    run_predictor_supervised_two_stage(
                        task, seed, cfg, args.outdir,
                        args.predictor_sup_stage1_steps, args.predictor_sup_stage2_steps,
                        args.eval_every or 250, args.no_ckpt
                    )
            else:
                # Single-stage training
                for seed in seeds:
                    out_path = os.path.join(args.outdir, task, f"{method}_seed{seed}.json")
                    if os.path.exists(out_path):
                        print(f"skip {out_path}: exists")
                        continue
                    train_run(method, task, seed, cfg, out_path, ckpt=not args.no_ckpt)


if __name__ == "__main__":
    main()