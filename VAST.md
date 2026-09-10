# GPU Runbook — vast.ai A100 (ICLR expansion)

Local Mac/MPS is only for correctness smokes at 11M (P0). Everything below is
the paper data, run on rented GPUs. The playbook is the proven one from
`~/Developer/neuromodulation/VAST.md`; only the plumbing differs (torch, not
JAX). The launcher does stage-code-rsync → env → tmux → self-stop watchdog →
periodic result pull. It **never destroys** an instance — stop only, so
`/workspace` data survives.

## Cost model (plan §2 of the ICLR plan)

Training run ≈ `6 · N · D` FLOPs. A100 bf16 at ~40% MFU ≈ 1.2e14 FLOP/s;
A100 on-demand ≈ $0.7–1.9/hr. So a 1B model on 0.5B tokens ≈ 3e18 FLOPs ≈
7–14 h ≈ **$7–14**; 11M regression run ≈ minutes; fine-tune forks (Stage-2
only) ≈ cents-to-dollars. Stage gates from the plan:

- **No P3 spend** until P1 probes confirm the size window and $/run.
- **No P6** until P4 lands (Sep 19 tripwire).
- Cuts, in order: P3→400m fallback, P5b, P6. **Never cut P4/P5a.**

## Setup (once, on the laptop)

```bash
pip install vastai
vastai set api-key <KEY>        # https://cloud.vast.ai/account/cli/
export VAST_API_KEY=<KEY>       # the launcher/watchdog read this
```

If every command fails with "Session expired", pass the key explicitly:
`vastai --api-key "$VAST_API_KEY" tfa send-email`, then `tfa login` with the
emailed code (learned-the-hard-way note from the neuromodulation runbook).

## Rent

```bash
# ≥40GB VRAM, reliable host, best perf/$; add gpu_ram >= 76000 for 80GB
vastai search offers 'gpu_ram >= 40000 reliability > 0.95 rentable = true' -o 'dlperf_usd-'
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  --disk 80 --ssh --direct --onstart-cmd "nvidia-smi"
```

Prefer **verified non-interruptible** offers for the 1B/1.4B runs — a
preemption mid-run loses everything since the last checkpoint. H100 offers at
≤2× A100 price are worth it for the 1B stage (half the wall-clock). Set a
**spend alert** in the dashboard; the watchdog is a backstop, not the budget.

## Launch a stage

Write a manifest (one command per line), then:

```bash
scripts/vast_launch.sh <INSTANCE_ID> manifests/p1_probes.sh 24 15
```

- `manifests/` holds one manifest per stage (P1 probes, P2 controls, P3 …).
- The launcher rsyncs the working tree (code only), installs scipy/matplotlib
  into the image's python if missing, runs the manifest under `tmux`, arms
  `scripts/vast_watchdog.sh` (stops at MAX_HOURS, dollar cap via dashboard,
  or manifest completion), and pulls `results/` → `./results_vast/` every
  10 min.
- Results land in `results_vast/`; move them into the canonical `results-*`
  dirs locally (or point analysis at both).

## Day-one $2 validation (before any real spend)

```bash
echo "python run_experiments.py --tasks tagged --methods baseline,fixed-schedule \
  --seeds 0 --steps 200 --outdir results-smoke --no-ckpt --device cuda" > manifests/validate.sh
scripts/vast_launch.sh <ID> manifests/validate.sh 2 5
```

Gate: runs complete, `results_vast/tagged/*.json` show device `cuda`, billed
FLOPs logged. Then the P1 probe manifest goes in.

## After a stage

```bash
vastai stop instance <ID>      # billing → storage-only; /workspace kept
vastai destroy instance <ID>   # ONLY after copying everything out
```

`rsync` again or `vastai copy` before destroying. Checkpoints: save `.pt`
state dicts only for Stage-1 bodies you will fork from (bf16-cast them if
3B); never `git push` weights.

## Interrupted run

`run_experiments.py` and the manifests are resumable — rerun the same
manifest; runs whose JSON exists are skipped. Stage-1 `.pt` checkpoints
survive on `/workspace` across stop/start of the same instance.
