# Day-one validation on H100 (~$0.5 total).
# 1. 11M legacy-path regression pair (dense accounting = 33.2 MFLOPs/tok)
python run_experiments.py --tasks tagged --methods baseline,chemical-off --seeds 0 \
  --steps 2000 --eval-every 1000 --outdir results-validate --no-ckpt --device cuda
# 2. New-stack smoke: RoPE+SDPA at 150m (exercises presets/rope/sdpa/bf16-off path)
python run_experiments.py --tasks tagged-v2 --methods baseline,fixed-schedule --seeds 0 \
  --size 150m --steps 50 --batch-size 32 --seq-len 512 --eval-every 25 \
  --outdir results-validate --no-ckpt --device cuda
# 3. New-stack with bf16 autocast + cosine schedule (the P1 configuration)
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 150m --steps 50 --batch-size 32 --seq-len 512 --eval-every 25 \
  --lr-schedule cosine --weight-decay 0.1 --amp \
  --outdir results-validate --no-ckpt --device cuda
# 4. P0 shuffled controls (all 3 seeds; Stage-1 ckpts staged; ~4 min each)
python -m ct.gategrad --seeds 0,1,2 --outdir results-p0 --arms shuffled --device cuda
echo VALIDATE_DONE
