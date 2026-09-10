# P1 v2: per-size preset LRs (150m=1e-4, 400m=6e-5 — Sep 6 instability fix).
# 150m window probe: 5000 x 16k = 82M tokens
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 150m --steps 5000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --outdir results-p1 --device cuda
# 400m window probe: 4000 x 32k = 131M tokens
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 400m --steps 4000 --batch-size 64 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p1 --device cuda
# value-migration probes (pre-registered readout; per-size preset LR)
python -m ct.gategrad --task tagged-v2 --size medium --seeds 0 \
  --stage1-steps 4000 --stage2-steps 1500 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 150m --seeds 0 \
  --stage1-steps 4000 --stage2-steps 1500 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 400m --seeds 0 \
  --stage1-steps 2500 --stage2-steps 800 --outdir results-p1 --device cuda
touch /workspace/P1B_DONE
