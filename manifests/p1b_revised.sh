# Revised P1 (descoped to budget): no 1b probe (1b P3 unaffordable at $2.67/hr
# anyway); batch 64 for MFU; trimmed gategrad stages. ~3.5-4h, ~$10.
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 400m --steps 5000 --batch-size 64 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size medium --seeds 0 \
  --stage1-steps 4000 --stage2-steps 1500 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 150m --seeds 0 \
  --stage1-steps 4000 --stage2-steps 1500 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 400m --seeds 0 \
  --stage1-steps 3000 --stage2-steps 1000 --outdir results-p1 --device cuda
echo P1B_DONE
touch /workspace/P1B_DONE
