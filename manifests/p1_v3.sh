# P1 v3: rerun of the migration probes with the validated recipe (cosine +
# warmup + wd via preset/mk fix), plus the 400m window probe at batch 32.
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 400m --steps 4000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size medium --seeds 0 \
  --stage1-steps 4000 --stage2-steps 1500 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 150m --seeds 0 \
  --stage1-steps 4000 --stage2-steps 1500 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 400m --seeds 0 \
  --stage1-steps 2500 --stage2-steps 1000 --outdir results-p1 --device cuda
touch /workspace/P1B_DONE
