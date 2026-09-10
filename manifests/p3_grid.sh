# P3: merged 150m grid on tagged-v2 (82M tokens/run, batch 64, seq 512,
# lr 1e-4 cosine + wd 0.1 + amp — the validated recipe, identical across methods).
# Ours first: shared Stage-1 per seed -> tag targets + shuffled control forks.
python -m ct.gategrad --task tagged-v2 --size 150m --seeds 0,1,2 \
  --stage1-steps 2000 --stage2-steps 750 --batch-size 64 --seq-len 512 \
  --arms stage1,tag,shuffled --outdir results-p3 --device cuda
python run_experiments.py --tasks tagged-v2 --methods baseline,fixed-schedule,mod --seeds 0,1,2 \
  --size 150m --steps 2500 --batch-size 64 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p3 --device cuda --no-ckpt
touch /workspace/P3_DONE
