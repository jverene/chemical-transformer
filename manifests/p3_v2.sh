# P3 v2 (fresh instance): single-stage runs FIRST (never completed on the
# crashed host), ours forks LAST (their completed twins may be rescuable
# from the old instance's disk later).
python run_experiments.py --tasks tagged-v2 --methods baseline,fixed-schedule,mod --seeds 0,1,2 \
  --size 150m --steps 2500 --batch-size 64 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p3 --device cuda --no-ckpt
python -m ct.gategrad --task tagged-v2 --size 150m --seeds 0,1,2 \
  --stage1-steps 2000 --stage2-steps 750 --batch-size 64 --seq-len 512 \
  --arms stage1,tag,shuffled --outdir results-p3 --device cuda
touch /workspace/P3_DONE
