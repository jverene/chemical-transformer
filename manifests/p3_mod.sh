# P3 follow-up: MoD x3 with the scatter dtype fix (restaged code).
python run_experiments.py --tasks tagged-v2 --methods mod --seeds 0,1,2 \
  --size 150m --steps 2500 --batch-size 64 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p3 --device cuda --no-ckpt
touch /workspace/P3_MOD_DONE
