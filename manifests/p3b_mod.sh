# Complete the 150m probe-recipe table: MoD x2 (dtype fix applied).
python run_experiments.py --tasks tagged-v2 --methods mod --seeds 0,1 \
  --size 150m --steps 5000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --outdir results-p3b --device cuda --no-ckpt
touch /workspace/P3MOD_DONE
