# Isolate the 150m failure: A legacy stack / B rope+sdpa / C 42m rope+sdpa
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 150m --steps 300 --batch-size 32 --seq-len 512 --eval-every 100 \
  --pos-type learned --attn-impl manual \
  --outdir results-diagA --no-ckpt --device cuda
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 150m --steps 300 --batch-size 32 --seq-len 512 --eval-every 100 \
  --outdir results-diagB --no-ckpt --device cuda
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size medium --steps 300 --batch-size 32 --seq-len 512 --eval-every 100 \
  --outdir results-diagC --no-ckpt --device cuda
echo DIAG_DONE
