# D: 150m legacy on ORIGINAL tagged (task vs optimization discriminator)
python run_experiments.py --tasks tagged --methods baseline --seeds 0 \
  --size 150m --steps 300 --batch-size 32 --seq-len 512 --eval-every 100 \
  --pos-type learned --attn-impl manual \
  --outdir results-diagD --no-ckpt --device cuda
# E: 150m legacy + tagged-v2 at seq 256 (seq-512 discriminator)
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 150m --steps 300 --batch-size 32 --seq-len 256 --eval-every 100 \
  --pos-type learned --attn-impl manual \
  --outdir results-diagE --no-ckpt --device cuda
# G: 150m legacy + tagged-v2 + lr 1e-4 (LR discriminator)
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 150m --steps 300 --batch-size 32 --seq-len 512 --eval-every 100 \
  --pos-type learned --attn-impl manual --lr 1e-4 \
  --outdir results-diagG --no-ckpt --device cuda
# H: 42m legacy + tagged-v2 (scale discriminator at the hot LR)
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size medium --steps 300 --batch-size 32 --seq-len 512 --eval-every 100 \
  --pos-type learned --attn-impl manual \
  --outdir results-diagH --no-ckpt --device cuda
echo DIAG2_DONE
