# P1: calibration probes on tagged-v2 (~$12 total).
# P1a: difficulty-window curve probes — acc-vs-tokens per size (eval_every 250).
#   Gate: largest size with hard-bin accuracy inside [40%, 85%].
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 150m --steps 10000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --outdir results-p1 --device cuda
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 400m --steps 10000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p1 --device cuda
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 1b --steps 10000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p1 --device cuda

# P1b (PRE-REGISTERED READOUT): compute-value migration across 42M/150M/400M.
#   Prediction (DECISIONS.md): as scale grows, reducible mass migrates toward
#   harder bins — per-bin mean gate-grad score rises and frac_nonpos on H
#   falls toward M's. Crossover = the "allocation tracks capability" figure.
#   If H's frac_nonpos stays pinned high: irreducibility story in trouble;
#   know at ~$3, not at P3.
python -m ct.gategrad --task tagged-v2 --size medium --seeds 0 \
  --stage1-steps 6000 --stage2-steps 2000 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 150m --seeds 0 \
  --stage1-steps 6000 --stage2-steps 2000 --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 400m --seeds 0 \
  --stage1-steps 6000 --stage2-steps 2000 --outdir results-p1 --device cuda
