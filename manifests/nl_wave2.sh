# NL wave 2: the two appendix ablations (full-unfreeze, per-layer targets)
# on the seed-1 body. Append this to the D1 instance's manifest after D1'.
set -eo pipefail
cd /workspace/chemical-transformer
PY=/opt/conda/bin/python

# W2-1: full-unfreeze ablation
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method ours-gategrad --stage 2 --seed 1 --steps 2000 --batch-seqs 4 --grad-accum 2 \
  --full-unfreeze --suffix=_fullunf \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_w1.log | tail -3
test -f results-nl/webtext/ours-gategrad_fullunf_seed1.json || { echo "W2-1 FAILED"; exit 1; }

# W2-2: per-layer oracle (same body, per-(token,layer) scores)
$PY -m ct.nl.oracle --domain webtext --model EleutherAI/pythia-1.4b \
  --ckpt results-nl/webtext/baseline_seed1.pt --data-root data/nl \
  --out-root results-nl --pool-seqs 20000 --batch-seqs 4 --device cuda --save-layers \
  2>&1 | tee /workspace/phase_w2.log | tail -4
test -f data/nl/webtext/oracle.npz || { echo "W2-2 FAILED"; exit 1; }

# W2-3: per-layer-target stage 2
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method ours-gategrad --stage 2 --seed 1 --steps 2000 --batch-seqs 4 --grad-accum 2 \
  --per-layer-targets --suffix=_perlay \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_w3.log | tail -3
test -f results-nl/webtext/ours-gategrad_perlay_seed1.json || { echo "W2-3 FAILED"; exit 1; }
echo "WAVE2 COMPLETE"
