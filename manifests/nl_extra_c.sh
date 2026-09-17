# NL extra C: wall-clock realization bench (real skip-FFN training) +
# webtext seed 2 (dense + rotation) for a 3-seed headline.
set -eo pipefail
trap 'touch /workspace/NLX_DONE' EXIT
cd /workspace/chemical-transformer
PY=/opt/conda/bin/python
$PY -c "import transformers, datasets, accelerate" 2>/dev/null || \
  /opt/conda/bin/pip install -q --no-input "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1

# Phase C0: wall-clock bench (no corpus needed; dense vs masked vs skip)
$PY -m ct.nl.bench_wallclock --model EleutherAI/pythia-1.4b \
  --batch-seqs 4 --seq-len 2048 --device cuda --out results-nl/wallclock.json \
  2>&1 | tee /workspace/phase_c0.log | tail -7
test -f results-nl/wallclock.json || { echo "C0 FAILED"; exit 1; }

# Phase C1: webtext corpus
if [ ! -f data/nl/webtext/meta.json ]; then
  $PY -m ct.nl.data --domain webtext --tokenizer EleutherAI/pythia-1.4b \
    --out-root data/nl --max-tokens 350000000 2>&1 | tee /workspace/phase_c1.log | tail -2
fi
test -f data/nl/webtext/train_tokens.npy || { echo "C1 FAILED: no corpus"; exit 1; }

# Phase C2: dense seed 2
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 2 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_c2.log | tail -3
test -f results-nl/webtext/baseline_seed2.json || { echo "C2 FAILED"; exit 1; }

# Phase C3: rotation seed 2
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 2 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_c3.log | tail -3
test -f results-nl/webtext/rotation_seed2.json || { echo "C3 FAILED"; exit 1; }
echo "MANIFEST C COMPLETE"
