# NL extra B: code (CodeSearchNet python) + math (MetaMathQA) domain pairs —
# dense vs rotation at seed 0. The code dataset id moved on the Hub; the
# builder now uses the live mirror code-search-net/code_search_net.
set -eo pipefail
trap 'touch /workspace/NLX_DONE' EXIT
cd /workspace/chemical-transformer
PY=/opt/conda/bin/python
$PY -c "import transformers, datasets, accelerate" 2>/dev/null || \
  /opt/conda/bin/pip install -q --no-input "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1

# Phase B1: code corpus (builder stops early if the source exhausts)
if [ ! -f data/nl/code/meta.json ]; then
  $PY -m ct.nl.data --domain code --tokenizer EleutherAI/pythia-1.4b \
    --out-root data/nl --max-tokens 350000000 2>&1 | tee /workspace/phase_b1.log | tail -2
fi
test -f data/nl/code/train_tokens.npy || { echo "B1 FAILED: no code corpus"; exit 1; }

# Phase B2: code dense seed 0
$PY -m ct.nl.train_lm --domain code --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 0 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_b2.log | tail -3
test -f results-nl/code/baseline_seed0.json || { echo "B2 FAILED"; exit 1; }

# Phase B3: code rotation seed 0
$PY -m ct.nl.train_lm --domain code --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 0 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_b3.log | tail -3
test -f results-nl/code/rotation_seed0.json || { echo "B3 FAILED"; exit 1; }

# Phase B4: math corpus
if [ ! -f data/nl/math/meta.json ]; then
  $PY -m ct.nl.data --domain math --tokenizer EleutherAI/pythia-1.4b \
    --out-root data/nl --max-tokens 350000000 2>&1 | tee /workspace/phase_b4.log | tail -2
fi
test -f data/nl/math/train_tokens.npy || { echo "B4 FAILED: no math corpus"; exit 1; }

# Phase B5: math dense seed 0
$PY -m ct.nl.train_lm --domain math --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 0 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_b5.log | tail -3
test -f results-nl/math/baseline_seed0.json || { echo "B5 FAILED"; exit 1; }

# Phase B6: math rotation seed 0
$PY -m ct.nl.train_lm --domain math --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 0 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_b6.log | tail -3
test -f results-nl/math/rotation_seed0.json || { echo "B6 FAILED"; exit 1; }
echo "MANIFEST B COMPLETE"
