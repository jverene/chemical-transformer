# NL redo D2: webtext seed-2 pair (dense + rotation) for the 3-seed headline.
set -eo pipefail
trap 'touch /workspace/NLX_DONE' EXIT
cd /workspace/chemical-transformer
PY=/opt/conda/bin/python
$PY -c "import transformers, datasets, accelerate" 2>/dev/null || \
  /opt/conda/bin/pip install -q --no-input "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1 || \
  /opt/conda/bin/pip install -q --no-input -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1

# D2-1: corpus
if [ ! -f data/nl/webtext/meta.json ]; then
  $PY -m ct.nl.data --domain webtext --tokenizer EleutherAI/pythia-1.4b \
    --out-root data/nl --max-tokens 350000000 2>&1 | tee /workspace/phase_d2_1.log | tail -2
fi
test -f data/nl/webtext/train_tokens.npy || { echo "D2-1 FAILED"; exit 1; }

# D2-2: dense seed 2
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 2 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_d2_2.log | tail -3
test -f results-nl/webtext/baseline_seed2.json || { echo "D2-2 FAILED"; exit 1; }

# D2-3: rotation seed 2
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 2 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_d2_3.log | tail -3
test -f results-nl/webtext/rotation_seed2.json || { echo "D2-3 FAILED"; exit 1; }
echo "MANIFEST D2 COMPLETE"
