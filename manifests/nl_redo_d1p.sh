# NL redo D1: webtext seed-1 chain (dense, rotation) + oracle + stage-2 arms
# (gategrad, shuffled, full-unfreeze, per-layer) on ONE consistent body.
# All phases gated; EXIT trap always marks DONE for the watchdog.
set -eo pipefail
trap 'touch /workspace/NLX_DONE' EXIT
cd /workspace/chemical-transformer
PY=/opt/conda/bin/python
$PY -c "import transformers, datasets, accelerate" 2>/dev/null || \
  /opt/conda/bin/pip install -q --no-input "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1 || \
  /opt/conda/bin/pip install -q --no-input -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1

# D1-1: corpus
if [ ! -f data/nl/webtext/meta.json ]; then
  $PY -m ct.nl.data --domain webtext --tokenizer EleutherAI/pythia-1.4b \
    --out-root data/nl --max-tokens 350000000 2>&1 | tee /workspace/phase_d1_1.log | tail -2 || true
fi
test -f data/nl/webtext/train_tokens.npy || { echo "D1-1 FAILED"; exit 1; }

# D1-2 (rotation first: headline piece lands early): rotation seed 1
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 1 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_d1_3.log | tail -3
test -f results-nl/webtext/rotation_seed1.json || { echo "D1-3 FAILED"; exit 1; }

# D1-3: dense seed 1 (fresh body; ckpt feeds every fork below)
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 1 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_d1_2.log | tail -3
test -f results-nl/webtext/baseline_seed1.json || { echo "D1-2 FAILED"; exit 1; }

# D1-4: oracle prepass (summed-score bins)
$PY -m ct.nl.oracle --domain webtext --model EleutherAI/pythia-1.4b \
  --ckpt results-nl/webtext/baseline_seed1.pt --data-root data/nl \
  --out-root results-nl --pool-seqs 20000 --batch-seqs 4 --device cuda \
  2>&1 | tee /workspace/phase_d1_4.log | tail -4
test -f data/nl/webtext/oracle.npz || { echo "D1-4 FAILED"; exit 1; }

# D1-5: ours-gategrad stage 2
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method ours-gategrad --stage 2 --seed 1 --steps 2000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_d1_5.log | tail -3
test -f results-nl/webtext/ours-gategrad_seed1.json || { echo "D1-5 FAILED"; exit 1; }

# D1-6: shuffled control stage 2
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method shuffled --stage 2 --seed 1 --steps 2000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_d1_6.log | tail -3
test -f results-nl/webtext/shuffled_seed1.json || { echo "D1-6 FAILED"; exit 1; }

echo "MANIFEST D1PRIME COMPLETE"
