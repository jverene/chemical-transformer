# NL extra A: webtext seed 1 (dense + rotation) + oracle prepass + stage-2
# ours-gategrad/shuffled on the seed-1 dense body. Gated phases; the EXIT
# trap always touches the DONE marker so the watchdog stops the instance.
set -eo pipefail
trap 'touch /workspace/NLX_DONE' EXIT
cd /workspace/chemical-transformer
PY=/opt/conda/bin/python
$PY -c "import transformers, datasets, accelerate" 2>/dev/null || \
  /opt/conda/bin/pip install -q --no-input "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1

# Phase A1: webtext corpus (350M tokens)
if [ ! -f data/nl/webtext/meta.json ]; then
  $PY -m ct.nl.data --domain webtext --tokenizer EleutherAI/pythia-1.4b \
    --out-root data/nl --max-tokens 350000000 2>&1 | tee /workspace/phase_a1.log | tail -2
fi
test -f data/nl/webtext/train_tokens.npy || { echo "A1 FAILED: no corpus"; exit 1; }

# Phase A2: dense seed 1
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 1 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_a2.log | tail -3
test -f results-nl/webtext/baseline_seed1.json || { echo "A2 FAILED"; exit 1; }

# Phase A3: rotation seed 1
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 1 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_a3.log | tail -3
test -f results-nl/webtext/rotation_seed1.json || { echo "A3 FAILED"; exit 1; }

# Phase A4: oracle prepass on the seed-1 dense body (score/CE per token)
$PY -m ct.nl.oracle --domain webtext --model EleutherAI/pythia-1.4b \
  --ckpt results-nl/webtext/baseline_seed1.pt --data-root data/nl \
  --out-root results-nl --pool-seqs 20000 --batch-seqs 4 --device cuda \
  2>&1 | tee /workspace/phase_a4.log | tail -6
test -f data/nl/webtext/oracle.npz || { echo "A4 FAILED: no oracle.npz"; exit 1; }

# Phase A5: stage-2 ours-gategrad seed 1 (body near-frozen, heads trained)
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method ours-gategrad --stage 2 --seed 1 --steps 2000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_a5.log | tail -3
test -f results-nl/webtext/ours-gategrad_seed1.json || { echo "A5 FAILED"; exit 1; }

# Phase A6: stage-2 shuffled control seed 1
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method shuffled --stage 2 --seed 1 --steps 2000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase_a6.log | tail -3
test -f results-nl/webtext/shuffled_seed1.json || { echo "A6 FAILED"; exit 1; }
echo "MANIFEST A COMPLETE"
