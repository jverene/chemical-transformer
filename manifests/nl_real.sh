# NL headline experiment: 1.4B dense vs rotation fine-tune on web-text.
# Gated phases: no phase proceeds unless the previous phase's output exists.
# The EXIT trap ALWAYS touches NL_REAL_DONE (success or failure) so the
# instance never idles after the manifest ends — the autopilot/watchdog stops
# on that marker, and the failure reason is in /workspace/nl*.log.
set -eo pipefail
trap 'touch /workspace/NL_REAL_DONE' EXIT
cd /workspace/chemical-transformer
PY=/opt/conda/bin/python
$PY -c "import transformers, datasets, accelerate" 2>/dev/null || \
  /opt/conda/bin/pip install -q --no-input "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1 || \
  /opt/conda/bin/pip install -q --no-input -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1
$PY -c "import transformers, datasets, accelerate" || { echo "PHASE0 FAILED: deps missing"; exit 1; }

# Phase 0: 30-step smoke at 160m on a 2M-token corpus — kills the 10h run
# early if precision or data issues produce nonfinite loss on this GPU
# (fp32 CE + skip guard now in the driver; abort exit code 3).
if [ ! -f data/nl-smoke/webtext/meta.json ]; then
  $PY -m ct.nl.data --domain webtext --tokenizer EleutherAI/pythia-160m \
    --out-root data/nl-smoke --max-tokens 2000000 2>&1 | tee /workspace/smoke_data.log | tail -1
fi
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-160m \
  --method rotation --stage 1 --seed 0 --steps 30 --batch-seqs 2 --grad-accum 2 \
  --eval-every 25 --device cuda --data-root data/nl-smoke --out-root results-nl-smoke \
  2>&1 | tee /workspace/smoke.log | tail -2
test -f results-nl-smoke/webtext/rotation_seed0.json || { echo "PHASE0 FAILED: smoke"; exit 1; }
grep -q '"skipped_nonfinite_steps": 0' results-nl-smoke/webtext/rotation_seed0.json || \
  grep -q '"skipped_nonfinite_steps":0' results-nl-smoke/webtext/rotation_seed0.json || \
  { echo "PHASE0 FAILED: smoke had nonfinite steps"; exit 1; }

# Phase 1: corpus (350M tokens)
if [ ! -f data/nl/webtext/meta.json ]; then
  $PY -m ct.nl.data --domain webtext --tokenizer EleutherAI/pythia-1.4b \
    --out-root data/nl --max-tokens 350000000 2>&1 | tee /workspace/phase1.log | tail -2
fi
test -f data/nl/webtext/train_tokens.npy || { echo "PHASE1 FAILED: no corpus"; exit 1; }

# Phase 2: dense arm (effective batch 8 = 4 seqs x grad-accum 2)
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 0 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase2.log | tail -3
test -f results-nl/webtext/baseline_seed0.json || { echo "PHASE2 FAILED"; exit 1; }
touch /workspace/DENSE_DONE

# Phase 3: rotation arm
$PY -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 0 --steps 3000 --batch-seqs 4 --grad-accum 2 \
  --eval-every 250 --device cuda --data-root data/nl --out-root results-nl \
  2>&1 | tee /workspace/phase3.log | tail -3
test -f results-nl/webtext/rotation_seed0.json || { echo "PHASE3 FAILED"; exit 1; }
