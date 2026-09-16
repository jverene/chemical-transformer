# NL headline experiment: 1.4B dense vs rotation fine-tune on web-text.
# Phase 1: corpus (300M tokens) + deps
cd /workspace/chemical-transformer
/opt/conda/bin/pip install -q --no-input "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" 2>&1 | tail -1
/opt/conda/bin/python -m ct.nl.data --domain webtext --tokenizer EleutherAI/pythia-1.4b \
  --out-root data/nl --max-tokens 350000000 2>&1 | tail -2
# Phase 2: dense arm (overnight)
/opt/conda/bin/python -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method baseline --stage 1 --seed 0 --steps 3000 --batch-seqs 8 --device cuda \
  --data-root data/nl --out-root results-nl 2>&1 | tail -3
touch /workspace/DENSE_DONE
# Phase 3: rotation arm (tomorrow)
/opt/conda/bin/python -m ct.nl.train_lm --domain webtext --model EleutherAI/pythia-1.4b \
  --method rotation --stage 1 --seed 0 --steps 3000 --batch-seqs 8 --device cuda \
  --data-root data/nl --out-root results-nl 2>&1 | tail -3
touch /workspace/NL_REAL_DONE
