# Rerun: MoD x2 (table completion) + the starved-150m positive test.
python run_experiments.py --tasks tagged-v2 --methods mod --seeds 0,1 \
  --size 150m --steps 5000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --outdir results-p3b --device cuda --no-ckpt
python -m ct.singletest --arm online --seed 0 --steps 2500 --size 150m \
  --batch-size 32 --seq-len 512 --device cuda \
  --out results-p3b/single_online_150m_seed0.json
python -m ct.singletest --arm shuffled --seed 0 --steps 2500 --size 150m \
  --batch-size 32 --seq-len 512 --device cuda \
  --out results-p3b/single_shuffled_150m_seed0.json
python -m ct.singletest --arm static --seed 0 --steps 2500 --size 150m \
  --batch-size 32 --seq-len 512 --device cuda \
  --out results-p3b/single_static_150m_seed0.json
touch /workspace/P3C2_DONE
# SELF-STOP IMMEDIATELY AT COMPLETION (never rely on the external watchdog:
# its credential can expire during long runs — the $15-20 idle-burn lesson).
/opt/conda/bin/vastai set api-key $(cat /workspace/vast_tfa_key) >/dev/null 2>&1 || true
/opt/conda/bin/vastai --api-key "$(cat /workspace/vast_tfa_key)" stop instance 50958705 \
  >> /workspace/self_stop.log 2>&1 || echo "stop failed — console-stop manually" >> /workspace/self_stop.log
