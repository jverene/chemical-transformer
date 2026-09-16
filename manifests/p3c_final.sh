# Final pieces: MoD x2 (complete the 150m table) + the starved-regime
# positive test (online vs shuffled vs static, where the field is alive).
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
touch /workspace/P3C_DONE
