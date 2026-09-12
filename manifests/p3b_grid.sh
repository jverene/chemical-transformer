# P3b: probe-recipe 150m grid (batch 32, 5000 steps = 82M tokens, fp32 —
# matches the validated window probe exactly). Seeds 0,1; seed 2 post-top-up.
# Ours first: shared Stage-1 -> tag + shuffled + fixed2stage (the 2x2 closer).
python -m ct.gategrad --task tagged-v2 --size 150m --seeds 0,1 \
  --stage1-steps 4000 --stage2-steps 1000 --batch-size 32 --seq-len 512 \
  --arms stage1,tag,shuffled,fixed2stage --outdir results-p3b --device cuda
python run_experiments.py --tasks tagged-v2 --methods baseline,fixed-schedule,mod --seeds 0,1 \
  --size 150m --steps 5000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --outdir results-p3b --device cuda --no-ckpt
# discriminator: ordering gap on the fresh starved-150m bodies
for S in 0 1; do
  python -m ct.headroom --ckpt results-p3b/tagged-v2/predictor-supervised_stage1_150m_seed$S.pt \
    --batches 6 --device cuda > results-p3b/headroom_150m_seed$S.json
done
touch /workspace/P3B_DONE
# THE POSITIVE PREDICTION TEST: the value field is structured in the starved
# regime (headroom discriminator, both seeds) -> online self-measuring
# allocation should now BEAT shuffled and static here. n=1 seed, ~25 min/arm.
python -m ct.singletest --arm online --seed 0 --steps 2500 --size 150m \
  --batch-size 32 --seq-len 512 --device cuda \
  --out results-p3b/single_online_150m_seed0.json
python -m ct.singletest --arm shuffled --seed 0 --steps 2500 --size 150m \
  --batch-size 32 --seq-len 512 --device cuda \
  --out results-p3b/single_shuffled_150m_seed0.json
python -m ct.singletest --arm static --seed 0 --steps 2500 --size 150m \
  --batch-size 32 --seq-len 512 --device cuda \
  --out results-p3b/single_static_150m_seed0.json
