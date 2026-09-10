# 400m rescue: LR sweep -> window probe -> migration probe, at the winning LR.
for LR in 8e-5 1.2e-4 1.6e-4 2e-4; do
  python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
    --size 400m --steps 400 --batch-size 32 --seq-len 512 --eval-every 200 \
    --lr-schedule cosine --weight-decay 0.1 --lr $LR --amp \
    --outdir results-sweep-$LR --no-ckpt --device cuda
done
LR=$(python3 - <<'PY'
import json, glob
best = ("1.5e-4", 1e9)
for d in glob.glob("results-sweep-*/tagged-v2/baseline_seed0.json"):
    r = json.load(open(d))
    L = r["final"]["loss"]
    if L < best[1]:
        best = (str(r["config"]["lr"]), L)
print(best[0])
PY
)
echo "SWEEP: chosen lr = $LR"
python run_experiments.py --tasks tagged-v2 --methods baseline --seeds 0 \
  --size 400m --steps 4000 --batch-size 32 --seq-len 512 --eval-every 250 \
  --lr-schedule cosine --weight-decay 0.1 --lr $LR --amp \
  --outdir results-p1 --device cuda
python -m ct.gategrad --task tagged-v2 --size 400m --seeds 0 \
  --stage1-steps 2500 --stage2-steps 1000 --lr $LR --outdir results-p1 --device cuda
touch /workspace/P1B_DONE
