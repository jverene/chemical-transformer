# P2: controls + ablations at 150m on tagged-v2 (~$30)
python run_experiments.py --tasks tagged-v2 --methods baseline,chemical-off,fixed-schedule,mod,random-gate,tag-only \
  --seeds 0,1,2 --size 150m --steps 10000 --batch-size 32 --seq-len 512 --eval-every 500 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p2 --device cuda
python run_experiments.py --tasks tagged-v2 --methods predictor-supervised \
  --seeds 0,1,2 --size 150m --steps 10000 --batch-size 32 --seq-len 512 --eval-every 500 \
  --lr-schedule cosine --weight-decay 0.1 --amp --outdir results-p2 --device cuda
# full-unfreeze Stage-2 ablation (both groups at full LR)
python - <<'PYEOF'
from ct.config import Config
from ct.train import train_run
for s in (0, 1, 2):
    cfg = Config.for_size("150m", device="cuda", eval_every=500, stage=2, n_steps=5000,
                          stage2_mode="supervised_budget", weight_decay=0.1,
                          lr_schedule="cosine", amp=True, batch_size=32, seq_len=512,
                          predictor_freeze_body=True)
    cfg.lr = 3e-4
    train_run("predictor-supervised", "tagged-v2", s, cfg,
              f"results-p2/tagged-v2/ours-fullunfreeze_seed{s}.json",
              stage1_ckpt=f"results-p2/tagged-v2/predictor-supervised_stage1_seed{s}.pt")
PYEOF
