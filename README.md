# chemical-transformer

~11M-parameter transformer with an **allostatic chemical state**: a per-token,
layer-to-layer signal (driven by attention entropy) that gates the FFN via a
Mixture-of-Depths skip, so per-token compute is genuinely input-dependent. The
thesis is that the model can read a `[E]/[M]/[H]` difficulty tag early in each
problem and pre-allocate more FFN compute to the upcoming hard answer tokens.

Three models are trained and compared:

- **Chemical** — the full allostatic MoD model.
- **Chemical-off** — ablation: same architecture with the chemical/router pathway
  frozen and the gate forced to 1.0 (full compute). Isolates whether any delta
  comes from the chemical mechanism vs. just having more parameters.
- **Baseline** — standard transformer, full FFN on every token, no modulation.

## The task (and why it's built this way)

Arithmetic with a leading difficulty tag, packed many problems per sequence:

```
[E]3+5=8   [M]123-456=-333   [H]4291*318=1364538   ...
```

- Easy `[E]` 1-digit ± → ~162 unique problems (memorizable).
- Medium `[M]` 3-digit ± → ~1.6M unique.
- Hard `[H]` 4-digit × 3-digit → ~9M unique → **not memorizable** at 11M params.

The difficulty tag is the early signal the chemical state is supposed to use.
Accuracy is measured on a **frozen held-out set** (4000 problems/difficulty,
sampled once at startup; training never sees them), reported as overall,
balanced (macro over difficulties), and per-difficulty.

## Run

```bash
uv pip install -e .            # or: pip install -e .
python main.py                 # ~8h on MPS for 3 models × 10k steps
```

Outputs `comparison.png` (6 panels) and a final comparison table. For a fast
sanity pass, set `CFG.n_steps = 3000`, `CFG.eval_every = 250`.

## What the 6 panels show

1. Train cross-entropy
2. Held-out overall accuracy
3. Held-out balanced accuracy (macro)
4. Per-difficulty accuracy (Chemical solid, Baseline dashed)
5. FLOPs/token with ±std band (dynamic counter: counts ops actually performed)
6. Mean FFN gate by difficulty for the Chemical model — **the thesis readout:
   Hard > Easy means the chemical state is allocating more compute to hard tokens.**

## Notes

- Optimized for Apple Silicon (MPS); falls back to CUDA/CPU.
- The allostatic thesis lives or dies on panel 6. If the gate stays flat across
  difficulties through a full run, the honest conclusion is the chemical state
  isn't finding the signal — not that the experiment is broken.
