# Chemical Transformer

This project trains a transformer model with an allostatic chemical state.
The model has about 11 million parameters.

## What it does

The chemical state is a per-token signal that flows from one layer to the next.
It uses attention entropy to drive the signal.
This signal controls a Mixture-of-Depths skip in the feed-forward network.
The result is that each token uses a different amount of compute.
The model reads a difficulty tag (`[E]`, `[M]`, or `[H]`) at the start of each problem.
It allocates more compute to tokens that need it.

The project compares three models:

- **Chemical**: The full model with the allostatic mechanism.
- **Chemical-off**: An ablation where the chemical pathway is frozen. The gate stays at 1.0 (full compute). This tests if any improvement comes from the mechanism itself or just from extra parameters.
- **Baseline**: A standard transformer. The FFN runs on every token. No modulation.

## How it works

The task is arithmetic with a difficulty tag at the start:

```
[E]3+5=8   [M]123-456=-333   [H]4291*318=1364538   ...
```

The training data uses three difficulty levels:

- Easy `[E]`: 1-digit addition and subtraction. About 162 unique problems. The model can memorize these.
- Medium `[M]`: 3-digit addition and subtraction. About 1.6 million unique problems.
- Hard `[H]`: 4-digit by 3-digit multiplication. About 9 million unique problems. The model cannot memorize these at 11 million parameters.

The model uses the difficulty tag to decide how much compute to use.
The test set is a frozen held-out set of 4000 problems per difficulty level.
The training run never sees these problems.
The output reports overall accuracy, balanced accuracy (macro over difficulties), and per-difficulty accuracy.

## How to install

```
uv pip install -e .
```

Or:

```
pip install -e .
```

## How to run

```
python main.py
```

A full run takes about 8 hours on Apple Silicon (MPS).
It produces `comparison.png` with 6 panels and a final comparison table.

For a fast test, set `CFG.n_steps = 3000` and `CFG.eval_every = 250`.

## What the panels show

1. Training cross-entropy loss.
2. Held-out overall accuracy.
3. Held-out balanced accuracy (macro average).
4. Per-difficulty accuracy. The Chemical model uses a solid line. The Baseline uses a dashed line.
5. FLOPs per token with a standard deviation band. The counter counts the operations that the model actually performs.
6. Mean FFN gate value by difficulty for the Chemical model. This shows if the model allocates more compute to hard tokens. If the gate stays flat, the chemical state does not find the signal.

## Notes

The code runs on Apple Silicon (MPS). It falls back to CUDA or CPU.
