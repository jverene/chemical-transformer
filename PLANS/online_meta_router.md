# Plan: learned online meta-router (post-review big swing)

Greenlit with the three modifications from the review: differentiable functional AdamW specified correctly, multi-batch ordinary-stream rewards (or accumulated meta-gradients), shared mid-training checkpoint start instead of initialization. This file adds what the review missed and fixes the build order.

## Additions to the review

### A1. Second-order through AdamW's second moment is the actual mine, not just "potentially unstable"

The hypergradient flows through v̂_t = v_t/(1−β2^t). Early in any window, v_t entries near zero turn the Adam normalization into a huge amplifier on hypergradients. Two consequences:

- Implement TWO router-gradient modes behind a flag from day one:
  - **first-order**: gradient flows through the body-loss ∇_θ L and the parameter update, but the Adam normalization terms (m̂, v̂) are treated as constants (stop-grad). This is DARTS/first-order-MAML style. Expected workhorse.
  - **full second-order**: everything differentiable. Only trusted after the finite-difference smoke test, and only with a v_t floor.
- Add a small ε floor and a warmup rule: router receives no meta-gradient for the first W steps after checkpoint load (v̂ settles). W fixed, preregistered in the run manifest, no sweeping.

### A2. Checkpoint equivalence is stricter than "properly initialized equivalent optimizer states"

To start every arm from the same mid-training checkpoint you must restore, identically per arm:
- AdamW m, v state tensors AND the step counter t (bias corrections 1−β1^t, 1−β2^t depend on it),
- weight decay as decoupled (AdamW) not L2 — and the decoupled decay term must be INSIDE the differentiable update, since the router gradient should see that decay shrinks parameters,
- dropout/RNG streams seeded identically, and the data stream position so all arms see the same batch order.

A single assertion test: load checkpoint, run one ordinary (non-meta) step in each arm's fresh process, assert bitwise-identical parameters. If that fails, nothing downstream is interpretable.

### A3. The STE for hard top-B is biased in a direction that flatters the router

Straight-through passes the same gradient whether the gate was barely on or barely off, so the router never learns margin. Two cheap mitigations: (a) log the gate-value histogram near the B-threshold; if mass piles at the boundary, the STE is hiding indecision; (b) one ablation arm with annealed softmax gates (soft execution) to check the hard-router result isn't an STE artifact. The random and shuffled arms partially cover this, but only the soft arm separates "allocation signal" from "STE bias."

### A4. The reward window is off-policy in a specific way; document it

M upcoming ordinary batches are evaluated BEFORE the body trains on them (reward measured pre-consumption), then trained on. Identical across arms, so the comparison is clean, but the reward is "loss on batches the body has not yet seen," which is exactly what we want, and it should be stated that way in the paper. Also: count the reward-evaluation forward passes (M forward passes, no backward if reward uses no-grad for the loss reduction — only the meta-backward needs grad) in the cost ledger from the first smoke run, not retrofitted.

## Arms (hero run)

From one shared mid-training checkpoint, fixed K, fixed M, fixed dev trajectory, no sweeping until green:
1. learned online meta-router (hard, STE)
2. random allocation
3. shuffled-router allocation
4. per-layer first-order (if compute permits)
5. dense
6. dense-fewer-steps (cost-matched)

Success ladder unchanged: Level 2 = beats random+shuffled on-policy with fresh confirmation seeds. Level 3 = beats dense-fewer-steps at matched total cost including meta-overhead. Level 2 without Level 3 is still a result; write it as such.

## Build order (each gated by a test)

1. `functional_adamw.py`: pure-function AdamW step (params, grads, m, v, t) → new tensors. Test: bitwise/1e-7 match against torch.optim.AdamW over 50 steps, including weight decay and bias correction.
2. Persistence + detach test: after meta-step, only intended tensors changed; graph freed (peak memory flat across 20 consecutive meta-steps).
3. Tiny soft-gate model (no top-B, soft gates): hypergradient vs central finite differences, rel err < 1e-3. This validates first-order AND second-order modes.
4. Hard execution: gather/scatter == binary masking in outputs and grads; exact budget B enforced.
5. Checkpoint-equivalence test (A2).
6. Smoke trajectory (~500 steps, tiny config): learned vs random arm, confirm nonzero router gradient, finite loss, no drift in v̂.
7. Hero run.

## Risk register (unmoved by this plan)

- Reward noise remains the top scientific risk. If single-batch meta-gradients are unusable even at M=4, the fallback is gradient accumulation over meta-steps BEFORE any window-size sweep.
- The differentiable optimizer remains the top engineering risk; the finite-difference gate (step 3) is the go/no-go.
- Overhead remains the top practical risk; the cost ledger ships with step 6, not after the hero run.
