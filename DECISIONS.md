# Pre-registered decisions — chemical-transformer ICLR expansion

Written before the relevant numbers exist. Sep 12 is a lookup, not a
negotiation. Amendments require a dated note at the bottom.

## 1. Shuffled-control kill rule (written Sep 6, seed 0 only existed)

**The measured-targets (compute-value) claim survives iff**: gate-grad beats
shuffled by **≥ 1.0 point** held-out accuracy, **paired by seed** (same
Stage-1 body), at **matched billed training FLOPs**, across **all 3 seeds**.
Otherwise: report that supervised variation under a budget — not the specific
measured targets — carries the testbed effect; the thesis then rests on the
scale + NL results, and the intro toggle flips to Plan A. Either outcome is
publishable; no post-hoc reinterpretation.

Billing symmetry: the shuffled arm consumes the same oracle measurement pass
as gate-grad (permutation happens after measurement), so its billed training
FLOPs are identical by construction. The NL shared prepass is billed once in
the compute ledger, not per arm.

## 2. Intro framing toggle (~Sep 12)

- **Plan B** (gate-grad leads the intro): iff rule 1 passes AND the
  fixed-schedule gap is ≥ ~1.5 points paired on ≥ 2/3 seeds (seed-0's +2.3
  is not yet load-bearing; testbed noise ±2–3 points).
- **Plan A** (failure diagnosis leads): otherwise. The title does not
  change under either plan.

## 3. Difficulty window (P1 gate)

Pick the **largest** size whose hard-bin held-out accuracy sits inside
**[40%, 85%]** at the probed token budget; that size is the P3 scale and its
token count sets B0. Two-sided on purpose: saturation (>85%) kills the
allocation signal; collapse (<40%) leaves no reducible mass. If 1B fails the
window → P3 runs at 400m; the scale story then rests on NL (1.4B/2.8B).

## 4. Value-migration prediction (P1b readout, pre-registered Sep 6)

As scale grows 42M → 150M → 400M, per-bin mean gate-grad score on H rises
relative to M, and H's frac(score≤0) falls toward M's. **Crossover or clear
shrinkage = confirmation** (figure: `scripts/value_migration.py`).
**H's frac_nonpos pinned high across sizes = the irreducibility mechanism is
wrong** — escalate before spending P3 money.

## 5. Budget-pinned targets are the design, not a bug

Targets (floor 0.1 + quartiles 0.2/0.4/0.6/0.8) have mean ≈ 0.31 and nominal
spread 0.7; λ_budget pins the mean gate at 0.5, so realized gates are lifted
and compressed (seed 0: realized spread +0.09, E gate 0.43 vs floor target
0.1). This is the deployment story — pinned mean = settable inference budget.
Keep the pinned-mean square loss; do **not** switch to an inequality
constraint. Report Table "targets vs realized gates" so compression reads as
redistribution.

## 6. Stage gates (unchanged from plan v3)

- No P3 spend until P1 confirms size + B0.
- P6 (2.8B) cut if P4 slips past Sep 19.
- Cut order when hot: P3→400m, then P5b (math), then P6. Never cut P4/P5a.
- Claims discipline: seed-0-only numbers are "matching," never "better";
  paired per-seed consistency is the load-bearing statistic.

## 7. Abstract discipline

Register conservative abstract Sep 17 (only P0/P1-level claims, scale as "up
to"); sharpen wording through Sep 24 (abstract editable until the paper
deadline). No sentence anywhere may claim the gate-grad arm "orders correctly
by difficulty" — it deliberately does not at 11M; it orders by compute value.

## 8. NL outcome fork (both branches pre-framed, Sep 7)

The natural-language track is designed to be publishable either way:
- **Endogenous signals fail on NL too** (MoD router / entropy baselines
  allocate no better than chance) → the taxonomy generalizes; same thesis,
  bigger blast radius.
- **Endogenous signals work on NL** → reframe as a boundary analysis: "when
  do endogenous difficulty signals work?" — §Discussion's "when static
  schedules suffice" paragraph is the skeleton. Not a failed experiment.
Triangulation guard against the circularity attack: the code domain carries
computable structural proxies (indentation depth, token class) that measured
value should track if it measures anything real; see
`oracle_diag.json:structural_by_grad_bin`.

---
Amendments:

**Sep 12 — Gate 1 readout (11M causal 5-arm, 1 seed, matched everything,
gates from step 0, budget 0.5, 3000 steps).** Dense-inference finals:
dense 1.167/0.642, dropout(per-step) 1.275/0.513, shuffled(100-step
windows) 1.274/0.516, online(field-chasing) 2.253/0.367, static
2.621/0.318. Readouts per the pre-registered rules: (a) per-step ~= windowed
(1.275 vs 1.274) -> **the mechanism is token-level FFN dropout
(noise-regularization); the "rotation" branding is dropped.** (b) static
scaling and field-chasing are the failing arms; intermittent full compute
beats uniformly reduced compute at the same mean gate. (c) dense wins
dense-inference at 3000 steps — no parity claim at this horizon; the 10k
convergence pair decides tonight. Pre-registered parity interpretation
(before the 10k pair): rotation held loss within ~2% of dense -> the
positive claim ("same quality, half the FFN training FLOPs,
dense-deployable") advances to 150m + NL. A 2-10% gap -> the claim scopes
to compute-limited regimes, stated plainly. >10% worse -> the dropout
recipe is dropped; the paper is measurement/life-cycle + taxonomy.

**Sep 12 — Gate 2 executed: KILL on method novelty.** Turbo Training (Han,
Xie, Zisserman, BMVC 2022, arXiv 2210.04889) already establishes train-time
token dropping with dense deployment and training-compute savings (~4x) on
video transformers. Surviving differentiation: LLM decoder fine-tuning
domain; token-level FFN granularity; the value-field life-cycle
measurement; the failure taxonomy of learned allocators; headroom
methodology. The method arm enters the paper as an adopted baseline
(cited), not a novel method. TokenTune (EMNLP 2024) differentiates on
objective (memory, not FLOPs).

**Sep 12 — the life-cycle law, measured at three points.** Ordering value
(waterfill - random, permutation-controlled): structured mid-training
(starved-150M, both seeds: waterfill beats random +0.30/+0.38, beats anti
+0.70/+0.78), decaying on the 11M trajectory (peak |gap| ~0.2 at steps
100-800), ~0 at convergence (all converged bodies; the step-5000 sign is
seed-inconsistent and logged as noise per pre-registration). This resolves
the earlier ambiguity: there is NO P0-vs-retrained discrepancy — every
healthy body measured (11M x3, 42M, 150M) shows a flat converged field;
the P0 pass was always an accuracy result, never a gradient-structure
result.

**Sep 7, 2026 (night) — Rule 4 EXECUTED on valid bodies: NOT CONFIRMED.**
With the validated recipe (cosine + warmup + wd 0.1, preset LRs), healthy
stage-1 bodies at 42M (acc 0.496) and 150M (acc 0.563) show NO difficulty-bin
structure in measured gate-gradient value: frac(score<=0) ~= 0.51 across
E/M/H at both sizes, H-M ~= 0, mean |score| ~ 1e-6 (coin-flip). On a
converged body the first-order marginal value of FFN compute is ~zero and
uninformative. Consequences: (a) the compute-value migration hypothesis is
not supported at 42M/150M and is parked; (b) the NL track's primary targets
are LOSS-QUANTILE, with gate-grad demoted to a diagnostic; (c) 400M is
untrainable in the current post-LN+Xavier arch at any LR in [6e-5, 3e-4]
(sweep-verified) — needs GPT-2-style init or pre-LN, out of critical path.
Live paper claims: failure taxonomy (11M, valid) + supervision/budget fix +
shuffled-control finding (target content irrelevant at testbed) + 150m
window pass (H=0.485 in [0.40,0.85]). Next: single 150m full grid (merged
P2/P3, all methods x 3 seeds, iso-FLOPs) ~ $11-13; needs top-up.
(H100). Paired gate-grad − shuffled accuracy: seed 0 −0.003, seed 1 −0.001,
seed 2 −0.000. Shuffled fully matches gate-grad (means 0.745 vs 0.747);
gate structure also matches (E/M/H 0.45/0.52/0.52 vs 0.47/0.52/0.52).
Per the rule: at testbed scale, **supervised variation under the budget
constraint carries the effect; the specific measured targets do not.** Intro
toggle locked to **Plan A** (failure diagnosis leads). The compute-value
concept is now load-bearing only on the two remaining pre-registered tests:
Rule 4 (value-migration at 42M→150M→400M — now the decisive experiment for
the concept) and the NL shuffled control. Honest reporting: the testbed
finding itself is sharpened, not weakened — dense-then-gate with ANY
varying supervision recovers full-compute accuracy at ~42% fewer FLOPs,
while a constant gate from scratch costs ~2 points; allocation structure
(tag arm's +0.21 spread) is real but not accuracy-load-bearing at 11M.
Billing note: gategrad/shuffled arms billed 3.76M TF each (oracle overhead
included, identical by construction); tag arm 1.41M TF.

**Sep 15 — Starved-regime test EXECUTED (pre-registered): the mid-life
field's structure is NOT exploitable online.** Starved-150M (2500 steps,
budget 0.5, seed 0, validated recipe: preset LR 1e-4, cosine, 2% warmup,
wd 0.1): random windows 44.2% / 1.342 > online field-chasing 41.7% / 1.363;
static g=0.5 from scratch collapses (22.4%, held loss diverging to 3.60).
Per the pre-registered fork ("if tie/shuffled → the field's structure is not
exploitable"): **online loses.** The discriminator's mid-life structure
(waterfill beats random +0.30/+0.38 on frozen bodies) is measurable but does
not transfer to a training signal — chasing the field during training is
worse than random windows at matched billed FLOPs, replicating the 11M
causal result (dense 64.2 > dropout 51.3 ≈ shuffled 51.6 ≫ online 36.7 ≫
static 31.8) at a second scale. Final verdict for the compute-value concept:
the field is (a) measurable and structured mid-life, (b) not exploitable by
online chasing at 11M or 150M, (c) flat at convergence. The surviving recipe
is token-level FFN dropout, dense-deployable, at half the FFN training FLOPs.

**Sep 15 — 150M probe-recipe grid COMPLETE (6 arms × 2 seeds, iso-FLOPs)
and 10k convergence pair; parity claim stays SCOPED.** Dense 68.0±7.0 at
467 MF/tok; the supervised 2-stage arms (tag / shuffled / fixed-2stage) all
62.9±9 at ~270 MF — content-irrelevance replicated at 150M and dense wins
the healthy recipe. MoD 67.5±6.1 at 269 MF with an anti-ordered router
(mean gate 0.25, spread −0.05: hard bins get the least compute); MoD ties
dense on mean accuracy (seed-paired diffs +8.9/−9.7 — a wash under huge
seed variance, not a win). 10k convergence pair (seed 1): rotation held loss
1.086 vs dense 1.032 → **5.3% gap at exactly 50% billed FLOPs** → per the
Sep 12 pre-registration (2–10% band) the claim is scoped to compute-limited
regimes; paper toggle set to \ifparity false.
