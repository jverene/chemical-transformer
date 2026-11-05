# ICLR 2027 Submission 52556 — Rebuttal Investigation and Revision Plan

**Status date:** 2026-11-07  
**Purpose:** Private working brief for the authors and a coding/research agent investigating what can be fixed, tested, and revised during the ICLR discussion period.  
**Core principle:** Do not try to argue the reviewers into acceptance. Produce a scientifically stronger paper and a body of evidence that makes the original objections look outdated.

> **Confidentiality / usage note:** This document contains private assessments of reviewers, speculative forecasts, and possible reporting language. None of those sections should be pasted into public OpenReview responses. Public responses should be concise, technical, civil, and evidence-driven.

---

## 1. Deadline and operating assumptions

According to the official ICLR 2027 timeline:

- Reviews were released on **November 5, 2026**.
- The author–reviewer discussion runs **November 5–18, 2026, Anywhere on Earth (AoE)**.
- Authors may revise the paper until the end of the public discussion period on November 18.
- The reviewer–AC discussion follows from November 19 to December 16.
- Final decisions are scheduled for December 16.

Official sources:

- [ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [ICLR 2027 Policy on LLM Use for Reviews and Meta-Reviews](https://iclr.cc/Conferences/2027/AIPolicyForReviewers)

**Planning convention:** Treat **November 16** as the internal target for a strong revised PDF and substantive results. Leave November 17–18 for reviewer follow-up, correction, and final uploading rather than making the first complete version appear at the deadline.

The available effort is approximately **60 focused hours over a five-day working block**, plus additional time during the remaining discussion window. The exact amount of GPU capacity, existing queued runs, and implementation readiness is not yet known. Those are major uncertainties.

### Important process constraints from the author guidelines

- During rebuttal, the main-text limit is **10 pages**; references and appendices are additional.
- Multiple revisions and multiple responses are allowed.
- Reviewers and ACs are not required to inspect every revision.
- Reviewers and ACs may ignore changes that make the submission substantially different from the original paper.
- A title, abstract, paper body, and supplementary material can be revised, but changes should be communicated clearly.
- A last-minute upload is not a substitute for engagement. Post initial responses early and follow up as results become available.

---

## 2. Current position in plain English

The current overall ratings are **1 / 1 / 2**:

| Reviewer | Overall rating | Confidence | Presentation | Informativeness | Soundness |
|---|---:|---:|---:|---:|---:|
| Reviewer 32DR (“Reviewer 1”) | 1 — clear rejection | 3 | 2 | 1 | 1 |
| Reviewer dmAa (“Reviewer 2”) | 1 — clear rejection | 4 | 1 | 2 | 1 |
| Reviewer rfcv (“Reviewer 3”) | 2 — weak rejection | 2 | 2 | 3 | 1 |

In simple terms:

- Two reviewers are firm “no” votes.
- One reviewer is a softer “no.”
- Nobody currently advocates acceptance.
- All three have significant soundness concerns.
- Reviewer 3 nevertheless gained substantial understanding from the paper and is the least certain reviewer.

This is a bad starting position, but not a dead paper. The useful opening is that the objections are specific enough to guide experiments and revision. Reviewer 3, in particular, has essentially described what evidence would address their concerns.

A useful analogy is: **down 3–0 at halftime against an elite team, not down 3–0 in stoppage time.** The score is bad, the venue’s standard is extremely high, but there is still time to change the game. The way to do that is not flashy rebuttal rhetoric; it is executing against the exact weaknesses identified.

### What ratings would mean after a successful rebuttal

If two reviewers moved to accept-leaning ratings and the third moved to weak reject, that would constitute a credible acceptance case. It would **not** guarantee acceptance. Conversely, one remaining weak reject would **not** veto acceptance.

Ratings are recommendations, not automatic votes. The AC will evaluate the revised evidence, reviewer arguments, unresolved soundness issues, confidence, and discussion. The strongest possible outcome is not merely a numerical change; it is a reviewer explicitly writing that the revision resolved their objections and that they now support acceptance.

---

## 3. Scientific strategy

### 3.1 Keep the core study, not literally every claim

The valuable core includes:

- The failure taxonomy for endogenous gating signals.
- The supervised-gate and shuffled-target controls.
- The distinction between fitting allocation structure and obtaining task benefit.
- The causal field-chasing / gradient-based allocation investigation.
- The negative results and failed predictions.
- The separation of nominal FLOP savings from executable speedups.
- The honest disclosure that dense-fewer-steps currently wins important comparisons.
- The random FFN-skipping recipe as a diagnostic/training-time intervention, where applicable.

What should **not** be preserved unchanged:

- Universal “compute value is not learnable” language.
- Any implication that the tested first-order procedure is a general oracle or upper bound on nonlinear allocation or learning.
- Any headline implying a demonstrated practical efficiency advantage when shorter dense training remains better.
- Any implication that failure of the tested allocators establishes failure of token-adaptive computation generally.
- Compressed, jargon-heavy writing that prevents reviewers from understanding the method and claims.

### 3.2 The two legitimate paper narratives

#### Branch A — No positive learned-allocation result is found

The paper should become a carefully scoped empirical negative-result and diagnostic study:

> In the tested architectures, training regimes, signals, and allocation procedures, naive and locally optimized allocation mechanisms fail to produce useful difficulty-aligned compute allocation. These failures do not establish that adaptive computation is generally unlearnable.

The contribution would be the controlled failure analysis, negative controls, implementation/accounting lessons, and boundary conditions—not a universal non-learnability theorem or a winning efficiency recipe.

#### Branch B — A robust positive learned-allocation result is found

The stronger narrative becomes:

> Naive difficulty signals and local gradient-based allocation fail for identifiable reasons, but a better-co-adapted allocator or allocation rule can succeed under specified conditions.

This is more interesting than either “nothing works” or “random dropout is useful.” It directly addresses Reviewer 3’s proposed alternative explanation and Reviewer 2’s hypothesis that better architectural and loss design can make adaptivity work.

The ideal evidence-backed sentence would be:

> We identified why the original allocator failed, changed that mechanism, and now observe learned allocation consistently beating both random allocation and dense-fewer-steps under corrected total-cost accounting.

This sentence must not be written publicly unless the experiments genuinely support it.

### 3.3 What counts as a meaningful breakthrough

A weak or cherry-picked positive result is not enough. The strongest target is:

- A **learned** allocator, not only random FFN skipping.
- Consistent improvement over **random allocation**.
- Consistent improvement over **dense-fewer-steps**, the strongest natural baseline.
- Correct **cumulative total-cost** accounting.
- Multiple seeds with uncertainty reported.
- No training/inference accounting mismatch.
- Realized wall-clock evidence where practical claims are made.
- Replication beyond one favorable seed, budget, or operating point.
- Preferably replication at another scale or regime.

If the result only shows that random dropout helps somewhere, it addresses practical motivation but not learnability. If a learned allocator beats random but not dense, it addresses learnability but not practical value. Both are potentially useful, but they are different claims and should not be conflated.

---

## 4. Reviewer-by-reviewer analysis

## 4.1 Reviewer 1 — 32DR

### What the reviewer recognized

- The method is simple and requires no architectural change.
- Gather-based skipping turns nominal savings into a reported **1.35× training throughput** improvement.
- The failure taxonomy and the finding that correctly ordered gates do not improve accuracy are useful diagnostics.

### Their objections

1. **Insufficient scale.** Results are at ≤150M except for a 1.4B 3k-step fine-tune, which the reviewer correctly does not treat as 1.4B pretraining.
2. **Questionable practical motivation.** If random FFN dropout loses to dense-fewer-steps at matched budget, the practical value of the recipe is unclear.
3. **Starved-150M robustness.** They ask whether the authors will commit to that arm and whether conclusions survive if gaps shrink within noise.
4. **MoD tension.** They ask how the claim that allocation signals “fail” is compatible with MoD matching dense at **0.58× billed compute**.

### Current private assessment of the review

This is the most template-like review: generic summary, formulaic “critical” strengths/weaknesses, and questions that largely restate the weaknesses. It may be LLM-assisted, but style alone cannot establish that.

It is **not content-free**. Its scale and dense-baseline objections overlap with the other reviews and must be addressed seriously.

The genuinely sloppy wording is the claim that MoD is a “production learned router” delivering “exactly the efficiency their framework denies.” This conflates:

1. Failure of the paper’s tested post-hoc/endogenous allocation signals.
2. A universal claim that token-adaptive computation cannot work.

MoD can contradict (2) without contradicting (1). The response should make that distinction calmly.

“Nitpicky” is not itself a valid complaint. Scale, accounting, and baseline dominance are legitimate subjects of technical review.

### What we want Reviewer 1 to think after revision

> The authors acknowledged the scale limitation rather than hiding behind fine-tuning. The starved-150M claims now have adequate statistical support. The paper no longer claims a practical advantage where dense-fewer-steps wins—or, if new experiments succeed, a realistic advantageous regime is now demonstrated. The MoD discussion now clearly distinguishes failure of the tested allocators from adaptive computation generally.

### Response strategy

- Be brief and technical.
- Cross-reference the revised experiments and paper sections rather than over-investing in argumentative text.
- If no regime beats dense-fewer-steps, say so explicitly.
- Do not mention suspected AI use in the public scientific response.

Reviewer 1 is currently a wildcard. Do not build the entire strategy around pleasing them, but do not ignore their legitimate concerns.

---

## 4.2 Reviewer 2 — dmAa

### What the reviewer recognized

- The high-level question is important given current interest in adaptive computation and LLMs.
- The experiments show that, in the selected setting, learned token-level adaptivity does not outperform random allocation.
- The shuffled-target experiment is particularly interesting.

### Their objections

1. **Writing quality.** They describe the writing as very poor and quote two contribution bullets containing compressed clauses, jargon, parentheticals, and incomplete sentence structures. They consider the readability problem serious enough to prevent acceptance.
2. **Continuous gates and FLOPs.** They believe that if gate output is continuous and nonzero, no compute is saved, and therefore billing `g` times FFN cost may be invalid.
3. **Dense inference motivation.** They do not understand why FFN layers are skipped during training but restored for inference, particularly in Tables 1 and 6.
4. **MoD counterevidence.** They interpret MoD as showing that token-level adaptivity works when architecture and loss design are appropriate.
5. **Other positive adaptive-computation literature.** They cite work on recurrent Transformers and Mixture-of-Recursions and argue that universal “not learnable” language is overstated.

### Current private assessment of the review

This review is harsh but concrete. It engaged directly with the text and supplied exact examples. It is not safe to dismiss.

The FLOPs concern may reflect a misunderstanding of hard execution versus continuous gate values—but that is only a hypothesis until the equation and implementation are audited. If a knowledgeable reviewer cannot tell whether gates are soft multipliers, hard masks, thresholds, or top-k decisions, the paper has an explanation problem even if the code is correct.

The writing complaint cannot be solved by a rebuttal paragraph. It requires a real rewrite, especially of the contributions and method sections.

The reviewer’s MoD interpretation offers an opening: a positive learned-allocation result would validate their hypothesis that better design can make adaptivity work.

### What we want Reviewer 2 to think after revision

> The paper is now readable and uses conventional, complete explanations. The gate semantics and FLOP accounting are mathematically and operationally unambiguous. The dense-inference setting has a clear, limited motivation. The authors now engage properly with MoD and recurrent adaptive-computation work and no longer overgeneralize. If new experiments succeed, the paper now demonstrates the better-designed adaptation mechanism I suspected was possible.

### Response strategy

- Thank them specifically for identifying unclear passages; do not be defensive about the writing criticism.
- Add a precise gate-execution algorithm or figure and correct Eq. (1) if necessary.
- Explain dense inference honestly as a training-time intervention/diagnostic setting, not an inference-efficiency claim.
- Discuss the two cited papers seriously.
- Show the revised contribution statement in the response or summarize the rewrite.
- If a positive result is found, frame it as addressing their proposed explanation.
- Do not assume the score will rise automatically: this is a clear rejection with confidence 4.

---

## 4.3 Reviewer 3 — rfcv

### What the reviewer recognized

- The negative controls and unfavorable results are meaningful.
- Shared-checkpoint forks and shuffled-target controls distinguish successful fitting of allocation structure from actual task benefit.
- The paper separates nominal savings from executable speedups.
- Masking gives no speedup, while gathering active rows produces a reported **1.35× throughput** result.
- The paper acknowledges that shorter dense training wins its natural-language cost-matched comparisons.
- The reviewer sees the basis for a valuable, carefully scoped empirical paper.

### Their objections

1. **The evidence does not establish general non-learnability.**
   - Table 1 tests one hard allocation strategy refreshed every 100 steps.
   - There is one seed per arm at each scale.
   - Supervised heads generally use a nearly frozen body.
   - Better co-adaptation or allocation rules are not excluded.
2. **The first-order procedure is not a general oracle.**
   - Sorting gate gradients optimizes a local linear surrogate.
   - This does not bound nonlinear loss under finite gate changes or retraining.
3. **Total-cost comparisons are inconsistent.**
   - Appendix D acknowledges that the 1.4B counter bills attention once rather than once per layer.
   - The approximately 50% figure concerns FFN-dominated accounting, not total training FLOPs.
4. **Implementation evidence is disconnected from training.**
   - The measured gather speedup comes from a separate short implementation benchmark.
   - The training driver uses masking.
5. **No demonstrated practical advantage.**
   - Shorter dense training is better at billed-cost and estimated wall-clock matches.

### Current private assessment of the review

This is the strongest, most precise, and highest-leverage review. It is also the harshest scientifically because it attacks the central claim rather than only presentation.

It is simultaneously the most useful review. It identifies the exact evidentiary gaps and offers a route to persuasion: test better co-adaptation or allocation rules, add seeds, correct total cost, verify implementation equivalence, and scope the conclusion.

More seeds alone do not solve the central objection. Repeating the same allocator more times only improves confidence about that allocator’s failure; it does not establish that other allocators cannot work.

### What we want Reviewer 3 to think after revision

> The authors understood the distinction between failure of one surrogate/update rule and general non-learnability. They removed the unsupported oracle language, tested the most important alternative explanation, added statistical replication, corrected total-cost accounting, verified gather–mask equivalence, and calibrated practical claims. The paper is now a rigorous scoped study—or, if the new allocator works, a substantially stronger boundary-result paper.

### Response strategy

- Lead with agreement where agreement is warranted.
- Separate each concern into a concrete action and artifact: new table, corrected equation, new seed panel, equivalence test, or rewritten claim.
- Test the proposed alternative explanation directly if feasible.
- Keep dense-fewer-steps front and center.
- Ask whether a specific concern is resolved, not whether the reviewer will raise the score.

Reviewer 3 is the most plausible future advocate and should receive the most rigorous technical response.

---

## 5. Investigation plan for the coding/research agent

The investigation should be open-ended. Do not assume a particular new allocator will work. First understand the code and existing evidence, then select the cheapest experiments that discriminate among explanations.

### 5.1 Phase 0 — Repository and claim audit

Before launching a large sweep, locate and document:

1. **Gate semantics**
   - Where `g` is defined.
   - Whether `g` is continuous during forward computation.
   - Whether execution uses a threshold, top-k, Bernoulli draw, or another hard decision.
   - Whether billed FLOPs describe an expectation, a relaxation, or actual executed operations.
   - Whether Eq. (1) matches the implementation.

2. **Masking versus gathering**
   - Which experiments use output masking.
   - Which benchmarks use gathering.
   - Whether the training driver can use gathering.
   - Whether random-number generation, row ordering, gradients, and optimizer updates differ.

3. **FLOP accounting**
   - Whether attention is counted once or once per layer.
   - Whether forward and backward costs are both included.
   - Whether gate/router overhead, target construction, sorting, allocator refreshes, and all training stages are included.
   - Whether “50%” refers to FFN-only FLOPs or total training FLOPs.
   - How wall-clock estimates are constructed.

4. **Allocator training**
   - Which body parameters are frozen or nearly frozen.
   - Why the allocator refreshes every 100 steps.
   - How gate gradients are computed and sorted.
   - Whether the surrogate uses infinitesimal changes while execution uses finite hard changes.
   - Whether supervised heads and endogenous gates co-adapt with the body.

5. **Baselines and controls**
   - Dense full training.
   - Dense-fewer-steps at correctly matched cost.
   - Random skipping.
   - Shuffled-target controls.
   - MoD implementation and compute accounting.
   - Any other adaptive-computation baselines already available.

6. **Existing runs**
   - Completed seeds and checkpoints.
   - Queued starved-150M seeds.
   - Existing positive or ambiguous signals.
   - Failed runs and whether failures were scientific or engineering failures.

**Deliverable from this audit:** A table mapping every reviewer objection to the relevant code path, configuration, existing evidence, missing evidence, and cheapest next test.

### 5.2 Track A — Guaranteed improvements regardless of a breakthrough

#### A1. Multi-seed starved-150M experiments

Launch these early because they have direct reviewer value and may have long runtimes.

Include, at minimum:

- Relevant dense baseline(s).
- Dense-fewer-steps.
- Random allocation.
- The existing learned/gradient allocator.
- Any new allocator variant only after a pilot justifies inclusion.

Requirements:

- Use enough seeds to report uncertainty honestly; the exact number depends on compute.
- Preserve common checkpoint forks and data controls where those are part of the original design.
- Predefine budgets, metrics, and comparisons.
- Report individual-seed results, not only selected averages.
- Do not cherry-pick a favorable operating point after seeing results.

#### A2. Correct cumulative total-cost accounting

Create a single authoritative accounting implementation or table that separates:

- Billed FFN FLOPs.
- Attention FLOPs, counted per layer.
- Forward and backward costs.
- Gate/router and allocator overhead.
- Target-generation and refresh costs.
- All training stages.
- Total training FLOPs.
- Measured wall-clock time.
- Tokens or optimizer steps processed.

Add sanity checks or unit tests comparing analytical accounting against logged execution counts where feasible.

#### A3. Gather–mask equivalence and realized throughput

Verify:

- Matching outputs under equivalent hard decisions.
- Matching gradients with respect to retained rows and relevant parameters.
- Correct handling of dropped rows.
- Equivalent optimizer updates under controlled conditions.
- Short training-curve equivalence, not merely one forward-pass comparison.
- Throughput at realistic model, batch, sequence, and sparsity configurations.
- Whether the measured implementation is the same path used for the reported training claims.

If exact equivalence is impossible because of RNG or implementation semantics, explain the difference and measure its effect rather than asserting equivalence.

#### A4. Dense-fewer-steps as the principal practical baseline

For any practical recipe claim:

- Compare at matched cumulative total cost.
- Compare at matched realized wall-clock cost where possible.
- Show quality–compute curves rather than one selected budget.
- Include dense-fewer-steps prominently in the main paper.
- If dense remains superior, state that no practical advantage is currently demonstrated.

### 5.3 Track B — Direct investigation of Reviewer 3’s alternative explanations

The goal is not to launch every conceivable variant. Use small pilots to determine which mechanism is most diagnostic.

Possible hypotheses to investigate, none of which should be presumed true:

1. **Insufficient co-adaptation:** A nearly frozen body prevents the model from adapting representations to the allocator.
2. **Stale allocation:** A 100-step refresh makes targets poorly aligned with the current model.
3. **Hard-decision mismatch:** Infinitesimal gate gradients do not predict the effect of finite hard skipping decisions.
4. **Local-surrogate failure:** Sorting first-order values is harmful because the actual loss response is nonlinear.
5. **Sensitivity versus value:** Large gradients measure sensitivity, not necessarily the improvement obtainable by allocating more compute.
6. **Objective mismatch:** A gate can learn the supervised difficulty order while the body receives no useful training benefit from that ordering.
7. **Budget/regime dependence:** Useful allocation may only appear in particular starvation, data, or scale regimes.
8. **Optimization interference:** Allocator learning and body learning may destabilize each other without an appropriate objective or update scheme.

Potential experimental manipulations include:

- Full versus partial body co-adaptation.
- Different allocator refresh intervals.
- Alternative hard/soft training relaxations.
- Finite-change evaluations rather than only infinitesimal gradients.
- Alternative allocation rules or update objectives.
- Better joint training of allocator and model.

These are options, not instructions to run a blind grid. Select a small number based on the audit and early signals.

### 5.4 Track C — Moonshot positive-result search

Use a staged process:

1. **Small diagnostic pilot:** Enough capacity to detect a meaningful mechanism-level signal.
2. **Predefined scale gate:** Scale only if the pilot improves the relevant comparison without accounting loopholes.
3. **Seed replication:** Test whether the result survives independent seeds.
4. **Baseline hardening:** Add or verify dense-fewer-steps and random controls.
5. **Regime/scale replication:** Test whether the result is more than one favorable operating point.
6. **Realized-cost validation:** Confirm that claimed savings survive implementation and wall-clock measurement.

Do not spend the entire compute budget scaling a variant with no early signal. Do not redefine success after the results merely to create a positive story.

### 5.5 Scale strategy

- The existing 1.4B result is a short fine-tune and should be described exactly that way.
- Do not claim 1.4B pretraining validation unless such pretraining actually occurs.
- If a positive result appears, first replicate it at the cheapest meaningful second scale or regime.
- A clean small-scale result with proper controls is more credible than a rushed large run with missing baselines.

### 5.6 MoD and related work

A new MoD sweep is not automatically necessary. First determine whether the scientific need is:

- A better discussion of scope.
- A corrected accounting comparison.
- A verification that the baseline is implemented and billed fairly.
- An architectural/co-adaptation contrast that informs the new allocator.

Discuss MoD and recurrent adaptive-computation work as evidence that token-adaptive computation can work under appropriate designs. Explain why this does not invalidate a properly scoped negative result about the tested allocators.

---

## 6. Suggested 60-hour work allocation

This is a planning budget, not a rigid schedule. GPU runs should proceed in parallel with writing and analysis.

| Workstream | Approximate focused effort |
|---|---:|
| Code/config audit and early run launches | 8 h |
| Multi-seed experiments and allocator pilots | 20 h |
| Accounting, equivalence, and wall-clock validation | 10 h |
| Full paper readability and claim rewrite | 12 h |
| Reviewer responses and result communication | 6 h |
| Buffer for failed runs and corrections | 4 h |
| **Total** | **60 h** |

### Ordering rules

1. **First 2–3 hours:** Launch the guaranteed multi-seed runs and the smallest informative allocator pilots. Do not polish prose while GPUs sit idle.
2. **Immediately afterward:** Complete the gate/FLOP/accounting audit so experiments do not inherit a known accounting bug.
3. **In parallel:** Rewrite the contributions and method explanation for Reviewer 2.
4. **Early discussion:** Post substantive initial responses listing agreed concerns and running experiments without promising unobserved results.
5. **Midpoint:** Scale only the most promising allocator pilot; freeze the main experimental plan soon enough to analyze results properly.
6. **By November 16:** Aim for a revised PDF, completed main tables, and evidence-based responses.
7. **November 17–18:** Use for reviewer follow-up, corrections, and final revisions.

**Experiment selection rule:** Every run should answer a specific sentence in a review. If it does not, it probably belongs in the next paper rather than this rebuttal.

**Operational caution:** Sleep deprivation creates accounting errors, broken controls, and overclaimed results. Let GPUs run overnight; do not confuse author hours with GPU hours.

---

## 7. Decision tree after experiments

### Outcome 1 — Learned allocator beats random and dense-fewer-steps robustly

- Move the result into the main paper.
- Reframe around why naive allocation failed and what changed.
- Remove universal non-learnability language.
- Include multi-seed uncertainty and corrected total/realized cost.
- Replicate at another regime or scale if possible.
- Tell all three reviewers exactly which of their hypotheses the result addresses.

### Outcome 2 — Learned allocator beats random but not dense-fewer-steps

- Claim evidence of learnability in the tested setting.
- Do not claim practical efficiency.
- Position the result as a mechanism/boundary finding.
- Keep dense-fewer-steps as a negative practical comparison.

### Outcome 3 — Random skipping beats dense in a limited regime, but learned allocation still fails

- Claim only a regime-specific training recipe result.
- Do not use it as evidence that difficulty-guided allocation is learnable.
- Scope practical claims narrowly and replicate before emphasizing them.

### Outcome 4 — No positive result appears

- Complete the rigorous negative-result revision.
- Add seeds, correct accounting, verify implementation, and narrow claims.
- Explain which alternative allocators were tested and which remain untested.
- Preserve the strongest controls and diagnostic findings.
- Accept that this cycle remains unlikely, while producing a much stronger next submission.

### Outcome 5 — Experiments are incomplete or ambiguous

- Report only what is actually available.
- Do not describe running experiments as completed evidence.
- Distinguish engineering failures from scientific negative results.
- Avoid adding speculative conclusions to the paper.

---

## 8. Paper revision checklist

### Title, abstract, and claims

- [ ] Remove universal non-learnability language unless genuinely proven.
- [ ] Remove “oracle” or upper-bound implications for the first-order surrogate.
- [ ] State the tested signals, update rule, allocator, and regimes precisely.
- [ ] Separate learnability claims from practical-efficiency claims.
- [ ] If a positive result exists, explain the boundary between failure and success.

### Contributions and writing

- [ ] Rewrite contribution bullets as complete, conventional sentences.
- [ ] Remove compressed parentheticals and insider jargon.
- [ ] Explain “iso-FLOPs,” “starved regime,” “field-chasing,” and other necessary terminology at first use.
- [ ] Make the paper understandable without knowledge of the earlier workshop version.
- [ ] Perform a full readability pass, not only a contribution-section edit.

### Gate and FLOP explanation

- [ ] Define continuous gate values and hard execution decisions separately.
- [ ] Reconcile Eq. (1) with actual implementation.
- [ ] State whether billing is expected, relaxed, or realized compute.
- [ ] Include a small algorithm or diagram if it removes ambiguity.
- [ ] Count attention per layer and include all stages in total-cost comparisons.

### Training versus inference

- [ ] Explain why dense inference is studied.
- [ ] Describe it as a training-time intervention/diagnostic where appropriate.
- [ ] Do not imply inference savings when dense inference is used.
- [ ] Identify exactly which tables use this setting.

### Results and implementation

- [ ] Add multi-seed results and uncertainty.
- [ ] Put dense-fewer-steps in the main comparisons.
- [ ] Verify gather–mask equivalence.
- [ ] Separate short-benchmark throughput from end-to-end training speedup.
- [ ] Report wall-clock results where practical claims depend on them.
- [ ] Report unfavorable outcomes and ambiguous results honestly.

### Related work

- [ ] Discuss MoD without treating it as irrelevant or as a contradiction of every tested negative result.
- [ ] Discuss Moosa et al., “Understanding Dynamic Compute Allocation in Recurrent Transformers,” ICML 2026.
- [ ] Discuss Bae et al., “Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation,” NeurIPS 2025.
- [ ] Narrow universal claims in light of positive adaptive-computation literature.

### Revision communication

- [ ] Provide a concise global list of major changes.
- [ ] Reference exact sections, tables, and figures in reviewer responses.
- [ ] Use the PDF diff to make changes easy to inspect.
- [ ] Ensure the revision remains recognizable as a major revision of the original paper, not an unrelated new submission.

---

## 9. Reviewer communication plan

### General response formula

Each major response should follow:

1. **Agreement or precise disagreement.**
2. **Concrete action taken.**
3. **Evidence now available.**
4. **Claim narrowed or changed.**
5. **Specific question about whether the concern remains.**

Example tone:

> We agree that the original version overgeneralized from one hard allocator and used inconsistent total-cost accounting. We have now (i) …, (ii) …, and (iii) …. New Table X reports … . We no longer claim …; the revised conclusion is limited to … . We would be grateful to know whether this addresses your concern about … .

Do not ask reviewers to raise scores. Do not argue that a concern is “nitpicky.” Do not make public accusations about AI use.

### Initial global response

Post early, before all experiments finish:

- Thank reviewers for specific criticisms.
- Acknowledge the three central categories: claim scope, statistical/allocator evidence, and accounting/implementation consistency.
- List the writing revision.
- State which experiments are running.
- Avoid promising positive outcomes.
- Commit to follow-up comments containing actual results.

### Reviewer 3 response

This should be the most detailed technical response:

- Address surrogate versus general learnability.
- Address co-adaptation and allocation-rule alternatives.
- Report new seeds.
- Explain corrected cost accounting.
- Report gather–mask equivalence and wall-clock evidence.
- State clearly whether any regime beats dense-fewer-steps.

### Reviewer 2 response

- Acknowledge the writing problem and describe the rewrite.
- Explain gate semantics and FLOPs with a precise reference to the revised method.
- Explain dense inference.
- Discuss MoD and recurrent models.
- Narrow the claim.
- Add new evidence if available.

### Reviewer 1 response

- Keep it concise.
- Answer scale, starved-arm, realistic-regime, and MoD questions.
- Cross-reference the common revised evidence.
- Do not disclose private speculation about how the review was written.

### Engagement principles

- Post follow-ups when important results arrive rather than waiting for one final wall of text.
- Make it easy for reviewers to inspect changes.
- If a reviewer remains unconvinced, ask for the narrowest remaining technical objection.
- If new evidence changes the paper’s story, say explicitly that the original interpretation was wrong or too broad.
- Serious concessions plus concrete evidence are more persuasive than exhaustive defensiveness.

---

## 10. Optional Reviewer 1 quality/AI-policy concern

This is separate from the scientific rebuttal and optional.

ICLR 2027 permits limited and responsible AI assistance in reviews but requires disclosure, including the reviewer’s original self-written assessment and LM interactions. The policy also identifies hallucinations, misrepresentations, and otherwise low-quality reviews as potential disciplinary issues.

Style alone is not evidence of undisclosed AI use. “The review is harsh” and “the review is nitpicky” are not policy violations. The only plausible concern is a combination of possible undisclosed AI use and concrete misrepresentation or low-quality characterization.

If the authors choose to raise it, use a short **private** message to the AC after the substantive response stands on its own:

> We would like to flag a review-quality and possible undisclosed-AI-use concern regarding Review 32DR. In particular, the characterization of MoD as a “production learned router” delivering “exactly the efficiency their framework denies” conflates adaptive computation generally with the specific allocators evaluated in our paper. The review’s highly templated construction raised additional concern, although we recognize that style alone is not evidence of AI use. We have addressed its substantive questions publicly and ask only that the AC assess its quality and policy compliance.

Do not make this complaint a centerpiece. Even if Reviewer 1 were discounted, Reviewers 2 and 3 independently raise enough overlapping concerns that the scientific revision remains necessary.

---

## 11. Subjective forecasts and scenario thinking

These numbers were developed as judgmental planning aids, **not empirical base rates or statistically calculated confidence intervals**. The intervals below are subjective 80% uncertainty ranges. They should not appear in public responses and should not be treated as precise predictions.

### Completion estimates

| Outcome by deadline | Point estimate | Subjective interval |
|---|---:|---:|
| Improved PDF, responses, corrected claims/accounting | 85% | 70–95% |
| Above plus meaningful multi-seed and gather/wall-clock results | 70% | 50–85% |
| Some promising positive learned-allocation signal | 25% | 10–45% |
| Clean result beating random and dense-fewer-steps across seeds | 15% | 5–35% |
| Materially stronger paper regardless of decision | 90% | 75–97% |

### Reviewer movement without a major positive breakthrough

| Outcome | Point estimate | Subjective interval |
|---|---:|---:|
| Reviewer 3 raises rating or soundness | 55% | 35–75% |
| Reviewer 2 raises rating or soundness | 35% | 20–55% |
| Reviewer 1 raises rating or soundness | 20% | 8–40% |
| At least one reviewer moves positively | 65% | 45–82% |
| Reviewer 3 becomes accept-leaning advocate | 25% | 12–45% |

With a clean positive result, the rough subjective estimates rise to approximately 80% for Reviewer 3 movement, 65% for Reviewer 2, and 35% for Reviewer 1. Reviewer engagement remains uncertain.

### Acceptance scenarios

| Scenario | Subjective acceptance estimate | Subjective interval |
|---|---:|---:|
| Solid fixes, no practical/learned-allocation breakthrough | 3% | 1–8% |
| Clean replicated positive result and excellent revision | 20% | 8–40% |
| Overall unconditional estimate before new investigation | 7% | 2–18% |

The strategic interpretation is more useful than the numbers: **the goal is to earn entry into the strong-positive-result branch, while guaranteeing that the paper improves even if that branch does not materialize.**

---

## 12. Open questions for the authors and agent

These should be answered during the audit:

1. What exactly does the gate multiply during training, and when is a hard skip executed?
2. Is Eq. (1) intended as expected billing, relaxed billing, or actual executed FLOPs?
3. Which reported tables use masking, and which use gathering?
4. Can the actual training driver use the gather path without changing optimization behavior?
5. Where exactly does attention-once accounting enter Appendix D and the headline percentages?
6. Are the starved-150M seeds already running? How many, and when will they finish?
7. What GPU types and total GPU-hours are realistically available before November 16?
8. Which existing runs provide the fastest clean pilot for co-adaptation?
9. Is the nearly frozen body a deliberate scientific control, an engineering convenience, or both?
10. What is the smallest experiment that could falsify the local-surrogate explanation?
11. Is there any existing evidence that another refresh interval or allocator helps?
12. Does dense-fewer-steps remain stronger after correcting total-cost accounting?
13. Is MoD compared under equivalent data, optimization, architecture, and cost assumptions?
14. What is the minimum result that would justify changing the paper’s title and central narrative?
15. Which experiments are essential for this rebuttal, and which should be deferred to the next paper?

---

## 13. Definition of done

A credible major revision should have:

- [ ] A completed objection-to-evidence matrix.
- [ ] Audited gate semantics and FLOP accounting.
- [ ] Corrected cumulative total-cost comparisons.
- [ ] Multi-seed evidence for the most important fragile comparisons.
- [ ] Gather–mask equivalence evidence and realized throughput measurements.
- [ ] Dense-fewer-steps treated as a principal baseline.
- [ ] At least one direct investigation of co-adaptation/allocation alternatives, if computationally feasible.
- [ ] No universal claims unsupported by the evidence.
- [ ] A full readability rewrite.
- [ ] Serious engagement with MoD and the cited recurrent adaptive-computation work.
- [ ] Reviewer-specific responses posted early and updated with results.
- [ ] A revised PDF available before the final deadline buffer.
- [ ] No public discussion of reviewer-authorship suspicions, private score forecasts, or rebuttal gamesmanship.

The realistic best case is a transformed paper with at least one informed reviewer advocate. The guaranteed valuable outcome—if the work is executed carefully—is a substantially stronger study and a reusable experimental/accounting foundation, even if this ICLR cycle still ends in rejection.

---

# Appendix A — Complete supplied review export, verbatim

The following text is copied from the supplied review export without editing its scientific content. Formatting artifacts from the export are preserved.

```text
Official Review of Submission52556 by Reviewer 32DR
Official Reviewby Reviewer 32DR21 Oct 2026 at 11:31 (modified: 06 Nov 2026 at 07:37)EveryoneRevisions
Paper Summary:
This paper asks whether Transformer FFN compute can be allocated according to token difficulty. Endogenous gating signals often fail, and even supervised gates that learn the intended difficulty order perform almost identically to gates trained on shuffled targets. The measured “value of compute” varies across tokens mid-training but flattens near convergence; allocating compute according to it performs worse than random allocation. The authors therefore train with random FFN skipping for 50% of tokens and deploy the full model. This improves measured throughput and dense training remains stronger at equal budget for the 1.4B model.
Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 1: I learned little or nothing new from this paper.
Soundness: 1: I have significant concerns about the paper’s soundness that do not appear easy to address.
Critical Strengths:
Simple method. Token-level FFN dropout needs no architectural changes and the gather-skip implementation turns billed FLOP savings into a measured 1.35× training throughput.
Interesting failure case study. The collapse taxonomy of endogenous gating signals showing that correctly ordered gates bring no accuracy benefit, is a clean and useful diagnostic for the adaptive-computation literature.
Critical Weaknesses:
Insufficient experiments. All result is at ≤150M scale (the 1.4B result is a 3k-step fine-tune, not pretraining).
Questionable motivation. If random FFN dropout loses to the trivial "dense, fewer steps" baseline at matched budget, the practical value of the entire gating study is unclear—the paper spends most of its effort ruling out allocators in service of a recipe that is itself dominated, and is admittedly a per-token variant of standard structured dropout.
Questions For Authors:
Limited applicability: Can the authors clarify whether there exists any realistic training setting in which the proposed recipe is preferable to the trivial baseline of training a dense model for proportionally fewer steps at matched budget?
Insufficient experiments: Would the authors commit to the starved-150M arm, and would the conclusions survive if the gaps shrink within noise?
Questionable motivation: Given that the MoD baseline matches dense at 0.58× billed compute, how do the authors reconcile their headline claim that allocation signals "fail" with the fact that a production learned router appears to deliver exactly the efficiency their framework denies—beyond arguing that its router ordering is not difficulty-aligned?
Flag For Ethics Review: No
Rating: 1: Clear rejection
Confidence: 3: I am familiar with this topic and its relevant literature.
Add:

Official Comment
Official Review of Submission52556 by Reviewer dmAa
Official Reviewby Reviewer dmAa20 Oct 2026 at 21:10 (modified: 06 Nov 2026 at 07:37)EveryoneRevisions
Paper Summary:
The central question in the paper is to understand whether Transformer models can automatically decide to use more computation -- by choosing to run the FFN layers -- for harder tokens and less compute for easier tokens. In the settings the authors have chosen, they show that none of the three signals -- entropy, complexity-based initialization, learned gates -- are able to do this. Learning the gate with complexity supervision does work, but does not improve accuracy. The authors also propose using loss gradients to learn the allocator and show that it actually performs worse than random allocation. With the total training compute fixed, dense Transformer training matches in performance with the FFN skipping method.
Presentation: 1: Difficult to follow despite careful reading.
Informativeness: 2: I learned something interesting or useful.
Soundness: 1: I have significant concerns about the paper’s soundness that do not appear easy to address.
Critical Strengths:
With recent increasing interest in adaptive computation and LLMs, the authors tackle an important high-level question to better understand if models can learn token-level adaptivity that depends on hardness.

For the particular setting and method for adaptive compute that the authors have chosen, the experiments are interesting and show that learned token-level adaptivity does not outperform random compute allocation. The experiment with shuffled targets is particularly interesting.

Critical Weaknesses:
I see several issues with this paper.

The writing is very poor, in my opinion. Several paragraphs in the paper have almost no full sentences. These are two examples from the contributions section:
"Iso-FLOPs methodology: all method comparisons at matched billed training FLOPs (gated billing, forward + backward), with within-segment interpolation and a budget sweep; the “similar total budget” caveat of an earlier workshop version of this work is retired (§2.5)."

"Scale replication, honestly scoped: the failure taxonomy, the shuffled-control content-irrelevance result, and the causal field-chasing test all replicate at 150M (probe grid, starved regime); the dropout recipe’s parity claim is scoped to compute limited regimes by its pre-registered decision rule (−5.3% loss at 50% billed FFN FLOPs at 150M; −7.79% at 1.4B web-text fine-tuning; the template-dominated data boundary is reported, not hidden"

I cannot imagine a reader easily understanding what these paragraphs mean. There is a lot of unusual jargon and sentence constructions and poor explanations which make it extremely difficult to understand the paper. In my opinion, this alone would make the paper not acceptable for ICLR.
My understanding is that the gate output is a continuous quantity between 0 and 1. But this means that, for any number other than 0, absolutely no compute savings are possible. So, I don't believe the FLOPs calculations are correct.
In many of the experiments, some FFN layers are dropped during training, but all the layers are kept for inference. I cannot understand the motivation for this, at all. I believe this is the setting used for all the results in Table 1 and Table 6.
There is also conflicting evidence within the paper. The Mixture-of-Depths baseline in Table 2 clearly shows that inference efficiency is improved without any significant loss in accuracy. However, it is the methods in this paper that have the large drops in performance compared to dense training and inference. I would interpret this as showing that token-level adaptivity works fine as long as the architectural and loss designs are good. It may be that this specific design chosen by the authors does not work.
There are several other works on adaptive computation applied for recurrent neural networks for different domains like algorithmic reasoning that do show that the idea works fine. So, at the very least, the authors are vastly overstating the claim that token-level adaptivity is "not learnable". These are recent examples that the authors may find interesting:

Moosa et al., "Understanding Dynamic Compute Allocation in Recurrent Transformers", ICML 2026. Bae et al., "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation", NeurIPS 2025
Questions For Authors:
Could you please explain the FLOP calculation you have used? Just because the gate output is 0.5, the FLOPs cannot become 50%. They would remain at 100% as per Eq (1). This is from the paper in Section 2.1: "Compute is billed proportionally: a position gated at g is billed g times the per-token FFN cost; attention is always paid in full."
Perhaps I am not understanding something correctly, please explain this clearly.

Could you please explain why having adaptive depth in training but keeping dense inference is an interesting setting to study?
Could you please also address the weaknesses I have mentioned above?
Flag For Ethics Review: No
Rating: 1: Clear rejection
Confidence: 4: I have substantial direct research experience on this topic.
Add:

Official Comment
Official Review of Submission52556 by Reviewer rfcv
Official Reviewby Reviewer rfcv18 Oct 2026 at 01:27 (modified: 06 Nov 2026 at 07:37)EveryoneRevisions
Paper Summary:
This paper studies whether endogenous signals can usefully allocate token-level feed-forward computation. It examines several gating failures, compares difficulty- and gradient-based targets with shuffled controls, and measures how gate-gradient structure changes during training. An online allocation experiment finds the tested gradient-following strategy worse than random allocation at two arithmetic scales. The paper also evaluates random token-level FFN dropout in Pythia-1.4B fine-tuning and benchmarks a gather-based implementation. Results reveal limited value from the tested allocation signals and document quality–compute trade-offs, although the broader claim that compute value is not learnable exceeds the demonstrated settings.
Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 3: I gained a substantially deeper understanding of the problem or question.
Soundness: 1: I have significant concerns about the paper’s soundness that do not appear easy to address.
Critical Strengths:
The study uses meaningful negative controls and reports unfavorable outcomes. Section 4.2 compares measured targets with shuffled targets on shared checkpoint forks, while Sections 4.4–4.5 disclose failed predictions and weak transfer. These experiments distinguish successful fitting of allocation structure from actual task benefit. The distinction is consequential and could support a valuable, carefully scoped empirical paper.
The paper separates nominal savings from executable speedups. Section 4.7 and Appendix F show that simply masking FFN outputs provides no speedup, whereas gathering active rows reaches 1.35× throughput in the reported benchmark. The paper also acknowledges that shorter dense training wins its natural-language cost-matched comparisons. These practical observations help prevent misleading interpretations of gated-compute accounting.
Critical Weaknesses:
The evidence does not establish that the measured structure is not learnable. Table 1 tests one hard allocation strategy refreshed every 100 steps, with one seed per arm at each scale; supervised heads generally use a nearly frozen body. Failure of these procedures does not exclude better co-adaptation or allocation rules. Moreover, sorting gate gradients optimizes a local linear surrogate, not the nonlinear loss under finite gate changes or retraining, so it is not a general upper bound. These gaps directly undermine the title and central conclusion.
The positive efficiency case remains unsupported at a consistent total-cost comparison. Appendix D acknowledges that the 1.4B counter bills attention once rather than once per layer, so the approximately 50% figure concerns FFN-dominated accounting rather than total training FLOPs. The measured speedup comes from a separate short implementation benchmark, whereas the reported training driver uses masking. Section 4.5 additionally finds shorter dense training better at both billed-cost and estimated wall-clock matches. These disclosures support a trade-off study, but not a demonstrated advantage of the proposed recipe.
Questions For Authors:
Clarification: Is there a formal argument that the first-order water-filling procedure bounds nonlinear allocation or learning performance? If not, will the oracle language and non-learnability conclusion be restricted to the tested surrogate and update rule?
Clarification: Are the additional starved-150M seeds mentioned as queued in the limitations available? What existing evidence distinguishes failure of the specific 100-step hard allocator from a general inability to learn useful compute allocation?
Clarification: Can the existing runs be compared using cumulative total training cost with attention counted per layer and all stages included? Was equivalent training behavior verified for the gather implementation, and is there any reported regime where the recipe outperforms shorter dense training at matched realized cost?
Flag For Ethics Review: No
Rating: 2: Weak rejection
Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.
Add:
```
