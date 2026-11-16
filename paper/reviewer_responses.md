# Submission 52556: reviewer-response plan and drafts

Private working document. The response plan is not for OpenReview. The three responses below are drafted to be usable before the 150M pilot finishes: none of their claims depends on that run. Scale-specific results should be added later as a separate follow-up, once the completed artifacts have been checked. If a comment-length limit requires it, split a response at its numbered headings rather than removing the qualifications.

## Response plan

| Review concern | Independent of 150M pilot? | Response strategy |
|---|---|---|
| dmAa: compressed and unclear writing | Yes | Acknowledge directly; describe the full rewrite of the title, abstract, contributions, methods, and claims rather than defending the original prose. |
| dmAa: fractional gates cannot realize proportional FLOP savings | Yes | Agree for the continuous relaxation. Separate Eq. (1) from the new binary Top-B execution; retain soft gates only as diagnostics and describe 50% as FFN-row activity, not total-FLOP savings. |
| dmAa: why train sparsely but deploy densely? | Yes | Explain the limited training-intervention/regularization motivation, explicitly disclaim inference savings, and separate sparse from dense evaluation throughout. |
| dmAa and 32DR: MoD and positive adaptive-computation literature | Yes | Withdraw the invalid numeric MoD comparison, engage with the published positive results, and narrow the conclusion. New positive allocation results make the revision consistent with their interpretation that better designs can work. |
| dmAa and rfcv: universal non-learnability and oracle language | Yes | State that the original conclusion was too broad. The gate gradient is only a local linear sensitivity. Report the new realized-update and frozen-rule positive results with explicit limits. |
| rfcv: frozen bodies and a single 100-step hard allocator | Yes | Describe 200-step body/router co-adaptation, meta-updates every ten steps, hard masks, and two confirmation bodies. Add the frozen-rule evidence across four bodies and two budgets. |
| rfcv: one seed per original arm | Partly | New 11M meta-router and frozen-program replication is available now. Do not claim that this resolves model-scale replication. |
| rfcv and 32DR: starved-150M arm | Withdrawal is independent; replacement scale evidence is not | Disclose that the driver ignored `--size` and that the claimed replication is withdrawn. Do not present the new 150M continuation pilot as the same starved causal experiment. |
| rfcv: cumulative total cost and attention accounting | Yes | Withdraw the affected 1.4B total-cost interpretation; describe per-layer attention, forward/backward, and auxiliary accounting in the revised practical comparisons. |
| rfcv: gather/mask equivalence and disconnected speed benchmark | Yes | Distinguish primitive numerical validation from end-to-end legacy training equivalence. Keep 1.35x as a microbenchmark only; base new execution claims on the actual hard-budget path. |
| rfcv and 32DR: any regime better than dense-fewer? | Yes | Answer no for replicated practical superiority. Report the small development win and its failed frozen replication, not just random-control wins. |
| 32DR and rfcv: second-scale evidence for the new mechanisms | **No—wait** | Add only completed 150M pilot outcomes. A positive pilot would be a 50-step continuation signal on two bodies, not full scale confirmation or Level 3. |

## Optional global revision comment

Thank you for the detailed and specific criticisms. We have substantially revised both the claims and the experimental basis of the paper.

The revised paper no longer claims that useful token-level allocation is generally unlearnable, and no longer treats the first-order gate-gradient rule as an oracle or upper bound. We directly test co-adaptation with an online router optimized through realized AdamW updates, and separately evaluate a frozen activation-based allocation rule. Both improve over exact-budget random allocation in the tested 11M continuation setting.

We also corrected the execution and accounting interpretation. Fractional gates are now described only as continuous relaxations. New allocation experiments use binary Top-B masks, with gather/scatter execution for the training claims. Sparse and dense deployment are reported separately. Practical comparisons account for attention per layer, forward and backward work, and auxiliary computation where applicable. The previous 1.4B total-FLOP interpretation, unsupported starved-150M causal claim, and invalid MoD numeric comparison have been withdrawn.

Finally, we now state the practical boundary plainly: neither new allocation mechanism has demonstrated replicated superiority over dense training for fewer steps at matched total cost. The contribution is therefore scoped evidence about when allocation is learnable, together with the original diagnostics and implementation audit—not a claim of universal non-learnability or practical efficiency superiority.

## Response to Reviewer dmAa — pilot-independent draft

Thank you for quoting the exact passages and for separating the writing, execution, and deployment concerns. We acknowledge each of them. The two quoted contribution paragraphs have been removed; the dense-inference tables no longer carry an inference-efficiency interpretation; the invalid numeric MoD comparison has been withdrawn; the cited recurrent adaptive-computation work is now discussed; and the universal non-learnability wording has been removed. Below we focus on the three issues that required substantial changes: the paper’s exposition, the meaning of the gate and its FLOPs, and the missing test of better allocation designs.

### 1. Writing and organization

We agree that the original contribution statement was too compressed to be useful, even to a careful reader familiar with the area. We have rewritten the title, abstract, introduction, contributions, methods, results, and limitations rather than only editing the two quoted bullets.

The revised paper is organized around three distinct questions:

1. Do the tested local signals produce useful allocation?
2. Can a different objective or a fixed allocation rule succeed under executable hard budgets?
3. Does any observed allocation benefit survive a total-cost comparison with dense-fewer-steps?

This separation replaces the previous mixture of workshop history, accounting terminology, and scale claims in the contribution paragraphs. The method section now defines soft-gate diagnostics, hard execution, and the shared continuation protocol before any results are presented. We also removed the old title and broad conclusion. The revised title is “When Is Token-Level Compute Allocation Learnable? Realized Updates, Frozen Rules, and the Dense-Training Boundary.”

### 2. Eq. (1), continuous gates, and actual FLOPs

Your criticism is correct when Eq. (1) is interpreted as realized computation. For

`x3 = LN(x2 + g * FFN(x2)),  g in [0,1]`,

a nonzero fractional `g` still requires the FFN forward and backward computation. Multiplying the output by 0.5 does not reduce its matrix-multiplication cost by 50%. The original proportional billing described a continuous relaxation, not executed savings. We now say this explicitly, and the soft-gate results are retained only as diagnostics of gate ordering and target content.

The new allocation experiments use a different operation. For each layer, a score selects an exact binary Top-B mask over valid, non-padding token rows. The implementation gathers the active rows, computes the FFN only on those rows, and scatters their outputs back with zero FFN contribution elsewhere. Padding is excluded and attention remains dense. Therefore:

- 50% activity means 50% of valid token-layer FFN rows execute.
- It does **not** mean a 50% reduction in total training FLOPs.
- No executable-savings claim is made from the value of a fractional gate.
- The new frozen-program training and ordinary online-router steps use this gather/scatter path. Periodic differentiable router updates are described and accounted separately.

We agree that the original Eq. (1) and billing sentence could not support the claimed realized savings. The revised claims instead depend on the hard-mask execution path.

### 3. The original conclusion was too broad; we tested better designs directly

We also agree with your interpretation that the original results might be specific to the architecture and learning design. A gate gradient at a fixed body measures local sensitivity under an infinitesimal linearization. It is not an oracle for a finite hard mask, nonlinear loss, joint training, or retraining. We removed the oracle and upper-bound language and no longer infer general non-learnability from that procedure.

More importantly, we ran two positive tests of the alternative you suggested.

**Realized-update routing.** A router is trained jointly with the body for 200 continuation steps. Every ten steps, it is updated by differentiating through the functional AdamW body update caused by its hard mask, using four subsequent ordinary training batches as the reward. On two confirmation bodies not used for meta-router development, final sparse losses are:

| Body | Meta-router | Random | Shuffled scores | Meta minus random |
|---|---:|---:|---:|---:|
| 8 | 1.2600 | 1.2967 | 1.2982 | -0.0366 |
| 9 | 1.2343 | 1.2979 | 1.3000 | -0.0636 |

The mean meta-minus-random difference is -0.0501 ± 0.0135 SE, and the mean meta-minus-shuffled difference is -0.0519 ± 0.0138 SE. The router is ahead of both controls at steps 50, 100, 150, and 200 in both bodies. This is direct evidence that a different co-adapted objective can learn useful allocation where the original local-gradient rule failed.

**A frozen allocation rule.** We also tested a simple rule that executes tokens with the smallest pre-FFN maximum absolute activation. After development selection, the unchanged rule beats exact-budget random in all eight body-budget regimes and at all 32 evaluations across four bodies and two activity budgets. Averaging first within each body, it improves final sparse loss by -0.0319 ± 0.0062 and answer accuracy by +3.37 ± 0.65 percentage points. A shuffled-score control degrades loss by 0.02530 on average, showing that the association between scores and token states matters.

These results do not establish that every adaptive-computation design works, and they do not show practical superiority over dense-fewer-steps. They do show that the original design-dependent failures should not have been generalized into a non-learnability claim.

### 4. Specific deployment and literature clarifications

**Dense inference.** The original random FFN-dropout experiments asked whether intermittent FFN updates could act as a training-time regularizer while restoring the full network at inference. That is a limited dense-deployment diagnostic, not an adaptive-inference method. We now label those tables as same-step quality comparisons, retain the unfavorable dense-fewer result, and make no inference-savings claim from them. For the new allocation experiments, sparse and dense evaluations are separate. In particular, the meta-router’s sparse gain does not improve dense-body quality: dense evaluation is worse than random by 0.0499 loss on average.

**MoD and recurrent adaptive computation.** Our numeric MoD baseline was withdrawn after we found a causality error in its implementation. We do not use that invalid comparison to argue against the published MoD result. The revision instead discusses MoD and Mixture-of-Recursions as positive evidence that adaptive computation can work under appropriate designs. Moosa et al. are also discussed: their finding that complexity-aligned allocation need not yield algorithmic generalization is consistent with our distinction between interpretable allocation structure and useful outcomes.

We would appreciate continuing the discussion on two specific points: does the hard Top-B execution description resolve the FLOP concern, and does the realized-update experiment address your hypothesis that a better learning design can make allocation useful? If the revised presentation remains unclear, a pointer to the first ambiguous passage would be especially useful.

## Response to Reviewer rfcv — pilot-independent draft

Thank you for identifying the exact gaps between the original experiments and its claims. We acknowledge all three questions directly: there is no formal nonlinear bound from the first-order procedure; the old starved-150M claim was unsupported and has been withdrawn; and the original efficiency comparisons did not consistently combine total cost, execution, and training behavior. The largest changes are a direct co-adaptation experiment, a new frozen allocation result, and a strict separation of learnability from practical efficiency.

### 1. What the first-order result does—and does not—establish

There is no formal argument that sorting gate gradients bounds nonlinear allocation or learning performance. The target

`-sum_l dL/dg_{t,l}` at `g=1`

is a local sensitivity at a fixed body. Sorting it is optimal only for the corresponding infinitesimal linearized problem. It does not bound the loss after a finite hard-mask change, another allocator, joint co-adaptation, or retraining. We removed the terms “oracle” and “upper bound” for this procedure and narrowed the conclusion to the tested surrogate and update rule.

We agree that adding seeds to that same 100-step allocator would only measure its failure more precisely; it would not answer your main objection. We therefore changed the allocation mechanism as well as the replication structure.

### 2. Direct test of co-adaptation through realized updates

The new online router and body train together for 200 updates from a shared mid-training checkpoint. All arms receive the same pregenerated ordinary training stream and exact executed FFN-row budget. The router uses current on-policy pre-FFN states and emits a hard 50% Top-B mask.

Every ten steps, we construct a functional AdamW update produced by that mask and evaluate the updated body on the next four ordinary training batches. The router is updated by descending the gradient of that realized post-update loss. Those reward batches are subsequently consumed by ordinary body training, so no additional examples are introduced; their forward evaluations and higher-order backward computation are auxiliary work. The actual body parameters and AdamW moments are installed after the functional update, and higher-order graphs are detached between steps.

This differs from the original experiment in the three ways your review identified: the body co-adapts, allocation is refreshed online rather than every 100 steps, and credit is based on a realized finite update rather than frozen-body gate sensitivity.

Final sparse-deployment results on two confirmation bodies not used for meta-router development are:

| Body | Meta-router | Random | Shuffled scores | Meta minus random | Meta minus shuffled |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 1.2600 | 1.2967 | 1.2982 | -0.0366 | -0.0382 |
| 9 | 1.2343 | 1.2979 | 1.3000 | -0.0636 | -0.0657 |
| Mean difference | — | — | — | -0.0501 ± 0.0135 SE | -0.0519 ± 0.0138 SE |

Meta-routing is ahead of both controls at steps 50, 100, 150, and 200 in each body. Thus the result is not a single favorable final evaluation. We interpret it as a scoped positive counterexample to the original broad non-learnability interpretation, not as evidence that all allocation schemes work.

### 3. Independent support from a cheap frozen allocation rule

We additionally tested whether a simple, reusable state-conditioned rule already contains useful allocation structure. The selected rule computes `-max_d |h_{t,l,d}|` from the current pre-FFN activation and executes exact Top-B rows. It requires no learned router, lookahead reward, gradient estimate, or dense FFN planning pass.

The rule was selected on development bodies and then frozen. Across four bodies and activity budgets of 37.5% and 50%, it wins all eight final comparisons against exact-budget random and all 32 evaluations at steps 50, 100, 150, and 200. Averaging first within body, the final loss difference is -0.0319 ± 0.0062 and the answer-accuracy difference is +3.37 ± 0.65 percentage points. Bodies 9 and 10 were fresh to program selection, but had appeared in earlier unrelated experiments; we describe them as fresh-to-program rather than globally untouched.

The shuffled control preserves each layer’s score multiset and active count but destroys the score/token association. It is worse than the frozen rule by 0.02530 loss on average. Strictly lagged gradient and Adam-directional rules also outperform random but remain behind low-max-abs in all four tested development regimes. These controls distinguish useful token association from the mere act of executing a fixed fraction of rows.

### 4. Deployment and total-cost accounting

We now keep three claims separate:

**Sparse-deployment learnability.** The comparisons above use equal updates and exact realized FFN-row budgets. They test whether allocation matters; they are not total-cost victories.

**Dense-body quality.** The meta-router’s advantage is specific to co-adapted sparse execution. Under dense evaluation it is worse than random by 0.0499 loss on average. Its sparse loss also remains 0.0240 above the fully dense same-step reference. We therefore do not claim that the router learns a better dense body.

**Practical total cost.** The original 1.4B counter counted attention once rather than once per layer; that total-FLOP interpretation is withdrawn. Our audit also found that the MPS counter omitted ordinary backward work, so old MPS-only totals are not used for practical claims. Revised practical comparisons use CPU-calibrated analytical forward-and-backward counts, attention per layer, and auxiliary work—including routing, reward evaluations, meta-updates, or averaging where applicable.

### 5. Gather/scatter validation and the throughput benchmark

We agree that the original text connected two paths that should have been distinguished: the legacy masked training driver and a separate gather microbenchmark.

The revised validation checks hard-mask/gather forward equivalence, exact rounded budgets and padding exclusion, functional AdamW against ordinary AdamW including moments, hypergradients against central finite differences, and sparse-update outputs, input gradients, and parameter gradients against autograd. New frozen-program training and ordinary online-router steps use executable gather/scatter. The periodic straight-through meta-update uses hard-output masking to preserve the router gradient and is distinguished from ordinary execution.

These checks establish numerical properties of the primitives used in the new experiments. We do **not** claim that they establish end-to-end equivalence of the legacy masked training driver and the separate benchmark. The 1.35x figure is now described only as a microbenchmark, not as an end-to-end training speedup.

### 6. Dense-fewer-steps and scale remain explicit limitations

We have not established a replicated regime in which any proposed recipe outperforms shorter dense training at matched realized cost. Under corrected accounting, a near-dense update-sparsity variant gave a 0.000366 loss advantage on one development body, but the frozen protocol lost by 0.003133 on the second body. The two-body mean difference is +0.0014 loss relative to dense. We report both outcomes and make no practical-superiority claim.

For scale, the original starved-150M causal claim was withdrawn because its driver ignored `--size`. None of the new positive claims depends on it. The new multi-body results establish replication within the tested 11M continuation setting, not second-scale generality. We will report any 150M continuation result separately with its exact protocol rather than treating it as validation of the invalid original arm.

We hope this separates the two claims that were previously conflated. Does the new realized-update, hard-mask evidence address your concern about learnability independently of practical efficiency? And if a soundness concern remains, would it be more useful to discuss the sparse co-adaptation boundary or the total-cost comparison first?

## Response to Reviewer 32DR — pilot-independent draft

Thank you for the direct questions. We acknowledge the three central concerns: the new mechanisms were not previously validated at a second scale, the paper did not justify the practical value of a recipe dominated by dense-fewer-steps, and the broad allocation-failure headline was incompatible with positive adaptive-computation results. We have narrowed the claims, withdrawn the affected comparisons, and added positive allocation experiments.

### 1. Practical applicability relative to dense-fewer-steps

The direct answer is that we have **not** demonstrated a replicated training regime in which the proposed recipe is preferable to dense-fewer-steps at matched total cost.

The closest result was a near-dense update-sparsity variant with dense forward computation and exponential weight averaging. It gave a 0.000366 loss advantage on development body 6, but the frozen protocol on body 7 lost by 0.003133. The mean difference across those two bodies is +0.0014 relative to dense. We report this failed replication rather than presenting the development point as a practical win.

Accordingly:

- Random token-level FFN dropout is retained as a diagnostic/training intervention, not recommended as a generally superior recipe.
- The gather result is described as a 1.35x microbenchmark, not as end-to-end acceleration of the legacy masked driver.
- The new random- and shuffled-control wins support allocation learnability only.
- Dense-fewer-steps is the principal practical baseline, and the abstract and discussion explicitly state that replicated practical superiority remains undemonstrated.

We agree that a paper whose only contribution were a dominated dropout recipe would have weak practical motivation. The revised contribution is instead the boundary between failed local allocation, successful co-adapted/frozen allocation, and the still-unresolved practical frontier.

### 2. MoD and the claim that allocation signals fail

We agree that MoD’s published success contradicts a universal statement that token-adaptive computation cannot work. It does not logically contradict the narrower observation that particular local signals and update rules failed in our testbed. Our original headline improperly conflated those statements.

We also found a causality error in our own MoD implementation. The numeric MoD comparison has therefore been withdrawn; we do not use it as evidence either for or against the published method. The revision discusses MoD, Mixture-of-Recursions, and recurrent dynamic-allocation work as positive results and removes the universal non-learnability claim.

Most importantly, we tested whether better designs could succeed in our own setting:

- An online router optimized through realized AdamW updates improves final sparse loss over random by -0.0366 and -0.0636 on two confirmation bodies, and also beats shuffled-score controls throughout training.
- A frozen low-max-abs allocation rule improves over exact-budget random in all eight body-budget regimes across four bodies and two budgets, with mean loss improvement -0.0319 ± 0.0062 and accuracy improvement +3.37 ± 0.65 percentage points.

These results change the paper’s conclusion from “allocation is not learnable” to “the original local procedures fail, but useful allocation is learnable under these specified alternative designs.” They still do not establish an advantage over dense-fewer-steps.

### 3. Experimental support, starved-150M, and scale

The revised online experiment uses two confirmation bodies not used for meta-router development. The frozen-rule experiment uses four bodies at two activity budgets, including fresh-to-program replications. Sparse and dense evaluations are reported separately, and the unfavorable dense-evaluation result is retained.

These are 11M continuation results. We do not describe the 1.4B three-thousand-step fine-tune as pretraining or claim that it validates the new mechanisms.

For the starved-150M arm, our audit found that the driver ignored the `--size` argument. The claimed replication was therefore unsupported, and we have withdrawn it rather than arguing about whether its gaps would survive additional noise. No revised positive claim relies on that experiment. Second-scale evidence for the new mechanisms remains a separate requirement. Any 150M continuation follow-up will be reported with its exact bodies, controls, continuation length, and held-out set, not presented as a repair of the invalid original causal arm.

We would value your view on which issue is most important for the remaining discussion: the absence of second-scale evidence for these new mechanisms, or the absence of a replicated dense-fewer advantage. We recognize that both remain limitations and do not want to conflate them.

## Later 150M follow-up — do not post until artifacts are complete

Use a short follow-up to Reviewer 32DR and Reviewer rfcv after the frozen pilot completes. Report both body-level paired differences, sparse accuracy, dense evaluation, and improvement from step 0. State all of the following limits regardless of outcome:

- 150M parameters, tagged-v2 arithmetic, sequence length 512.
- Fifty continuation steps from existing 4,000-step checkpoints.
- Two bodies, paired random controls, and a newly generated held-out set.
- Bodies are fresh to low-max-abs selection, not globally untouched.
- A positive result is a short-continuation scale signal, not a 200-step replication, natural-language result, dense-fewer victory, or practical-efficiency claim.
- If either final paired loss difference is nonnegative, state that the 11M allocation result did not pass this scale pilot; do not tune or extend the protocol afterward.

## Appendix: exact author-supplied review text

The following block preserves the exact lines supplied by the authors for reference while editing responses. It is not itself a response to post.

```text
Official Review of Submission52556 by Reviewer 32DR
Official Reviewby Reviewer 32DR21 Oct 2026, 11:31 (modified: 06 Nov 2026, 07:37)EveryoneRevisions
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
Official Review of Submission52556 by Reviewer dmAa
Official Reviewby Reviewer dmAa20 Oct 2026, 21:10 (modified: 06 Nov 2026, 07:37)EveryoneRevisions
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
Official Review of Submission52556 by Reviewer rfcv
Official Reviewby Reviewer rfcv18 Oct 2026, 01:27 (modified: 06 Nov 2026, 07:37)EveryoneRevisions
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
```
