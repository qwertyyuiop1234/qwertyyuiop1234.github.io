---
layout: post
title: "WikiSkill: Learning from Agent Experience"
date: 2026-09-21 00:00:00 +0900
description: "How persistent knowledge turns execution traces and failed interventions into better reusable skills."
tags: ["agent-skills","memory","skill-evolution"]
toc:
  sidebar: left
---

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/b68c8b1eae351270d1e86967473664df9577b814fbef32cae9090a99d079cc1a.png" class="img-fluid rounded z-depth-1" zoomable=true alt="An abstract engraved landscape of scattered experience becoming organized layers of reusable knowledge." %}
  </div>
</div>

## Problem &amp; Background {#problem}

An agent can finish a task yet fail to learn a reusable lesson from it. Its execution trace records what happened, but a future run does not automatically inherit a reliable procedure. WikiSkill addresses the additional problem that the process of improving those procedures can itself forget: why a change was proposed, why it failed, and which recurring failure it was meant to fix.

A skill packages procedural knowledge in a filesystem directory, usually with a SKILL.md containing applicability conditions and instructions. It changes the agent’s available guidance without changing model weights. Manual skills make expertise reusable, but their author must anticipate the relevant failure modes. Experience-driven skill evolution instead runs tasks, studies trajectories, proposes skill edits, and evaluates the edits. This turns observed successes and failures into candidate instructions.

| Earlier approach | What it preserves or improves | Remaining distinction |
| --- | --- | --- |
| Manual skills | Reusable workflows written in advance | Unanticipated failures still require diagnosis and maintenance. |
| EvoSkill | Candidate skill search and a history of proposal outcomes | A feedback history is not a separately maintained catalog of mechanisms and evidence. |
| Trace2Skill | Lessons from individual trajectories, merged into skill patches | Consolidation produces procedures without a distinct evolving knowledge layer. |
| SkillOpt | Reflections, patch synthesis, rejected-edit feedback and meta guidance | Optimization artifacts do not constitute the persistent structured wiki studied here. |

<div class="caption">The relevant contrast is how knowledge is organized, not whether earlier methods remember anything.</div>

Consider a household-task agent that repeatedly takes an object, examines it, and returns it to its starting location. An optimizer may propose “stay goal-directed.” If validation rejects this advice and the rejection becomes just another buried log entry, a later optimizer can rediscover the same vague fix. Conversely, putting every observation directly into the executable skill risks a long, contradictory procedure. Raw evidence, evolving explanations, and instructions have different jobs.

WikiSkill asks whether an explicit intermediate knowledge representation can make later skill edits better informed. The proposed wiki preserves patterns across iterations, including supporting traces and rejected interventions. The skill becomes a selected operational consequence of that knowledge, rather than the only durable location where learning can survive. The contribution is architectural and empirical: no new model-training objective is introduced, and it is not a solution to selecting the right skill from a large library.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/5cb0ac06d63f0152cd39969e1275a91f48a45742976497bdd1c7f870710896b5.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Three layers separate immutable evidence, editable persistent knowledge, and reversible executable skills." caption="Raw traces explain what happened; the wiki explains recurring patterns; skills tell the executor what to do. These layers have different retention policies." %}
  </div>
</div>

## Core Idea &amp; Method {#method}

The evolving state contains a skill set S and a wiki W. Both begin empty. Training, validation and test tasks are disjoint: training trajectories supply diagnostic experience, validation chooses whether a candidate is accepted, and held-out tests measure final performance. The inference agent, wiki maintainer and skill proposer are functional roles around an LLM; they need not imply different model families.

| Layer | Stored information | Update policy |
| --- | --- | --- |
| Raw | Step-by-step observations, actions, tool outputs and answers | Write once; original evidence remains available. |
| Wiki | Pattern pages, index, chronological log and skill-impact history | Incrementally refined; retained even when a skill edit fails. |
| Skills | SKILL.md procedure plus PURPOSE.md linking it to motivating patterns | Candidate changes survive only after validation improvement. |

<div class="caption">PURPOSE.md preserves the reason for a procedure; skill-impact.md records what was tried and its measured outcome.</div>

1. Execute training tasks using the current skills. Full skill content is injected into the system prompt; the default executor cannot consult the wiki during these rollouts.
2. Consolidate evidence. The maintainer receives the current wiki and up to eight sampled traces: at most five failures and three successes, each capped at 15,000 characters. It diagnoses causes and patches existing patterns or adds new ones.
3. Propose one targeted intervention. A ReAct-style proposer consults the wiki index, impact history and task-outcome summary, then reads selected pattern pages and raw traces. It creates or patches one skill. The published prompt requires at least four trace reads before a change.
4. Evaluate the candidate on validation tasks. Keep it only if its score strictly exceeds the best accepted score; otherwise restore the previous skill set.
5. Always append the proposal diff, score and decision to the wiki. A rejected procedure therefore becomes evidence for the next attempt.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/558f394b16fc32ef4efd32009d968674b348e349ef728243009081fad05895f3.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Training execution flows into wiki consolidation, one skill proposal and validation. Rejection returns to the prior skill but preserves accumulated wiki knowledge." caption="The asymmetric loop is the central mechanism: knowledge survives failure while active behavior can be rolled back." %}
  </div>
</div>

The maintainer and proposer see experience differently. The maintainer compiles a small stratified sample into a coherent knowledge base. The proposer starts from an index and actively fetches evidence, rather than receiving all trajectories in a single prompt. Incremental patch operations reduce unnecessary rewriting. The impact tracker is updated by the outer orchestration code after evaluation, so acceptance outcomes are not merely the model’s recollection of its own success.

$$
W'_k=\operatorname{Maintain}(W_{k-1},\mathcal{T}_{\mathrm{sample},k}),\qquad S'_k=\operatorname{Apply}(S_{k-1},P_k)
$$

$$
S_k=\begin{cases}S'_k,&R_{\mathrm{val}}(S'_k)>R_{\mathrm{best}},\\ S_{k-1},&\text{otherwise}.\end{cases}
$$

Here k indexes iterations, P is the proposed edit, and R is average task score. The best score starts at the empty-skill validation baseline and updates only on acceptance. Equal scores are rejected. Reaching validation score 1.0 stops evolution early. In both branches, the intermediate wiki is updated with the proposal and result to become the next persistent wiki.

> ##### Worked intuition: a failed experiment still changes knowledge
>
> Suppose the best validation score is 0.70. A new skill scoring 0.70 is rejected, as is one scoring 0.65; a candidate scoring 0.75 is accepted. These are illustrative numbers, not paper measurements. All three outcomes can still refine the wiki. Rollback means reverting executable instructions, not erasing the experiment.
{: .block-tip }

The ALFWorld case study makes this concrete. At iteration 0, a broad goal-directed-action skill is rejected. The persistent record exposes that failure to the next proposer. At iteration 1, a more concrete break-repetition-loop skill is accepted; subsequent evidence of multi-operation loops motivates another accepted refinement at iteration 4. The paper’s simplified figure contrasts vague goal-directed advice with explicit action patterns that stop repetition.

These domain-specific rules should not be copied as universal laws. “Never return an item to its origin” can be wrong for another task. The transferable lesson is to identify the state transition that causes a failure, state the intervention concretely, and validate its scope. PURPOSE.md and pattern evidence make that reasoning inspectable.

> ##### Why not let the executor read the entire wiki?
>
> The authors hypothesize that direct wiki access lets an executor solve tasks using knowledge that never enters the skill. Training traces then become less diagnostic of skill quality. This explanation is plausible and consistent with their ablation, but it is a hypothesis about the mechanism, not a proven general prohibition on memory access.
{: .block-tip }

## Evidence &amp; Results {#evidence}

The evaluation covers five task families and five inference models. Each method starts with an empty skill set and injects evolved skills into the executor’s prompt, holding skill retrieval out of the comparison. Reported test scores average three independent runs of the full evolution process. Paired bootstrap testing uses 1,000 resamples at p \< 0.05; cross-benchmark summaries give each benchmark equal weight.

| Benchmark | Train / validation / test | What is tested |
| --- | --- | --- |
| LiveMath | 35 / 18 / 124 | Single-step mathematical reasoning; no tools. |
| SealQA | 16 / 10 / 85 | Search-assisted factual answers. |
| SpreadSheet | 80 / 40 / 280 | Spreadsheet transformations through Python/bash. |
| OfficeQA | 50 / 24 / 172 | Treasury-document reasoning with local search tools. |
| ALFWorld | 39 / 18 / 134 | Text-action household tasks in a simulator. |

<div class="caption">OfficeQA starts with pre-parsed oracle reference pages as evidence while allowing searches over full files; this is not unrestricted document discovery from scratch.</div>

| Inference model | No skill | Best other method | WikiSkill | Gain vs no skill |
| --- | --- | --- | --- | --- |
| Qwen-3.5-4B | 26\.2 | 35\.2 · SkillOpt | 38\.5 | \+12.3 pp |
| Qwen-3.5-9B | 29\.9 | 42\.3 · EvoSkill | 47\.4 | \+17.5 pp |
| Qwen-3.6-27B | 39\.4 | 53\.3 · EvoSkill | 63\.3 | \+23.9 pp |
| Gemma-4-31B | 41\.3 | 49\.1 · SkillOpt | 54\.9 | \+13.6 pp |
| Gemini-3.5-Flash | 49\.5 | 56\.1 · EvoSkill | 68\.1 | \+18.6 pp |

<div class="caption">Five-benchmark macro-average accuracy (&#37;) from Table 1. Gains are percentage points, not relative percentage improvements; cells are rounded as reported.</div>

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/fea8de063f22bd26908c878c11264a1c54ba67b8808bff6c7cfaed3a0150fff5.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Paired horizontal bars compare no-skill and WikiSkill average accuracy across the five models, with a zero-based accuracy axis." caption="WikiSkill improves every model’s average, but an average does not imply improvement on every dataset." %}
  </div>
</div>

The gains can be large: Qwen-3.6-27B rises from 40.8% to 81.7% on SpreadSheet, while Gemini-3.5-Flash rises from 33.0% to 72.6% on LiveMath. Yet Qwen-3.5-4B falls from 30.2% to 28.5% on OfficeQA. WikiSkill is also not the numerical winner in every comparison with another evolution method: Qwen-3.5-9B obtains 56.3% on LiveMath versus EvoSkill’s 58.1%. Statistical ties in the original table matter; these examples should not be read as independently established significant differences.

Within the reported Qwen models, gains increase with size, and the 9B model with skills averages 47.4%, above the 27B no-skill result of 39.4%. This demonstrates a useful interaction between model capability and procedural guidance in this setup. It is not a controlled scaling law: the compared models span versions as well as parameter counts, and the comparison does not establish equal total compute or deployment cost.

| Executor wiki access | Proposer wiki access | Four-task average |
| --- | --- | --- |
| No skills | Not applicable | 40\.4 |
| Yes | No | 45\.3 |
| No | No | 48\.7 |
| Yes | Yes | 60\.9 |
| No | Yes — default | 63\.7 |

<div class="caption">Gemini-3.5-Flash ablation, Table 3. ALFWorld is excluded, so this average is not the 68.1&#37; five-task result above.</div>

With executor wiki access disabled, enabling proposer wiki access raises the average from 48.7% to 63.7%, a 15.0-point difference. However, the no-proposer-wiki condition also removes the maintainer. This supports the value of the combined persistent-knowledge pathway; it does not isolate persistence, structured representation, and additional maintenance computation individually. Giving both roles wiki access lowers the average to 60.9%, although SpreadSheet improves from 76.6% to 80.2%. The direction depends on the task.

Cross-model transfer separates finding a useful procedure from executing it. Qwen-3.5-9B reaches 50.5% on SpreadSheet using 27B-evolved skills, versus 33.6% with its own skills and 24.3% with none. Transfer can also be harmful: Gemini-3.5-Flash falls from 50.5% to 18.1% using 4B-evolved spreadsheet skills, while 27B-evolved skills yield 63.4%. The authors attribute this negative transfer to low-level workarounds and fragmented diagnostic steps that constrain stronger models or consume their interaction budget.

> ##### A ceiling can prevent learning before test performance is perfect
>
> Gemini-3.5-Flash scores 100% on the small ALFWorld validation split before evolution. All methods stop without evolving a skill, but its test score is only 85.9%. Thus this row is not evidence that learned skills reach 85.9%; it illustrates how validation saturation can suppress further search.
{: .block-tip }

$$
C_{\mathrm{opt}}=(1+T_{\mathrm{ReAct}})\frac{N_{\mathrm{train}}}{B};\qquad B=N_{\mathrm{train}}\implies C_{\mathrm{opt}}=1+T_{\mathrm{ReAct}}
$$

The appendix reports roughly 10–20 proposer turns per iteration. Its constant-in-training-size optimizer call count applies to full-batch operation and excludes training/validation rollouts. It also does not make tokens per call, wiki size, latency, or monetary cost constant. Therefore the complexity statement cannot establish that WikiSkill is the cheapest method.

## Limitations &amp; Confidence {#limits}

The evidence is strongest for a bounded offline claim: under the tested harnesses, a persistent wiki plus a wiki-informed proposer yields better average evolved-skill performance than the selected baselines. The paper is a preprint, and this brief analyzes reported experiments rather than independently reproducing them. Long-running autonomous improvement and general production reliability remain broader claims than these results support.

| Boundary | Why it matters | What would strengthen confidence |
| --- | --- | --- |
| Small validation sets | Repeated selection may reward noise; saturation can stop search early. | Larger held-out validation, repeated candidate scoring, final untouched tests. |
| Strict improvement gate | Neutral edits that enable later gains are rejected. | Compare plateau-tolerant search at matched evaluation budgets. |
| Growing wiki | There is no automated pruning; old or mistaken patterns may accumulate. | Longer runs with contradiction resolution, evidence aging and token accounting. |
| Full skill injection | Retrieval, applicability and routing failures are excluded. | Evaluate many-skill libraries with realistic triggering. |
| Limited task horizon | Hours-long runs and within-rollout learning are not evaluated. | Long-horizon online tests with recovery and resource budgets. |

<div class="caption">These are research boundaries, not reasons to dismiss the measured gains.</div>

The bootstrap tests and three evolution runs improve robustness, but they do not eliminate adaptive overfitting to validation. For a binary task with ten validation examples, one changed answer shifts the score by ten percentage points. That illustrative granularity is relevant to SealQA’s validation size, not an estimate of its actual uncertainty. An observed “best” score can reflect both real improvement and stochastic variation.

The wiki is retained, not guaranteed correct or literally immutable. Pattern pages can be revised, while flawed diagnoses can also persist. The paper supplies trace-level evidence and objective proposal outcomes, but does not demonstrate automatic truth maintenance under adversarial traces, changing tools, or distribution shifts. Treating retention as guaranteed compounding knowledge would overstate what is established.

A further design question is which component earns the benefit. A persistence-only ablation, a flat-memory baseline with matched token budget, and an equally interactive proposer without structured patterns would help distinguish representation from compute and retrieval strategy. Those are proposed follow-up experiments, not results reported by WikiSkill. Likewise, cross-model negative transfer argues for target-model evaluation before reusing another agent’s skills.

## Reusable Research Insights {#insights}

The most reusable idea is to separate experimental evidence, evolving hypotheses and deployed behavior. When all three live in one prompt, rejecting a procedure can erase its rationale, and retaining its rationale can burden execution. Three layers let a system remember failed experiments without executing their failed instructions. This pattern can apply to tool workflows, code-repair recipes, data-cleaning procedures and other tasks with inspectable outcomes.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/469258d4af9019cfa7337b791916b530509a9e3f7ace04b87697d734339664b6.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="A research loop preserves observation and hypothesis while gating interventions, then tests transfer under new conditions." caption="Reusable reasoning pattern: record evidence, state a causal hypothesis, intervene narrowly, validate, and carry the lesson forward." %}
  </div>
</div>

| Reusable technique | When useful | Assumption or caution |
| --- | --- | --- |
| Preserve rejected interventions | Search repeatedly revisits plausible but unhelpful fixes. | A past rejection may cease to apply after the model or environment changes. |
| Patch one skill at a time | You need an interpretable link between edits and outcomes. | Interactions between skills can still confound attribution. |
| Compare passing and failing traces | An error message alone does not reveal the cause. | Samples must cover important failure modes rather than only easy cases. |
| Separate discovery from execution | A strong procedure may outlive its originating model. | Transfer must be measured on the target model and tool budget. |
| Constrain diagnostic access | You want to know what the deployable artifact actually enables. | Restrictions should match the intended deployment question. |

<div class="caption">Each technique is a testable design choice; none guarantees improvement by itself.</div>

A practical pattern record can contain a trigger, observed action sequence, hypothesized cause, supporting successes and failures, proposed fix, and counterexamples. A skill should then express only the operational procedure and its applicability conditions. For example, “avoid loops” is too vague; “after an operation, check whether the target state changed before repeating it” exposes a measurable decision point. This is an illustrative generalization, not the exact ALFWorld rule evaluated in the paper.

The gate also provides a reusable experimental unit: hold the previous accepted artifact fixed, modify one component, evaluate the candidate, and preserve the diff with its score. If adapting this to a noisy task, estimate uncertainty and reserve an untouched test set. A stricter numerical gate alone does not prevent false acceptance after many comparisons. If adapting it to changing tools, attach the tool/model version and applicability conditions to remembered rejections so history does not become dogma.

$$
\Delta_{A\rightarrow B}=R_B(S_A)-R_B(\emptyset)
$$

This diagnostic notation is introduced here to clarify transfer, not as a new formula claimed by the paper. Measure both this gain and performance with the target’s self-evolved skill. Positive transfer answers whether the imported procedure helps; beating self-evolution answers a different question. Reporting only the strongest source-target pair hides harmful transfers such as the spreadsheet example.

A promising follow-up study would hold proposer turns, wiki token budget and task splits constant while varying persistent structured memory, persistent flat history and reset memory. Another would test whether neutral-but-shorter skill edits improve later evolution under a fixed context budget. Both turn a broad enthusiasm for “memory” into specific hypotheses about what information is stored, who reads it, and how its utility is measured.

## Related Reading {#related}

These short comparisons identify the closest methodological neighbors through WikiSkill’s related-work discussion and baseline appendix. They are reading pointers, not full independent reviews.

EvoSkill: Automated Skill Discovery for Multi-Agent Systems — explores candidate skill programs using rollout feedback and a frontier of evaluated candidates. Read it to compare search over candidates with WikiSkill’s explicit persistent pattern layer. Paper: <https://arxiv.org/abs/2603.02766>

Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills — analyzes individual successful and failed trajectories, then hierarchically merges patches. Read it to contrast systematic trace-level consolidation with cross-iteration wiki maintenance. Paper: <https://arxiv.org/abs/2603.25158>

SkillOpt: Executive Strategy for Self-Evolving Agent Skills — organizes optimization into rollout, reflection, aggregation, selection, update and evaluation. Read it to compare reflection-driven patch synthesis with the wiki-informed proposer and retained rejection history. Paper: <https://arxiv.org/abs/2605.23904>

Main paper: WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution. <https://arxiv.org/abs/2608.27454v1>
