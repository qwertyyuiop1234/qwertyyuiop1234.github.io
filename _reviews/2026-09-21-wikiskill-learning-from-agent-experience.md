---
layout: post
title: "WikiSkill: Learning from Agent Experience"
date: 2026-09-21 00:00:00 +0900
description: "How a persistent knowledge layer turns execution traces and failed interventions into better reusable skills."
tags: ["agent-skills","memory","skill-evolution"]
toc:
  sidebar: left
---

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/b68c8b1eae351270d1e86967473664df9577b814fbef32cae9090a99d079cc1a.png" class="img-fluid rounded z-depth-1" zoomable=true alt="An abstract engraved landscape of scattered experience becoming organized layers of reusable knowledge." %}
  </div>
</div>

## Problem and background {#problem}

An agent can finish a task without learning anything reusable from it. Its execution trace records what happened, but a later run does not automatically inherit a reliable procedure. WikiSkill also targets a second problem: the process that improves those procedures can itself forget things, such as why a change was proposed, why it failed, and which recurring failure it was supposed to fix.

A skill packages procedural knowledge in a filesystem directory, usually with a SKILL.md file that holds applicability conditions and instructions. It changes the guidance available to the agent without changing model weights. Manual skills make expertise reusable, but whoever writes them has to anticipate the relevant failure modes. Experience-driven skill evolution takes a different route. It runs tasks, studies the trajectories, proposes skill edits and evaluates them, so that observed successes and failures become candidate instructions.

| Earlier approach | What it preserves or improves | Remaining distinction |
| --- | --- | --- |
| Manual skills | Reusable workflows written in advance | Unanticipated failures still require diagnosis and maintenance. |
| EvoSkill | Candidate skill search and a history of proposal outcomes | A feedback history is not a separately maintained catalog of mechanisms and evidence. |
| Trace2Skill | Lessons from individual trajectories, merged into skill patches | Consolidation produces procedures without a distinct evolving knowledge layer. |
| SkillOpt | Reflections, patch synthesis, rejected-edit feedback and meta guidance | Optimization artifacts are not the persistent structured wiki studied here. |

<div class="caption">The relevant contrast is how knowledge is organized, not whether earlier methods remember anything.</div>

Consider a household-task agent that keeps picking up an object, examining it, and putting it back where it started. An optimizer might propose "stay goal-directed." If validation rejects that advice and the rejection ends up as one more buried log entry, a later optimizer can rediscover the same vague fix. Writing every observation straight into the executable skill has the opposite risk: a long, contradictory procedure. Raw evidence, evolving explanations and instructions each have a different job.

WikiSkill asks whether an explicit intermediate knowledge representation can make later skill edits better informed. The proposed wiki keeps patterns across iterations, together with the traces that support them and the interventions that were rejected. The skill then holds a selected operational consequence of that knowledge and is no longer the only durable place where learning can survive. The contribution is architectural and empirical. The paper introduces no new model-training objective, and it does not address how to select the right skill from a large library.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/5cb0ac06d63f0152cd39969e1275a91f48a45742976497bdd1c7f870710896b5.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Three layers separate immutable evidence, editable persistent knowledge, and reversible executable skills." caption="Raw traces explain what happened, the wiki explains recurring patterns, and skills tell the executor what to do. Each layer has its own retention policy." %}
  </div>
</div>

## Core idea and method {#method}

The evolving state is a skill set S and a wiki W, and both start empty. Training, validation and test tasks are disjoint. Training trajectories supply diagnostic experience, validation decides whether a candidate is accepted, and held-out tests measure final performance. The inference agent, wiki maintainer and skill proposer are functional roles built around an LLM. Having three roles does not imply three different model families.

| Layer | Stored information | Update policy |
| --- | --- | --- |
| Raw | Step-by-step observations, actions, tool outputs and answers | Write once; original evidence remains available. |
| Wiki | Pattern pages, index, chronological log and skill-impact history | Incrementally refined; retained even when a skill edit fails. |
| Skills | SKILL.md procedure plus PURPOSE.md linking it to motivating patterns | Candidate changes survive only after validation improvement. |

<div class="caption">PURPOSE.md preserves the reason for a procedure; skill-impact.md records what was tried and its measured outcome.</div>

1. Run the training tasks with the current skills. The full skill content goes into the system prompt, and by default the executor cannot consult the wiki during these rollouts.
2. Consolidate the evidence. The maintainer gets the current wiki and up to eight sampled traces (at most five failures and three successes), each capped at 15,000 characters. It diagnoses causes, then patches existing patterns or adds new ones.
3. Propose one targeted intervention. A ReAct-style proposer looks at the wiki index, the impact history and a task-outcome summary, then reads the pattern pages and raw traces it picks. It creates or patches one skill. The published prompt requires at least four trace reads before any change.
4. Score the candidate on the validation tasks. Keep it only if its score is strictly higher than the best accepted score. Otherwise, restore the previous skill set.
5. Append the proposal diff, score and decision to the wiki every time. A rejected procedure is then evidence for the next attempt.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/558f394b16fc32ef4efd32009d968674b348e349ef728243009081fad05895f3.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Training execution flows into wiki consolidation, one skill proposal and validation. Rejection returns to the prior skill but preserves accumulated wiki knowledge." caption="The main mechanism is an asymmetric loop. Knowledge survives a failure, but active behavior can be rolled back." %}
  </div>
</div>

The maintainer and the proposer see experience differently. The maintainer compiles a small stratified sample into a coherent knowledge base. The proposer starts from an index and fetches evidence on demand instead of receiving every trajectory in a single prompt. Patch operations are incremental, which cuts down on unnecessary rewriting. The outer orchestration code updates the impact tracker after evaluation, so acceptance outcomes are not just the model's recollection of its own success.

$$
W'_k=\operatorname{Maintain}(W_{k-1},\mathcal{T}_{\mathrm{sample},k}),\qquad S'_k=\operatorname{Apply}(S_{k-1},P_k)
$$

$$
S_k=\begin{cases}S'_k,&R_{\mathrm{val}}(S'_k)>R_{\mathrm{best}},\\ S_{k-1},&\text{otherwise}.\end{cases}
$$

Here k indexes iterations, P is the proposed edit, and R is the average task score. The best score starts at the empty-skill validation baseline and changes only when a candidate is accepted. A candidate that only ties the best score is rejected. Evolution stops early if the validation score reaches 1.0. In both branches, the proposal and its result are added to the intermediate wiki, which becomes the next persistent wiki.

> ##### Worked example: a failed experiment still changes knowledge
>
> Suppose the best validation score is 0.70. A new skill scoring 0.70 is rejected, and so is one scoring 0.65. A candidate scoring 0.75 is accepted. These are illustrative numbers, not paper measurements. All three outcomes can still refine the wiki. A rollback reverts the executable instructions but keeps the record of the experiment.
{: .block-tip }

The ALFWorld case study is a concrete example. At iteration 0, a broad goal-directed-action skill is rejected, and the persistent record shows that failure to the next proposer. At iteration 1, a more concrete break-repetition-loop skill is accepted. Later evidence of multi-operation loops leads to another accepted refinement at iteration 4. A simplified figure in the paper contrasts the vague goal-directed advice with explicit action patterns that stop repetition.

These domain-specific rules should not be copied as universal laws. "Never return an item to its origin" can be wrong for another task. The lesson that transfers is to identify the state transition that causes a failure, state the intervention concretely, and validate its scope. PURPOSE.md and the pattern evidence make that reasoning inspectable.

> ##### Why not let the executor read the entire wiki?
>
> The authors hypothesize that direct wiki access lets an executor solve tasks with knowledge that never enters the skill, which makes training traces less diagnostic of skill quality. The explanation is plausible and consistent with their ablation. It is still a hypothesis about the mechanism, not a proven general rule against memory access.
{: .block-tip }

## Evidence and results {#evidence}

The evaluation covers five task families and five inference models. Every method starts from an empty skill set and injects the evolved skills into the executor's prompt, so skill retrieval is held out of the comparison. Each reported test score is the average of three independent runs of the full evolution process. Paired bootstrap tests use 1,000 resamples at p \< 0.05, and cross-benchmark summaries weight each benchmark equally.

| Benchmark | Train / validation / test | What is tested |
| --- | --- | --- |
| LiveMath | 35 / 18 / 124 | Single-step mathematical reasoning; no tools. |
| SealQA | 16 / 10 / 85 | Search-assisted factual answers. |
| SpreadSheet | 80 / 40 / 280 | Spreadsheet transformations through Python/bash. |
| OfficeQA | 50 / 24 / 172 | Treasury-document reasoning with local search tools. |
| ALFWorld | 39 / 18 / 134 | Text-action household tasks in a simulator. |

<div class="caption">OfficeQA starts with pre-parsed oracle reference pages as evidence and allows searches over the full files. This is not unrestricted document discovery from scratch.</div>

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
    {% include figure.liquid loading="lazy" path="assets/img/reviews/wikiskill-learning-from-agent-experience/fea8de063f22bd26908c878c11264a1c54ba67b8808bff6c7cfaed3a0150fff5.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Paired horizontal bars compare no-skill and WikiSkill average accuracy across the five models, with a zero-based accuracy axis." caption="WikiSkill improves every model&#39;s average, but a better average does not mean improvement on every dataset." %}
  </div>
</div>

Some gains are large. Qwen-3.6-27B rises from 40.8% to 81.7% on SpreadSheet, and Gemini-3.5-Flash rises from 33.0% to 72.6% on LiveMath. But Qwen-3.5-4B falls from 30.2% to 28.5% on OfficeQA. WikiSkill also does not have the top number in every comparison with another evolution method. On LiveMath, Qwen-3.5-9B gets 56.3% with WikiSkill and 58.1% with EvoSkill. The original table includes statistical ties, so these examples should not be read as independently established significant differences.

Among the reported Qwen models, gains grow with size. With skills, the 9B model averages 47.4%, which is above the 27B model's no-skill result of 39.4%. This shows a useful interaction between model capability and procedural guidance in this setup. It is not a controlled scaling law: the compared models differ in version as well as parameter count, and the comparison does not establish equal total compute or deployment cost.

| Executor wiki access | Proposer wiki access | Four-task average |
| --- | --- | --- |
| No skills | Not applicable | 40\.4 |
| Yes | No | 45\.3 |
| No | No | 48\.7 |
| Yes | Yes | 60\.9 |
| No | Yes (default) | 63\.7 |

<div class="caption">Gemini-3.5-Flash ablation, Table 3. ALFWorld is excluded, so this average is not the 68.1&#37; five-task result above.</div>

With executor wiki access disabled, enabling proposer wiki access raises the average from 48.7% to 63.7%, a 15.0-point difference. The no-proposer-wiki condition, however, also removes the maintainer. The result supports the combined persistent-knowledge pathway, but it does not isolate persistence, structured representation and the extra maintenance computation from one another. Giving both roles wiki access lowers the average to 60.9%, although SpreadSheet improves from 76.6% to 80.2%. The direction depends on the task.

Cross-model transfer separates finding a useful procedure from executing it. On SpreadSheet, Qwen-3.5-9B reaches 50.5% with 27B-evolved skills, compared with 33.6% with its own skills and 24.3% with none. Transfer can also hurt. Gemini-3.5-Flash falls from 50.5% to 18.1% with 4B-evolved spreadsheet skills, while 27B-evolved skills give it 63.4%. The authors attribute this negative transfer to low-level workarounds and fragmented diagnostic steps that constrain stronger models or use up their interaction budget.

> ##### A ceiling can prevent learning before test performance is perfect
>
> Gemini-3.5-Flash scores 100% on the small ALFWorld validation split before any evolution. Every method stops without evolving a skill, yet its test score is only 85.9%. So this row is not evidence that learned skills reach 85.9%. It shows how a saturated validation set can cut off further search.
{: .block-tip }

$$
C_{\mathrm{opt}}=(1+T_{\mathrm{ReAct}})\frac{N_{\mathrm{train}}}{B};\qquad B=N_{\mathrm{train}}\implies C_{\mathrm{opt}}=1+T_{\mathrm{ReAct}}
$$

The appendix reports roughly 10 to 20 proposer turns per iteration. Its claim that the optimizer call count stays constant as training size grows holds for full-batch operation only, and the count excludes training and validation rollouts. A constant call count also does not keep tokens per call, wiki size, latency or monetary cost constant. The complexity statement therefore cannot establish that WikiSkill is the cheapest method.

## Limitations and confidence {#limits}

The evidence is strongest for a bounded, offline claim. Under the tested harnesses, a persistent wiki plus a wiki-informed proposer gives better average evolved-skill performance than the selected baselines. The paper is a preprint, and this review analyzes the reported experiments rather than reproducing them independently. Long-running autonomous improvement and general production reliability are broader claims than these results support.

| Boundary | Why it matters | What would strengthen confidence |
| --- | --- | --- |
| Small validation sets | Repeated selection may reward noise; saturation can stop search early. | Larger held-out validation, repeated candidate scoring, final untouched tests. |
| Strict improvement gate | Neutral edits that enable later gains are rejected. | Compare plateau-tolerant search at matched evaluation budgets. |
| Growing wiki | There is no automated pruning; old or mistaken patterns may accumulate. | Longer runs with contradiction resolution, evidence aging and token accounting. |
| Full skill injection | Retrieval, applicability and routing failures are excluded. | Evaluate many-skill libraries with realistic triggering. |
| Limited task horizon | Hours-long runs and within-rollout learning are not evaluated. | Long-horizon online tests with recovery and resource budgets. |

<div class="caption">These are research boundaries, not reasons to dismiss the measured gains.</div>

The bootstrap tests and the three evolution runs make the results more robust, but they do not rule out adaptive overfitting to validation. On a binary task with ten validation examples, one changed answer moves the score by ten percentage points. That illustrative granularity is relevant at SealQA's validation size; it is not an estimate of SealQA's actual uncertainty. An observed "best" score can mix real improvement with stochastic variation.

The wiki is retained, but it is not guaranteed to be correct, and it is not literally immutable. Pattern pages can be revised, and flawed diagnoses can also persist. The paper supplies trace-level evidence and objective proposal outcomes. It does not demonstrate automatic truth maintenance under adversarial traces, changing tools or distribution shifts. Treating retention as guaranteed compounding knowledge would overstate what has been established.

Another open design question is which component produces the benefit. Three experiments would help separate representation from compute and retrieval strategy: a persistence-only ablation, a flat-memory baseline with a matched token budget, and an equally interactive proposer without structured patterns. These are proposed follow-up experiments, not results reported by WikiSkill. The negative cross-model transfer also argues for evaluating on the target model before reusing another agent's skills.

## Reusable research insights {#insights}

The most reusable idea is the separation of experimental evidence, evolving hypotheses and deployed behavior. When all three live in one prompt, rejecting a procedure can erase its rationale, and keeping the rationale can weigh down execution. With three layers, a system can remember failed experiments without executing their failed instructions. The same pattern can apply to tool workflows, code-repair recipes, data-cleaning procedures and other tasks with inspectable outcomes.

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

A practical pattern record can contain a trigger, the observed action sequence, a hypothesized cause, supporting successes and failures, a proposed fix, and counterexamples. The skill should then express only the operational procedure and its applicability conditions. For example, "avoid loops" is too vague, while "after an operation, check whether the target state changed before repeating it" gives a decision point that can be measured. This is an illustrative generalization, not the exact ALFWorld rule evaluated in the paper.

The gate is also a reusable experimental unit. Hold the previous accepted artifact fixed, modify one component, evaluate the candidate, and keep the diff with its score. To adapt this to a noisy task, estimate uncertainty and reserve an untouched test set. A stricter numerical gate on its own does not prevent false acceptance after many comparisons. To adapt it to tools that change, attach the tool/model version and the applicability conditions to each remembered rejection, so that an old rejection is not applied after it stops being valid.

$$
\Delta_{A\rightarrow B}=R_B(S_A)-R_B(\emptyset)
$$

The diagnostic notation above is introduced here to clarify transfer; it is not a new formula claimed by the paper. Measure both this gain and the target's performance with its own self-evolved skill. A positive transfer gain says whether the imported procedure helps. Whether it beats self-evolution is a separate question. Reporting only the strongest source-target pair hides harmful transfers such as the spreadsheet example.

One promising follow-up study would hold proposer turns, wiki token budget and task splits constant while varying the memory: persistent structured memory, persistent flat history, or memory that resets. Another would test whether skill edits that are neutral but shorter improve later evolution under a fixed context budget. Both would turn a broad enthusiasm for "memory" into specific hypotheses about what information is stored, who reads it, and how its usefulness is measured.

## Related reading {#related}

The short comparisons below cover the closest methodological neighbors, as identified in WikiSkill's related-work discussion and baseline appendix. They are reading pointers, not full independent reviews.

EvoSkill: Automated Skill Discovery for Multi-Agent Systems. The paper explores candidate skill programs using rollout feedback and a frontier of evaluated candidates. Read it to compare search over candidates with WikiSkill's explicit persistent pattern layer. Paper: <https://arxiv.org/abs/2603.02766>

Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills. It analyzes individual successful and failed trajectories, then merges patches hierarchically. Read it to contrast systematic trace-level consolidation with WikiSkill's cross-iteration wiki maintenance. Paper: <https://arxiv.org/abs/2603.25158>

SkillOpt: Executive Strategy for Self-Evolving Agent Skills. The paper organizes optimization into rollout, reflection, aggregation, selection, update and evaluation. Read it to compare reflection-driven patch synthesis with WikiSkill's wiki-informed proposer and retained rejection history. Paper: <https://arxiv.org/abs/2605.23904>

Main paper: WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution. <https://arxiv.org/abs/2608.27454v1>
