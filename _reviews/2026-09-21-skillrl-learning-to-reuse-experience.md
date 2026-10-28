---
layout: post
title: "SkillRL: Learning to Reuse Experience"
date: 2026-09-21 00:00:00 +0900
description: "How distilled skills, supervised preparation, and failure-driven revision work together in agent reinforcement learning."
tags: ["agents","reinforcement-learning","skill-learning","memory"]
toc:
  sidebar: left
---

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/skillrl-learning-to-reuse-experience/2b8957ca77a5f9bd340caa28f1d2a24eef5c362c3dc6a68a5e16c3c2e1cdde30.png" class="img-fluid rounded z-depth-1" zoomable=true alt="Scattered experience sheets become organized skill cards in an editorial feedback-loop illustration." %}
  </div>
</div>

## Problem &amp; Background {#problem}

An agent can complete a task and still fail to become better at the next one. A successful interaction leaves behind observations, actions, detours, and an outcome. None of these automatically becomes a rule that another task can use. SkillRL addresses this gap by turning interaction histories into explicit behavioral guidance and training the agent to act with that guidance.

Consider a shopping agent that finds the right product but selects an option that changes its price. Retaining the entire session preserves the mistake, yet a later agent must still discover which detail mattered. A reusable lesson is more specific: after configuring the requested option, check the resulting price before committing. The useful unit of memory is the relation between a condition and an action, rather than every screen the agent visited.

| Approach | What it contributes | Remaining difficulty |
| --- | --- | --- |
| Reasoning and acting | Plans interleaved with environment feedback | A new episode need not retain previous lessons. |
| Reflection or experience memory | Past failures and solutions remain available | Retrieval can carry irrelevant steps and instance-specific details. |
| Compressed or procedural memory | Shorter, more reusable guidance | A frozen policy may not apply the guidance reliably. |
| Policy optimization | Successful behavior becomes more likely | Sparse outcomes do not explain a missing prerequisite. |
| SkillRL | A skill library and a trained skill-using policy | Both guidance quality and learning quality must be controlled. |

<div class="caption">Problem progression: these are overlapping design choices, not a claim that all prior systems store only raw trajectories.</div>

ReAct provides a reasoning-and-action interface; Reflexion adds feedback expressed in language. ExpeL and procedural-memory approaches already aim to extract useful experience, so abstraction itself is not an entirely new category. Memory-augmented learning also predates SkillRL. The distinctive combination here is outcome-aware distillation, a general/task-specific library, supervised training for skill use, and failure-driven library updates during reinforcement learning.

The resulting problem has two coupled parts. The memory designer must choose information that changes a decision under a limited context budget. The policy trainer must make the agent use that information in the right state. Better memory can fail with an unprepared policy, while a capable policy can be distracted by poor memory. SkillRL makes both parts explicit.

> ##### What “skill” means here
>
> A skill is a short textual strategy with an applicability condition. It is not a separately executable program or a formally learned temporal abstraction with its own termination rule. General means broadly applicable within an environment; it does not establish transfer across arbitrary domains.
{: .block-tip }

## Core Idea &amp; Method {#method}

The system maintains two kinds of learned state: model parameters and an external SkillBank. Qwen2.5-7B-Instruct supplies the base policy; OpenAI o3 acts as the teacher for distillation and supervised examples. The teacher changes the library through language generation, whereas GRPO updates the policy through reward-based optimization. “Recursive” describes the repeated feedback cycle between these two processes.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/skillrl-learning-to-reuse-experience/222609cda1e257c388ecfa089ae9f9c2e53cac1a4dfbcba7dd6022acafcebb5c.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Base rollouts feed teacher distillation and SkillBank. Cold-start SFT precedes skill-conditioned RL. Validation failures feed teacher revision and library updates." caption="Two coupled learning processes: reward updates the policy, and teacher analysis updates the external skills. Cold-start SFT is an initialization stage." %}
  </div>
</div>

1. Collect successful and unsuccessful base-agent trajectories in the target environment. Failure is retained as evidence about missing constraints or bad action ordering.
2. Distill successful decisions into reusable patterns. For failures, identify the breakdown, the mistaken action or assumption, a proposed correction, and its applicability.
3. Organize general guidance and category-specific guidance in SkillBank. Store a short name, a principle, and a condition for use.
4. Generate skill-augmented demonstrations with the teacher and perform cold-start supervised fine-tuning. The resulting policy also becomes the fixed KL reference for RL.
5. Retrieve relevant skills, sample groups of trajectories, and update the policy with GRPO. At validation checkpoints, analyze failures and expand or refine the library.

Retrieval combines always-included general skills with a bounded set of specialized skills. A task embedding is compared with skill embeddings; candidates above a similarity threshold are ranked, and at most six specialized skills are selected in the reported setup. The notation below separates the retrieval threshold from the evolution threshold because they have different meanings.

$$
\mathcal{S}_{\mathrm{ret}}=\operatorname{TopK}_{K_r}\{s\in\mathcal{S}_{k}:\operatorname{sim}(e_d,e_s)>\delta_{\mathrm{ret}}\}
$$

$$
a_t\sim\pi_\theta(a_t\mid o_{\le t},d,\mathcal{S}_g,\mathcal{S}_{\mathrm{ret}})
$$

Here d is the task description, k its task category, and the two embeddings represent the task and a candidate skill. The general set is not counted inside the top-six specialized retrieval cap. Consequently, growing the general library can still enlarge prompts even when specialized retrieval remains bounded. The text does not fully specify category assignment or the embedding implementation.

> ##### Illustrative application: a changing price
>
> Suppose a product initially displays \\\$38, but choosing the required size raises the price to \\\$52 under a \\\$50 budget. “This product was cheap” is an instance-bound memory. “Verify the final configured price before purchase” remains useful on another product. These numbers are an explanatory example, not an experimental observation.
{: .block-tip }

The applicability field matters as much as the principle. A price-check rule applies after configuration; executing it only before choosing the size does not prevent this failure. Similarly, household tasks can require locating an object, changing its state, and then placing it. A general instruction to deliver a held object must be interpreted together with the task-specific requirement to clean or heat it first. A library needs coherent precedence, not merely more entries.

Cold-start SFT teaches this conditional use through demonstrations. Its loss is ordinary cross-entropy on teacher-generated trajectories conditioned on the task and skills. This is a substantial source of supervision: 7,500 examples for ALFWorld and 2,400 for WebShop are reported. The RL stage therefore does not begin with a base model that has only received a helpful prompt.

$$
\theta_{\mathrm{sft}}=\arg\min_\theta\mathcal{L}_{\mathrm{CE}}(\mathcal{D}_{\mathrm{SFT}};\theta),\qquad\pi_{\mathrm{ref}}=\pi_{\theta_{\mathrm{sft}}}
$$

For GRPO, several complete trajectories are sampled for the same task and skill context. Their relative rewards supply an advantage without a separately trained critic. At the mathematical level described for task success, rewards are binary. The appendix additionally lists an invalid-action penalty, so a reproduction must establish how that penalty enters the implemented reward.

$$
A_i=\frac{R_i-\overline R}{\operatorname{std}(R_1,\ldots,R_G)},\qquad\rho_i=\frac{\pi_\theta(\tau_i\mid d,\mathcal{S}_g,\mathcal{S}_{\mathrm{ret}})}{\pi_{\mathrm{old}}(\tau_i\mid d,\mathcal{S}_g,\mathcal{S}_{\mathrm{ret}})}
$$

$$
\begin{aligned}\mathcal{J}(\theta)=\mathbb{E}\bigg[\frac1G\sum_{i=1}^{G}&\min\big(\rho_iA_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\big)\\&-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\bigg]\end{aligned}
$$

For an illustrative four-trajectory group with rewards 1, 1, 0, 0, the mean is 0.5. Using population standard deviation gives 0.5, hence advantages 1, 1, -1, -1. Successful trajectories become relatively more likely. If all rewards are equal, the written denominator is zero; numerical stabilization or a skip rule is necessary in implementation. The equation alone does not specify that choice.

Clipping limits the incentive to change likelihood ratios too far in one update. The KL term anchors the agent to the skill-using SFT policy. Neither mechanism validates the contents of a newly generated skill. Policy stability and correctness of external guidance remain separate concerns.

At evolution checkpoints, validation failures are grouped and sampled to expose gaps in current guidance. The teacher proposes additions or refinements. The main description targets categories with success below a threshold, while the appendix gives failure-analysis budgets both below and above 0.4; this discrepancy matters for exact reproduction. The reported schedule checks every five steps and allows at most three new skills per evolution.

$$
\mathcal{S}_{\mathrm{new}}=\mathcal{M}_T(\mathcal{T}_{\mathrm{val}}^{-},\mathcal{S}),\qquad\mathcal{S}\leftarrow\mathcal{S}\cup\mathcal{S}_{\mathrm{new}}
$$

This update is an outer-loop knowledge-editing operation, not backpropagation through the teacher or the skill text. The union notation makes addition clear but leaves replacement, deduplication, conflict resolution, and retirement less explicit. A deployed version would need those rules as the library becomes larger.

## Evidence &amp; Results {#evidence}

The experiments cover two interactive environments and seven search-augmented QA benchmarks. ALFWorld evaluates household tasks through text commands; WebShop evaluates simulated shopping constraints. The QA policy is trained on NQ and HotpotQA, while five other datasets probe transfer within search-based question answering. These settings test useful agent behaviors, but do not represent unrestricted computer use or physical-world robotics.

| Setting | Reported configuration |
| --- | --- |
| Student / teacher | Qwen2.5-7B-Instruct / OpenAI o3 |
| Cold-start SFT | 3 epochs; learning rate 0.0001; batch 16 |
| RL | Learning rate 0.000001; group size 8; KL coefficient 0.01 |
| Batch reporting | Main text: 16 with 4 accumulation steps; appendix: 64 |
| Token limits | Prompt 6,000; response 1,024 |
| Compute | 8 H100 80GB GPUs; approximately 30 hours per experiment |

<div class="caption">Reported training configuration. The batch descriptions are compatible with an effective batch of 64, but that interpretation needs implementation confirmation.</div>

The strongest clean headline is the reported success rate in each interactive benchmark: 89.9% for ALFWorld and 72.7% for WebShop. Relative to the listed GRPO baseline, the absolute differences are 12.3 and 6.6 percentage points. The ALFWorld relative increase is about 15.9%, which illustrates why a percentage increase and a percentage-point difference should not be interchanged.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/skillrl-learning-to-reuse-experience/c68441e95befee31afa02bda764cf34f18c9e3461b50b98d0386d56d4a01920c.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="ALFWorld success is 77.6 for GRPO, 84.4 for static SkillRL, and 89.9 for full SkillRL. WebShop values are 66.1, 70.3, and 72.7 percent." caption="Reported success rates on a shared 0–100 scale. Static SkillRL means the variant without dynamic skill evolution." %}
  </div>
</div>

| Method | ALFWorld success | WebShop success |
| --- | --- | --- |
| GRPO | 77\.6% | 66\.1% |
| Mem0 + GRPO | 54\.7% | 37\.5% |
| SimpleMem + GRPO | 62\.5% | 46\.9% |
| SkillRL | 89\.9% | 72\.7% |

<div class="caption">Selected reported comparisons. Adding a memory system to GRPO does not automatically improve this evaluation.</div>

ALFWorld gains are not uniform across tasks. Against GRPO, Cool improves from 72.5 to 95.5, Pick2 from 64.7 to 87.5, and Heat from 74.7 to 90.0. These differences are 23.0, 22.8, and 15.3 points. They fit the hypothesis that explicit ordering and state checks help multistep tasks, but task-level scores alone do not identify the causal mechanism.

| Removed component | ALFWorld drop | WebShop drop |
| --- | --- | --- |
| General layer of hierarchy | 13\.1 pp | 11\.3 pp |
| Skill abstraction; use raw trajectories | 28\.2 pp | 22\.5 pp |
| Cold-start SFT | 24\.7 pp | 26\.2 pp |
| Dynamic evolution | 5\.5 pp | 2\.4 pp |

<div class="caption">Absolute drops calculated from the full-system scores and the ablation table. The largest drop differs by environment.</div>

The ablations favor the entire construction, especially abstraction and supervised preparation. A static skill system already reaches 84.4% and 70.3%, so the recursive update provides an additional gain rather than explaining all of the improvement. These interventions are not additive: removing SFT changes the policy that uses the library, while removing abstraction changes its input distribution.

| QA benchmark | SkillRL | EvolveR |
| --- | --- | --- |
| NQ (training domain) | 45\.9 | 43\.5 |
| HotpotQA (training domain) | 43\.2 | 38\.2 |
| TriviaQA | 63\.3 | 63\.4 |
| PopQA | 45\.9 | 44\.6 |
| 2Wiki | 40\.3 | 42\.0 |
| MuSiQue | 20\.2 | 15\.6 |
| Bamboogle | 73\.8 | 54\.4 |
| Reported average | 47\.1 | 43\.1 |

<div class="caption">QA scores as reported. The printed average is 47.1, but the seven SkillRL entries have a simple mean of 47.51; this discrepancy is unresolved.</div>

SkillRL has the best reported complete seven-task average, but its printed average does not equal the simple mean of the listed entries. This is too large a difference to attribute to one-decimal rounding. It is also not best on every dataset. EvolveR is ahead on TriviaQA and 2Wiki, ZeroSearch reaches 51.5 on PopQA versus 45.9, and StepSearch reaches 22.6 on MuSiQue versus 20.2. Bamboogle supplies a particularly large 19.4-point advantage over EvolveR. Thus, the transfer result is heterogeneous rather than universal.

> ##### Two different efficiency quantities
>
> The method description claims 10–20-fold compression of trajectories into skills. The prompt-length analysis reports roughly 1,450 tokens for raw memory versus fewer than 1,300 for SkillRL, about a 10.3% reduction. Compressing one memory component does not imply a 10–20-fold reduction in total inference context, latency, or cost.
{: .block-tip }

The illustrated library grows from 55 to 100 skills by training step 150. The reported breakdown moves from 12 to 20 general skills and 43 to 80 task-specific skills. Learning curves also favor evolution, but their plotted trajectories do not substitute for repeated-seed confidence intervals. The reported compute budget includes a 24-hour RL stage and does not establish an equal-cost comparison across every baseline.

## Limitations &amp; Confidence {#limitations}

Confidence is strongest in the table-level claim: this particular training system reports substantially better interactive-task performance than its listed comparators. Confidence is lower in explanations that attribute the entire advantage to one mechanism, or in claims of open-ended generality. The evaluation combines teacher supervision, skill representation, retrieval, and parameter learning.

| Issue | What it limits |
| --- | --- |
| Teacher and data confounds | Same base model and optimizer do not imply matched supervision or compute. |
| One-at-a-time ablations | They identify sensitivity within the pipeline, not independent causal contributions. |
| Uncertainty reporting | Main tables provide no repeated-seed spread or confidence intervals. |
| Adaptive validation | Failure-driven updates consume validation information; an untouched test split is essential. |
| Retrieval and update details | Threshold use, embedding choice, and edit semantics require clarification. |
| Growing library | No demonstrated long-horizon guarantee against stale or conflicting guidance. |

<div class="caption">Interpretive boundaries and reproduction questions, rather than additional experimental findings.</div>

The comparison with larger closed models is interesting but task-specific. A specialized 7B agent trained with teacher-generated demonstrations and environment feedback is compared with prompted general models. This supports specialization under these conditions; it does not establish a general replacement for the larger models or eliminate dependence on a powerful teacher.

Failure analysis also needs epistemic care. An unsuccessful episode tells us that the observed action sequence failed, but not necessarily which alternative would have succeeded. Teacher-written corrections are hypotheses. Without replay or controlled counterfactual testing, a plausible explanation can introduce a misleading rule. Improvements in aggregate reward do not prove every generated lesson is valid.

Several reporting details deserve explicit caution. The appendix gives failure budgets for categories above 0.4 despite the main-text below-threshold trigger. Raw-trajectory ablation losses are 28.2 and 22.5 points, so an approximate narrative claim of a maximum 25-point loss does not match both table values. The illustrative library chart also contains a mistakes category, making its fine-grained visual grouping less straightforward than the two-part textual breakdown.

Validation-guided evolution is a form of training information use. That is not automatically test leakage, but it means validation performance cannot be treated as an independent final estimate. To establish generalization, all adaptive skill edits and checkpoint choices should be separated from a held-out evaluation. Search-domain transfer is evidence of breadth within QA, not a demonstration of transfer from shopping to household manipulation.

> ##### Calibrated takeaway
>
> The evidence supports a useful joint design: compact conditional guidance, an agent trained to use it, and targeted revision. It does not yet establish unlimited self-improvement, universal cross-domain skills, or a net cost advantage after teacher calls and all training expenses.
{: .block-tip }

## Reusable Research Insights {#insights}

The transferable research idea is to treat experience representation as part of the learning system. A memory can be judged by whether it changes the right decision under a bounded context budget. That shifts evaluation from “How much history was saved?” toward “Which errors does this representation prevent, and under what conditions?”

One useful design pattern is to factor a rule into a trigger, an action, and an expected consequence. The trigger prevents guidance from being applied everywhere; the action makes it operational; the consequence gives an evaluator something to test. SkillRL explicitly stores applicability and a principle. Adding a verifiable postcondition would be a research extension, not an already evaluated feature.

| Research question | Controlled comparison | Failure signal |
| --- | --- | --- |
| Does abstraction help beyond brevity? | Skills vs length-matched summaries and raw excerpts | Gain disappears when token budget is matched. |
| Does SFT teach skill use? | Matched SFT with useful, shuffled, or absent skills | Performance ignores relevance of supplied skills. |
| Does evolution beat extra teacher help? | Static and evolving banks with equal teacher budgets | Benefit vanishes under matched supervision. |
| Does retrieval select causally useful guidance? | Retrieved vs random vs oracle category-matched skills | Semantically similar skills trigger wrong actions. |
| Does growth remain useful? | Add-only vs merge/prune policies at fixed context | More entries reduce accuracy or raise token cost. |

<div class="caption">Proposed experiments for separating mechanisms; these are not reported results.</div>

A particularly informative experiment would separate acquisition from utilization. Hold the policy fixed and compare two libraries to estimate the immediate value of their guidance. Then hold the library fixed and train two policies with matched data budgets to estimate skill-use learning. Finally, let both evolve. This factorial structure can reveal interaction effects that one-at-a-time removal does not identify.

The GRPO example exposes another research opportunity. A group in which all trajectories fail has no relative success signal under binary rewards. Failure explanations may still suggest a useful precondition, so the outer language-based loop can supply information when the inner reward-based loop is uninformative. This is a plausible complementarity, not proof that the generated precondition is correct. Testing the proposed repair in the environment would strengthen the learning signal.

Evaluation should account for resources across the whole pipeline. Measure successful held-out tasks, environment interactions, teacher tokens, student tokens, retrieval latency, and total elapsed time. A reduction in retrieved-context size is only one component. If the teacher repeatedly rewrites a library, its cost can dominate the savings from a shorter prompt.

1. Start with an immutable held-out task set and a versioned initial library. Record category coverage and the budget for collecting experience.
2. Compare compact skills against equally short controls under the same policy and retrieval budget. This isolates representation more directly.
3. Train skill-using policies with matched demonstrations and environment interactions. Evaluate both relevant and deliberately mismatched guidance.
4. Enable failure-driven revision with a fixed teacher budget. Log each proposed change, the failure it addresses, and a replay test where possible.
5. Test on held-out instances and categories, report uncertainty, and measure whether new skills harm earlier capabilities.

For an agent builder, the practical lesson is to keep experience, a proposed rule, and evidence of the rule’s usefulness distinct. For a researcher, it is to study the interface between nonparametric knowledge and parametric learning. SkillRL offers a concrete system in which that interface matters, and a useful starting point for experiments that test why it works.

## Related Reading {#related}

These brief comparisons connect SkillRL to earlier experience-based agents and a later skill-evolution approach. They are reading pointers, not full independent reviews.

ExpeL: LLM Agents Are Experiential Learners — extracts natural-language insights from accumulated task experience and retrieves those insights without updating model parameters. Read it to separate the value of experiential knowledge from SkillRL’s additional supervised and reinforcement learning. Paper: <https://arxiv.org/abs/2308.10144>

EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle — alternates offline distillation of reusable strategic principles with online retrieval and interaction, and improves the policy through reinforcement learning. Read it as a close predecessor to compare the experience lifecycle with SkillRL’s hierarchical SkillBank and failure-driven updates. Paper: <https://arxiv.org/abs/2510.16079v1>

WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution — separates execution experience, accumulated knowledge, and skills through a persistent wiki. This later work offers a useful comparison between maintaining an intermediate knowledge layer and SkillRL’s joint evolution of a skill library and policy; it is not a predecessor evaluated in SkillRL. Paper: <https://arxiv.org/abs/2608.27454v1>

Main paper: SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning. <https://arxiv.org/abs/2602.08234v1>
