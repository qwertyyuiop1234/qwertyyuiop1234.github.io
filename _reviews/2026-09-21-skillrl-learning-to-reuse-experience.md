---
layout: post
title: "SkillRL: Learning to Reuse Experience"
date: 2026-09-21 00:00:00 +0900
description: "How SkillRL combines distilled skills, supervised fine-tuning, and failure-driven revision in agent reinforcement learning."
tags: ["agents","reinforcement-learning","skill-learning","memory"]
toc:
  sidebar: left
---

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/skillrl-learning-to-reuse-experience/2b8957ca77a5f9bd340caa28f1d2a24eef5c362c3dc6a68a5e16c3c2e1cdde30.png" class="img-fluid rounded z-depth-1" zoomable=true alt="Scattered experience sheets become organized skill cards in an editorial feedback-loop illustration." %}
  </div>
</div>

## Problem and background {#problem}

An agent can finish a task and be no better at the next one. A successful interaction leaves behind observations, actions, detours, and an outcome, but none of them automatically becomes a rule that another task can use. SkillRL converts interaction histories into explicit behavioral guidance and trains the agent to act on that guidance.

Take a shopping agent that finds the right product but picks an option that changes the price. Storing the whole session keeps the mistake, and a later agent still has to work out which detail mattered. A reusable lesson is more specific: after setting the requested option, check the resulting price before committing. What is worth remembering is the relation between a condition and an action, not every screen the agent visited.

| Approach | What it contributes | Remaining difficulty |
| --- | --- | --- |
| Reasoning and acting | Plans interleaved with environment feedback | A new episode need not retain previous lessons. |
| Reflection or experience memory | Past failures and solutions remain available | Retrieval can carry irrelevant steps and instance-specific details. |
| Compressed or procedural memory | Shorter, more reusable guidance | A frozen policy may not apply the guidance reliably. |
| Policy optimization | Successful behavior becomes more likely | Sparse outcomes do not explain a missing prerequisite. |
| SkillRL | A skill library and a trained skill-using policy | Both guidance quality and learning quality must be controlled. |

<div class="caption">Problem progression: these are overlapping design choices, not a claim that all prior systems store only raw trajectories.</div>

ReAct provides an interface that interleaves reasoning and action, and Reflexion adds feedback in natural language. ExpeL and procedural-memory approaches already try to extract useful experience, so abstraction by itself is not an entirely new category. Memory-augmented learning also predates SkillRL. What distinguishes SkillRL is the combination of outcome-aware distillation, a library with general and task-specific skills, supervised training for skill use, and failure-driven library updates during reinforcement learning.

The problem therefore has two coupled parts. The memory has to hold information that changes a decision while fitting in a limited context budget. The policy has to be trained to use that information in the right state. Better memory can still fail if the policy is not prepared for it, and poor memory can distract a capable policy. SkillRL makes both parts explicit.

> ##### What "skill" means here
>
> A skill is a short text strategy with a condition for when it applies. It is not a separately executable program, and it is not a formally learned temporal abstraction with its own termination rule. "General" means broadly applicable within an environment; calling a skill general does not establish transfer across arbitrary domains.
{: .block-note }

## Core idea and method {#method}

The system keeps two kinds of learned state: model parameters and an external SkillBank. Qwen2.5-7B-Instruct is the base policy. OpenAI o3 is the teacher, used for distillation and for generating supervised examples. The teacher changes the library by generating language, while GRPO updates the policy through reward-based optimization. "Recursive" refers to the repeated feedback cycle between these two processes.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/skillrl-learning-to-reuse-experience/222609cda1e257c388ecfa089ae9f9c2e53cac1a4dfbcba7dd6022acafcebb5c.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Base rollouts feed teacher distillation and SkillBank. Cold-start SFT precedes skill-conditioned RL. Validation failures feed teacher revision and library updates." caption="Two coupled learning processes: reward updates the policy, and teacher analysis updates the external skills. Cold-start SFT is an initialization stage." %}
  </div>
</div>

1. Collect successful and unsuccessful base-agent trajectories in the target environment. Failed trajectories are kept as evidence of missing constraints or bad action ordering.
2. Distill successful decisions into reusable patterns. For failures, identify where things broke down, the mistaken action or assumption, a proposed correction, and when the correction applies.
3. Organize general and category-specific guidance in SkillBank. Each entry has a short name, a principle, and a condition for use.
4. Have the teacher generate skill-augmented demonstrations, then run cold-start supervised fine-tuning. The resulting policy is also the fixed KL reference for RL.
5. Retrieve relevant skills, sample groups of trajectories, and update the policy with GRPO. At validation checkpoints, analyze failures and add to or refine the library.

General skills are always included in the prompt; specialized skills are retrieved up to a fixed limit. The task embedding is compared with skill embeddings, candidates above a similarity threshold are ranked, and the reported setup selects at most six specialized skills. The notation below uses separate symbols for the retrieval threshold and the evolution threshold because they mean different things.

$$
\mathcal{S}_{\mathrm{ret}}=\operatorname{TopK}_{K_r}\{s\in\mathcal{S}_{k}:\operatorname{sim}(e_d,e_s)>\delta_{\mathrm{ret}}\}
$$

$$
a_t\sim\pi_\theta(a_t\mid o_{\le t},d,\mathcal{S}_g,\mathcal{S}_{\mathrm{ret}})
$$

Here d is the task description, k is its task category, and the two embeddings represent the task and a candidate skill. General skills do not count toward the top-six cap on specialized retrieval. A growing general library can therefore still lengthen prompts while specialized retrieval stays bounded. The paper does not fully specify how tasks are assigned to categories or how the embeddings are implemented.

> ##### Illustrative example: a changing price
>
> Suppose a product first shows \\\$38, but choosing the required size raises the price to \\\$52, and the budget is \\\$50. "This product was cheap" is a memory tied to one instance. "Verify the final configured price before purchase" is still useful for a different product. These numbers are for explanation only; they are not experimental observations.
{: .block-note }

The applicability field matters as much as the principle. A price check has to happen after configuration; running it only before the size is chosen does not prevent this failure. Household tasks have the same issue. They can require finding an object, changing its state, and then placing it, so a general instruction to deliver a held object has to be read together with the task-specific requirement to clean or heat it first. More entries are not enough on their own; the library also needs a consistent order of precedence.

Cold-start SFT teaches this conditional use from demonstrations. Its loss is ordinary cross-entropy on teacher-generated trajectories, conditioned on the task and skills. The amount of supervision is substantial: the paper reports 7,500 examples for ALFWorld and 2,400 for WebShop. RL therefore does not start from a base model that has only been given a helpful prompt.

$$
\theta_{\mathrm{sft}}=\arg\min_\theta\mathcal{L}_{\mathrm{CE}}(\mathcal{D}_{\mathrm{SFT}};\theta),\qquad\pi_{\mathrm{ref}}=\pi_{\theta_{\mathrm{sft}}}
$$

For GRPO, several complete trajectories are sampled for the same task and skill context. Each trajectory's advantage comes from its reward relative to the others in the group, so no separately trained critic is needed. In the mathematical formulation given for task success, rewards are binary. The appendix also lists an invalid-action penalty, and a reproduction would need to establish how that penalty enters the implemented reward.

$$
A_i=\frac{R_i-\overline R}{\operatorname{std}(R_1,\ldots,R_G)},\qquad\rho_i=\frac{\pi_\theta(\tau_i\mid d,\mathcal{S}_g,\mathcal{S}_{\mathrm{ret}})}{\pi_{\mathrm{old}}(\tau_i\mid d,\mathcal{S}_g,\mathcal{S}_{\mathrm{ret}})}
$$

$$
\begin{aligned}\mathcal{J}(\theta)=\mathbb{E}\bigg[\frac1G\sum_{i=1}^{G}&\min\big(\rho_iA_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\big)\\&-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\bigg]\end{aligned}
$$

As an illustration, take a group of four trajectories with rewards 1, 1, 0, 0. The mean is 0.5, and the population standard deviation is also 0.5, so the advantages are 1, 1, -1, -1. The successful trajectories become relatively more likely. If all rewards are equal, the denominator as written is zero, and an implementation needs numerical stabilization or a rule to skip the group. The equation alone does not specify which.

Clipping limits the incentive to push likelihood ratios too far in a single update. The KL term keeps the agent close to the skill-using SFT policy. Neither mechanism checks the contents of a newly generated skill. A stable policy and correct external guidance are separate concerns.

At evolution checkpoints, validation failures are grouped and sampled to find gaps in the current guidance, and the teacher proposes additions or refinements. According to the main description, this targets categories whose success rate is below a threshold. The appendix, however, gives failure-analysis budgets both below and above 0.4, and that discrepancy matters for exact reproduction. The reported schedule checks every five steps and allows at most three new skills per evolution.

$$
\mathcal{S}_{\mathrm{new}}=\mathcal{M}_T(\mathcal{T}_{\mathrm{val}}^{-},\mathcal{S}),\qquad\mathcal{S}\leftarrow\mathcal{S}\cup\mathcal{S}_{\mathrm{new}}
$$

This update edits external knowledge in an outer loop. It does not backpropagate through the teacher or the skill text. The union notation makes addition clear but says less about replacement, deduplication, conflict resolution, and retiring old skills. A deployed version would need rules for these as the library grows.

## Evidence and results {#evidence}

The experiments cover two interactive environments and seven search-augmented QA benchmarks. ALFWorld tests household tasks issued through text commands, and WebShop tests shopping under constraints in a simulated store. The QA policy is trained on NQ and HotpotQA; five other datasets probe transfer within search-based question answering. These settings test useful agent behaviors, but they do not represent unrestricted computer use or physical-world robotics.

| Setting | Reported configuration |
| --- | --- |
| Student / teacher | Qwen2.5-7B-Instruct / OpenAI o3 |
| Cold-start SFT | 3 epochs; learning rate 0.0001; batch 16 |
| RL | Learning rate 0.000001; group size 8; KL coefficient 0.01 |
| Batch reporting | Main text: 16 with 4 accumulation steps; appendix: 64 |
| Token limits | Prompt 6,000; response 1,024 |
| Compute | 8 H100 80GB GPUs; approximately 30 hours per experiment |

<div class="caption">Reported training configuration. The batch descriptions are compatible with an effective batch of 64, but that interpretation needs implementation confirmation.</div>

The cleanest headline result is the reported success rate on each interactive benchmark: 89.9% on ALFWorld and 72.7% on WebShop. Against the listed GRPO baseline, the absolute differences are 12.3 and 6.6 percentage points. The relative increase on ALFWorld is about 15.9%. The two figures show why a percentage increase and a percentage-point difference should not be used interchangeably.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/skillrl-learning-to-reuse-experience/c68441e95befee31afa02bda764cf34f18c9e3461b50b98d0386d56d4a01920c.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="ALFWorld success is 77.6 for GRPO, 84.4 for static SkillRL, and 89.9 for full SkillRL. WebShop values are 66.1, 70.3, and 72.7 percent." caption="Reported success rates on a shared 0-100 scale. Static SkillRL means the variant without dynamic skill evolution." %}
  </div>
</div>

| Method | ALFWorld success | WebShop success |
| --- | --- | --- |
| GRPO | 77\.6% | 66\.1% |
| Mem0 + GRPO | 54\.7% | 37\.5% |
| SimpleMem + GRPO | 62\.5% | 46\.9% |
| SkillRL | 89\.9% | 72\.7% |

<div class="caption">Selected reported comparisons. Adding a memory system to GRPO does not automatically improve this evaluation.</div>

The ALFWorld gains are not uniform across tasks. Compared with GRPO, Cool improves from 72.5 to 95.5, Pick2 from 64.7 to 87.5, and Heat from 74.7 to 90.0, differences of 23.0, 22.8, and 15.3 points. This fits the hypothesis that explicit ordering and state checks help on multistep tasks, but task-level scores alone do not identify the causal mechanism.

| Removed component | ALFWorld drop | WebShop drop |
| --- | --- | --- |
| General layer of hierarchy | 13\.1 pp | 11\.3 pp |
| Skill abstraction; use raw trajectories | 28\.2 pp | 22\.5 pp |
| Cold-start SFT | 24\.7 pp | 26\.2 pp |
| Dynamic evolution | 5\.5 pp | 2\.4 pp |

<div class="caption">Absolute drops calculated from the full-system scores and the ablation table. The largest drop differs by environment.</div>

The ablations support the whole design, abstraction and supervised preparation in particular. A static skill system already reaches 84.4% and 70.3%, so the recursive update adds to the gain without explaining all of the improvement. The interventions are not additive. Removing SFT changes the policy that uses the library, while removing abstraction changes the input distribution that policy sees.

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

SkillRL has the best reported average over all seven tasks, but the printed average does not equal the simple mean of the listed entries, and the gap is too large to be explained by one-decimal rounding. SkillRL is also not the best on every dataset. EvolveR is ahead on TriviaQA and 2Wiki, ZeroSearch reaches 51.5 on PopQA versus 45.9, and StepSearch reaches 22.6 on MuSiQue versus 20.2. On Bamboogle the margin over EvolveR is especially large: 19.4 points. Across these datasets, then, the transfer result is heterogeneous rather than universal.

> ##### Two different efficiency quantities
>
> The method description claims that turning trajectories into skills compresses them 10 to 20-fold. The prompt-length analysis reports roughly 1,450 tokens for raw memory versus fewer than 1,300 for SkillRL, a reduction of about 10.3%. Compressing one memory component does not imply a 10 to 20-fold reduction in total inference context, latency, or cost.
{: .block-note }

In the illustration, the library grows from 55 to 100 skills by training step 150. In the reported breakdown, general skills go from 12 to 20 and task-specific skills from 43 to 80. The learning curves also favor evolution, but plotted curves are not a substitute for confidence intervals over repeated seeds. The reported compute budget includes a 24-hour RL stage and does not show that every baseline was compared at equal cost.

## Limitations and confidence {#limitations}

The table-level claim deserves the most confidence: this particular training system reports substantially better interactive-task performance than the comparators it lists. Explanations that credit the entire advantage to one mechanism deserve less, as do claims of open-ended generality. The evaluation combines teacher supervision, skill representation, retrieval, and parameter learning.

| Issue | What it limits |
| --- | --- |
| Teacher and data confounds | Same base model and optimizer do not imply matched supervision or compute. |
| One-at-a-time ablations | They identify sensitivity within the pipeline, not independent causal contributions. |
| Uncertainty reporting | Main tables provide no repeated-seed spread or confidence intervals. |
| Adaptive validation | Failure-driven updates consume validation information; an untouched test split is essential. |
| Retrieval and update details | Threshold use, embedding choice, and edit semantics require clarification. |
| Growing library | No demonstrated long-horizon guarantee against stale or conflicting guidance. |

<div class="caption">Interpretive boundaries and reproduction questions, rather than additional experimental findings.</div>

The comparison with larger closed models is interesting but task-specific. It sets a specialized 7B agent, trained with teacher-generated demonstrations and environment feedback, against prompted general models. That supports specialization under these conditions. It does not establish the 7B agent as a general replacement for the larger models, and it does not remove the dependence on a powerful teacher.

Failure analysis also needs epistemic care. A failed episode shows that the observed action sequence did not work, but not necessarily which alternative would have succeeded. Corrections written by the teacher are hypotheses. Without replay or controlled counterfactual testing, a plausible explanation can introduce a misleading rule. A rise in aggregate reward does not prove that every generated lesson is valid.

Several reporting details call for caution. The appendix gives failure budgets for categories above 0.4, even though the main text describes a trigger for categories below the threshold. The raw-trajectory ablation losses are 28.2 and 22.5 points, so an approximate narrative claim of a maximum 25-point loss does not match both table values. The illustrative library chart also includes a mistakes category, so its fine-grained visual grouping is less straightforward than the two-part breakdown in the text.

Evolution guided by validation failures uses validation data as training information. That is not automatically test leakage, but it does mean validation performance cannot be treated as an independent final estimate. To establish generalization, all adaptive skill edits and checkpoint choices should be kept separate from a held-out evaluation. Transfer across search datasets is evidence of breadth within QA. It does not demonstrate transfer from shopping to household manipulation.

> ##### What the evidence supports
>
> The evidence supports the joint design as useful: compact conditional guidance, an agent trained to use it, and targeted revision. It does not yet establish unlimited self-improvement, universal cross-domain skills, or a net cost advantage once teacher calls and all training expenses are counted.
{: .block-note }

## Reusable research insights {#insights}

The idea that transfers to other research is to treat the representation of experience as part of the learning system. A memory can be judged by whether it changes the right decision within a bounded context budget. The evaluation question then shifts from "How much history was saved?" to "Which errors does this representation prevent, and under what conditions?"

One useful design pattern splits a rule into a trigger, an action, and an expected consequence. The trigger keeps guidance from being applied everywhere, the action makes it something the agent can carry out, and the consequence gives an evaluator something to test. SkillRL explicitly stores applicability and a principle. A verifiable postcondition would be a research extension; it is not a feature the paper has evaluated.

| Research question | Controlled comparison | Failure signal |
| --- | --- | --- |
| Does abstraction help beyond brevity? | Skills vs length-matched summaries and raw excerpts | Gain disappears when token budget is matched. |
| Does SFT teach skill use? | Matched SFT with useful, shuffled, or absent skills | Performance ignores relevance of supplied skills. |
| Does evolution beat extra teacher help? | Static and evolving banks with equal teacher budgets | Benefit vanishes under matched supervision. |
| Does retrieval select causally useful guidance? | Retrieved vs random vs oracle category-matched skills | Semantically similar skills trigger wrong actions. |
| Does growth remain useful? | Add-only vs merge/prune policies at fixed context | More entries reduce accuracy or raise token cost. |

<div class="caption">Proposed experiments for separating mechanisms; these are not reported results.</div>

A particularly informative experiment would separate acquiring skills from using them. First hold the policy fixed and compare two libraries, which estimates the immediate value of their guidance. Then hold the library fixed and train two policies with matched data budgets to estimate how much skill use is learned. Finally, let both evolve. A factorial design like this can reveal interaction effects that removing one component at a time does not identify.

The GRPO example points to another research opportunity. Under binary rewards, a group in which all trajectories fail has no relative success signal. An explanation of the failures may still suggest a useful precondition, so the outer, language-based loop can supply information when the inner, reward-based loop is uninformative. This complementarity is plausible, but it does not prove that the generated precondition is correct. Testing the proposed repair in the environment would strengthen the learning signal.

Evaluation should account for resources across the whole pipeline. That means measuring successful held-out tasks along with environment interactions, teacher tokens, student tokens, retrieval latency, and total elapsed time. A smaller retrieved context is only one component of the total. If the teacher rewrites the library repeatedly, its cost can outweigh the savings from a shorter prompt.

1. Start with an immutable held-out task set and a versioned initial library. Record category coverage and the budget for collecting experience.
2. Compare compact skills against equally short controls under the same policy and retrieval budget. This isolates the effect of representation more directly.
3. Train skill-using policies with matched demonstrations and environment interactions. Evaluate them with relevant guidance and with deliberately mismatched guidance.
4. Enable failure-driven revision with a fixed teacher budget. Log each proposed change, the failure it addresses, and a replay test where possible.
5. Test on held-out instances and categories, report uncertainty, and measure whether new skills harm earlier capabilities.

For an agent builder, the practical lesson is to keep experience, a proposed rule, and evidence that the rule is useful distinct from one another. For a researcher, it is to study the interface between nonparametric knowledge and parametric learning. SkillRL is a concrete system in which that interface matters, and a useful starting point for experiments that test why it works.

## Related reading {#related}

These short comparisons place SkillRL next to earlier experience-based agents and one later skill-evolution approach. They are pointers for further reading, not full independent reviews.

ExpeL: LLM Agents Are Experiential Learners. ExpeL extracts natural-language insights from accumulated task experience and retrieves those insights without updating model parameters. Read it to separate the value of experiential knowledge from the supervised and reinforcement learning that SkillRL adds. Paper: <https://arxiv.org/abs/2308.10144>

EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle. EvolveR alternates offline distillation of reusable strategic principles with online retrieval and interaction, and improves the policy through reinforcement learning. It is a close predecessor, useful for comparing its experience lifecycle with SkillRL's hierarchical SkillBank and failure-driven updates. Paper: <https://arxiv.org/abs/2510.16079v1>

WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution. WikiSkill uses a persistent wiki to keep execution experience, accumulated knowledge, and skills separate. This is later work and not a predecessor evaluated in SkillRL. It offers a useful comparison between maintaining an intermediate knowledge layer and SkillRL's joint evolution of a skill library and policy. Paper: <https://arxiv.org/abs/2608.27454v1>

Main paper: SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning. <https://arxiv.org/abs/2602.08234v1>
