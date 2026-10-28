---
layout: post
title: "AgentKVShift: Correcting Reused KV"
date: 2026-09-24 00:00:00 +0900
description: "How a small probe set can repair a much larger KV cache, and where the evidence stops."
tags: ["agentic-memory","kv-cache","inference-optimization"]
toc:
  sidebar: left
---

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/agentkvshift-correcting-reused-kv/f5cf10148eb2d57cf899be544b40fdf6237071dca4a678b1b113b71a15cf4577.png" class="img-fluid rounded z-depth-1" zoomable=true alt="An editorial illustration of sample tabs guiding misaligned memory cards into registration." %}
  </div>
</div>

## Problem and background {#problem}

An agent can retrieve the right memory and still answer poorly if it reuses the wrong representation of that memory. The stored text may be unchanged, but its transformer keys and values depend on the context it was encoded in. If the earlier context, the retrieved neighbors, their ordering, or their positions have changed since then, reusing the note can introduce a representation mismatch. In a causal decoder, the influence that matters comes from the preceding context. Appending a later query does not go back and change the states of earlier tokens.

The practical cost is repeated prefill. Before decoding an answer, the model runs the retrieved tokens through its layers. Over a long interaction history an agent may retrieve overlapping notes many times, so re-encoding them in full each time is expensive. KV reuse saves this computation only if the reused states stay useful. It does not change which memories are retrieved or fill in a missing fact, and it is no substitute for the memory-management policy.

Raw-passage RAG retrieves text largely as stored. Agentic memory systems instead curate summaries, keywords, tags, links, or graph nodes. A-MEM is an example of note-based organization, and LiCoMemory organizes memory as a graph. These systems improve the information the model gets, but structured retrieval does not remove the cost of encoding the retrieved units. AgentKVShift asks whether the errors from reusing them have a structure that is cheaper to estimate than to recompute token by token.

| Approach | What it solves | What remains |
| --- | --- | --- |
| Full recomputation | Current-context KV for every token | Repeated prefill cost |
| Prefix caching | Reuses identical shared prefixes | Changed preceding context or ordering reduces hits |
| CacheBlend | Refreshes high-divergence tokens | Other token states remain stale |
| ProphetKV | Uses query relevance for refresh selection | Selection alone leaves unrefreshed states unchanged |
| AgentKVShift | Uses probes to correct an entire memory unit | Chunk-local correction may miss reasoning dependencies |

<div class="caption">The progression from exact reuse to selective refresh to correction of unrefreshed states.</div>

What changes is the role of a recomputed token. In selective reuse it is one repaired entry. In AgentKVShift it is also a measurement, since its fresh-minus-cached residual helps estimate how the other entries should move. This helps when many errors share a direction. It helps less when the errors are heterogeneous or the probes are unrepresentative, or when the expensive part of serving is somewhere else.

## Core idea and method {#method}

Consider one memory chunk of n tokens, one layer, and one attention head of dimension d. The residual is the vector difference between the desired current-context state and the reused state. Keys and values each get this construction separately. The layer index in the notation makes explicit that every layer needs its own correction.

$$
R_K^{\ell}=K_{\mathrm{fresh}}^{\ell}-K_{\mathrm{reuse}}^{\ell},\qquad R_V^{\ell}=V_{\mathrm{fresh}}^{\ell}-V_{\mathrm{reuse}}^{\ell}
$$

The spectral observation says more than that the error is low rank. Subtracting the mean residual vector from every token greatly reduces the dominant singular mode, especially for keys. That is the motivation for a shared mean plus a token-specific remainder. A dominant singular value would not by itself establish a common offset, because a general rank-one matrix can have different coefficients for different tokens.

$$
R=\mathbf{1}\mu^{\mathrm{T}}+\Xi,\qquad r_j=\mu+\xi_j
$$

Here T denotes the transpose, the all-ones vector repeats the offset across tokens, and the centered remainder holds what the shared correction cannot explain. The appendix reports that oracle and random-probe corrections explain roughly 64-86% of residual energy in the blocks it examines. These are diagnostic observations on sampled settings. They do not guarantee the same for every memory or layer.

1. At check layer 1, compute fresh-versus-reused divergence scores across the chunk, separately for keys and for values.
2. Choose the top b tokens for each score, where b is the ceiling of r times n. These probe indices and the clipped per-token weights stay fixed for subsequent layers.
3. At each layer, recompute KV for the union of the key and value probe sets. Estimate the key mean from the key probes and the value mean from the value probes.
4. Keep fresh K on key probes and fresh V on value probes. Add the corresponding weighted mean to every other state, then move on to the next layer.

$$
\begin{aligned}b&=\operatorname{ceil}(rn),\\ d_{K,j}^{1}&=\Vert K_{j,\mathrm{fresh}}^{1}-K_{j,\mathrm{reuse}}^{1}\Vert_2,\\ w_{K,j}&=\min(d_{K,j}^{1},1)\end{aligned}
$$

$$
\widehat{\mu}_{K,C}^{\ell}=\frac{1}{b}\sum_{j\in S_C^K}\left(K_{j,\mathrm{fresh}}^{\ell}-K_{j,\mathrm{reuse}}^{\ell}\right)
$$

$$
K_{j,\mathrm{corr}}^{\ell}=\begin{cases}K_{j,\mathrm{fresh}}^{\ell},&j\in S_C^K,\\ K_{j,\mathrm{reuse}}^{\ell}+w_{K,j}\widehat{\mu}_{K,C}^{\ell},&j\notin S_C^K.\end{cases}
$$

Values use the same equations with their own scores, probe set, and mean. The mean is an ordinary average of probe residuals. The token-specific weight enters only when a reused entry is corrected. Probe locations and weights stay fixed across layers, while the mean is re-estimated at each layer. One global offset shared by all chunks and layers would be a different algorithm.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/agentkvshift-correcting-reused-kv/b95bb9272a47ccdd3fef77187f1c316b6bf3bebb57ecc08108f2d8fb522135e8.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Separate key and value probes estimate layer-specific offsets, which correct non-probes while probe states remain fresh." caption="One chunk, one layer: two estimators and two correction paths. Probe indices are selected once; offsets are estimated at every layer." %}
  </div>
</div>

> ##### A nominal 10% budget is not automatically 10% of total work
>
> The key and value probe sets can differ, so their union has between b and min(2b, n) distinct tokens. Check-layer scoring, loading cached states, moving tensors, and correcting non-probes also take time. The reported recompute ratio is therefore not the same as the positions actually refreshed, the FLOPs, or the wall-clock latency.
{: .block-note }

As a small numerical example, take one scalar coordinate of a value vector. Suppose the two selected probe residuals are 2.2 and 1.8, which gives an estimated offset of 2.0. The example is explanatory and not a reported experiment. In the real method the coordinates are corrected together as a vector.

| Non-probe token | Reused value | Fresh target | Weight | Corrected value |
| --- | --- | --- | --- | --- |
| A | 10\.0 | 12\.1 | 1\.0 | 12\.0 |
| B | 20\.0 | 21\.0 | 0\.5 | 21\.0 |
| C | 30\.0 | 30\.2 | 0\.1 | 30\.2 |

<div class="caption">Illustrative scalar correction: reused value plus weight times 2.0. Weights were fixed from the check layer, not fitted to this example&#39;s current-layer targets.</div>

Token A is left with a residual of 0.1 instead of 2.1, and B and C happen to match their targets exactly. The probes themselves use their fresh values. The example also exposes the underlying assumption: if an unseen token needed a negative shift, a positive shared estimate could make it worse. At runtime, the algorithm does not know the fresh targets of non-probes.

The idealized analysis samples probes uniformly with replacement and applies an unweighted mean correction. Write each token residual as mean plus fluctuation, and let e be the difference between the estimated mean and the true mean. Subtraction then gives a simple identity: the shared bias cancels, and the remaining error is the fluctuation minus the estimation error.

$$
e=\widehat{\mu}-\mu,\qquad r_j^{\mathrm{corr}}=\xi_j-e
$$

Under suitable independent sampling and concentration assumptions, the mean-estimation term shrinks at the familiar inverse-square-root rate in the number of probes. A tighter bound becomes plausible when the shared bias is larger than the cost of estimating it. The argument motivates the method. A smaller upper bound does not, however, establish a smaller realized error or better F1, and it gives no guarantee for the weighted top-b implementation. The printed attention-bound constant also needs correcting, as discussed below.

## Evidence and results {#evidence}

LoCoMo evaluates long conversational memory on 10 multi-session conversations. Three models are tested with A-MEM and LiCoMemory at r = 0.1: Qwen2.5-3B-Instruct, Qwen3-4B-Instruct, and Mistral-7B-Instruct-v0.3. AMA-Bench uses Qwen3-32B with A-MEM at r = 0.3 and spans six domains and four memory capabilities. The implementation keeps each memory system's default ingestion and retrieves the top 10 units. So the results are not a full factorial evaluation of every model, memory system, and benchmark.

| Model / memory | Full F1 | CacheBlend | ProphetKV | AgentKVShift |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B / A-MEM | 0\.339 | 0\.178 | 0\.125 | 0\.319 |
| Qwen3-4B / A-MEM | 0\.444 | 0\.360 | 0\.315 | 0\.429 |
| Mistral-7B / A-MEM | 0\.305 | 0\.208 | 0\.170 | 0\.282 |
| Qwen2.5-3B / LiCo | 0\.390 | 0\.286 | 0\.329 | 0\.384 |
| Qwen3-4B / LiCo | 0\.446 | 0\.360 | 0\.385 | 0\.435 |
| Mistral-7B / LiCo | 0\.509 | 0\.290 | 0\.357 | 0\.491 |

<div class="caption">LoCoMo F1 at nominal r = 0.1. LiCo is short for LiCoMemory. Values are reported measurements, not a reproduction.</div>

AgentKVShift beats both reuse baselines in all six LoCoMo F1 configurations. For Qwen2.5-3B with A-MEM, 0.319 versus 0.339 is a 0.020 absolute gap, or a relative reduction of about 5.90%. Over all six rows the relative reduction is approximately 1.54-7.54%, with the upper end coming from Mistral with A-MEM (0.282 versus 0.305). The repeated 1.5-6% summary therefore does not cover every entry. Absolute F1 points and relative percent changes should not be used interchangeably.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/agentkvshift-correcting-reused-kv/83aca860ad4359c4bcebe29853f3b5ef5042a0b8107edf6a44c1afa1c5732971.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Grouped bar chart of LoCoMo F1 for six configurations, showing full recomputation and three reuse methods." caption="All six LoCoMo F1 comparisons at r = 0.1. The shared zero baseline avoids exaggerating small gaps to full recomputation." %}
  </div>
</div>

The recompute sweep with Qwen2.5-3B indicates that AgentKVShift at roughly 0.10 refresh reaches quality similar to the selective baselines at 0.45-0.55. The approximately fivefold saving that is often quoted compares nominal refresh ratios near a quality target. It is not a fivefold end-to-end speedup, and it has not been established for every model and task.

| AMA capability | Full F1 | AKS F1 | Full judge | AKS judge |
| --- | --- | --- | --- | --- |
| Recall | 0\.296 | 0\.284 | 0\.287 | 0\.279 |
| Causal inference | 0\.284 | 0\.252 | 0\.277 | 0\.218 |
| State updating | 0\.320 | 0\.263 | 0\.252 | 0\.182 |
| State abstraction | 0\.311 | 0\.235 | 0\.176 | 0\.118 |
| All, weighted | 0\.302 | 0\.263 | 0\.257 | 0\.213 |

<div class="caption">AMA-Bench at r = 0.3 with Qwen3-32B and A-MEM. AKS is AgentKVShift. Judge accuracy is a binary GPT-4o evaluation.</div>

On Recall, AgentKVShift retains approximately 95.95% of full F1 and 97.21% of judge accuracy. Across all capabilities, retention drops to approximately 87.09% and 82.88%. State abstraction is a concrete exception to any universal ranking: ProphetKV has F1 0.254 against AgentKVShift's 0.235, and both have judge accuracy 0.118. In the Web domain under Recall, AgentKVShift also trails both reuse baselines in F1. The stronger aggregate result should not hide these failures.

Speed depends on the operating point. At 4,096 tokens and batch 16, r = 0.1 gives a 3.33× prefill speedup for the 3B model and 3.65× for the 32B model. For r = 0.3, the paper's text reports roughly 1.5-1.6× in the same profiling regime. These are prefill measurements. The same number does not capture retrieval, memory construction, decoding, or complete agent runtime.

| 3B setting, r = 0.1 | CacheBlend | ProphetKV | AgentKVShift |
| --- | --- | --- | --- |
| 2,048 tokens, batch 1 | 1\.19× | 1\.13× | 0\.94× |
| 4,096 tokens, batch 1 | 2\.55× | 2\.28× | 1\.88× |
| 4,096 tokens, batch 16 | 3\.69× | 3\.21× | 3\.33× |
| 16,384 tokens, batch 8 | 2\.66× | 2\.17× | 2\.39× |

<div class="caption">Reported prefill speedups over full recomputation. Values below 1 mean a slowdown; at equal r, correction can be slower than selective refresh.</div>

The A100 measurements cover the smaller models, and the 32B accuracy and speed runs use H200. The paper is inconsistent about the A100's memory capacity: the throughput caption says 40GB, while the appendix says 80GB. Without clarification, no precise capacity can be claimed. In the reported throughput table, all methods run out of memory at 16,384 tokens and batch 16.

Quantization changes storage precision, and reuse changes how much computation is repeated. The combination is appealing because a probe residual computed as fresh minus quantized cache contains both contextual drift and quantization error. A shared correction can absorb the average component, but it cannot remove arbitrary token-specific quantization noise.

| Quantization / r | CacheBlend F1 | ProphetKV F1 | AgentKVShift F1 |
| --- | --- | --- | --- |
| KIVI 2-bit / 0.1 | 0\.076 | 0\.012 | 0\.203 |
| KIVI 4-bit / 0.1 | 0\.190 | 0\.102 | 0\.277 |
| OTT 2-bit / 0.1 | 0\.080 | 0\.038 | 0\.202 |
| OTT 4-bit / 0.1 | 0\.196 | 0\.100 | 0\.317 |
| KIVI 2-bit / 0.5 | 0\.255 | 0\.169 | 0\.244 |

<div class="caption">Selected quantized LoCoMo results for Qwen2.5-3B with A-MEM, including a high-refresh counterexample.</div>

With KIVI 2-bit at r = 0.1, AgentKVShift's 0.203 is about 2.67 times CacheBlend's F1. With KIVI 4-bit the ratio is only about 1.46, so "over 2×" does not hold across all bit widths and comparators. At r = 0.5, CacheBlend wins slightly with both KIVI 2-bit and 4-bit. Under aggressive quantization, more refresh is not always better: AgentKVShift's KIVI 2-bit F1 falls from 0.203 at r = 0.1 to 0.184 at r = 0.3.

## Limitations and confidence {#limitations}

> ##### Strong experimental direction, qualified theoretical support
>
> In the evaluated settings, the reported comparisons support useful quality recovery at low refresh ratios. They do not establish that the method dominates everywhere or that every cell shows a statistically significant improvement. Nor do they guarantee that the practical algorithm approaches full recomputation on arbitrary agent workloads.
{: .block-note }

The theory has two separate gaps. First, uniform random sampling supports a mean-estimation argument, but top-b divergence selection can bias the estimator. The weighted update used in practice also differs from applying one unweighted shift to all entries. And an analysis of single-layer attention with a fixed query does not directly control the hidden-state evolution of a multilayer transformer, or downstream generation.

Second, the proof uses a Lipschitz constant of 1/2 from the logit infinity norm to the probability L1 norm. That mixed-norm claim fails even with two logits. Perturbing logits (0, 0) to (0.1, −0.1) changes the probabilities by about 0.099668 in L1 norm, which is greater than 0.05. For this pair of norms, a constant of 1 is valid. The lemma as displayed and the later proof also use different coefficients. The residual-estimation intuition is still useful, but the printed constants should not be treated as verified guarantees.

$$
\begin{aligned}z&=(0,0),\quad z^{\prime}=(0.1,-0.1),\\ \Vert\operatorname{softmax}(z^{\prime})-\operatorname{softmax}(z)\Vert_1&\approx0.099668>0.05.\end{aligned}
$$

A further distinction is between a common key shift and a common value shift. Adding exactly the same key vector to every visible key adds one common scalar to every attention logit, and softmax cancels it. A per-chunk key shift can still matter because it changes the relative mass between chunks, and token-dependent weights break the global invariance. A common value shift, on the other hand, changes the attention output directly. So shared residual energy alone is not a complete measure of task-relevant error.

The residual diagnostics fit keys more cleanly than values. Value residuals in shallow layers show more non-Gaussian behavior, and projection plots cannot prove the vector concentration assumptions that a theorem needs. The authors also identify cross-chunk reasoning as a remaining weakness. When a state changes across multiple notes, resolving it can require more than making each note locally resemble its fresh representation.

The tables report point estimates without confidence intervals, so the reliability of small differences is not established. GPT-4o judge scores depend on the chosen reference and judge protocol. Full recomputation is a reference for comparison, not a mathematical upper bound on answer metrics, and a reuse method occasionally scores higher. The unquantized CacheBlend F1 for A-MEM is also inconsistent between the main LoCoMo table (0.178) and the quantization tables (0.150). The values above stay in their own table context and are not silently reconciled.

## Reusable research insights {#insights}

The research move worth reusing is to model the error of reuse before deciding what to refresh. A cache does not have to be treated as simply correct or stale. Its residual can contain a shared bias, a few structured directions, and a token-specific remainder, and each error shape suggests a different way to spend a limited measurement budget. A mean shift is especially attractive when the mean explains much of the error that matters and the remaining correction is bandwidth-efficient.

1. Diagnose the structure by comparing full and reused states on a small held-out workload. Compare residual norms, singular spectra, and spectra after mean-centering, separately for K, V, layer, and memory type.
2. Separate selection from estimation. Compare random probes, top-divergence probes, query-based probes, and an oracle chunk mean at matched actual compute. Measure held-out-token error as well as probe error.
3. Separate correction from recomputation. Ablate K-only, V-only, both, unweighted shifts, and clipped weights. Keep token selections fixed where possible so the ablation shows what produces the quality gain.
4. Compare quality at equal wall-clock cost and latency at equal quality. Record the K/V probe union, check-layer work, cache transfer, correction kernels, batch size, and context length.
5. Stress-test state changes and cross-chunk reasoning. Track Recall and state abstraction separately, and report uncertainty across repeated runs before designing an adaptive fallback.

One possible extension is asymmetric budgeting, which allocates different effort to keys and values, or to layers where the shared component is weak. Another is state-conditioned correction for memories whose relevance changes during reasoning. These are research proposals, not demonstrated improvements. Extra predictors, low-rank correction, or more frequent probing have to justify their complexity with quality gains measured after their own latency is counted.

A practical acceptance criterion is the quality-latency frontier, not a single best-looking speedup. If the workload has stable exact prefixes, ordinary prefix reuse may already solve much of the problem. When retrieval or decoding dominates total time, prefill gains may have limited impact. And if the batch is small enough that correction overhead dominates, full recomputation or selective reuse may be the better choice. The decision should follow the measured workload and the acceptable quality loss.

The same idea applies beyond KV caches. Expensive recomputation can repair what it recomputes and also act as a sensor for a larger population. That only works when the error structure measured on some items transfers to the unmeasured ones, so testing the transfer matters as much as choosing the probes.

## Related reading {#related-reading}

CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. <https://arxiv.org/abs/2405.16444>. An earlier selective-refresh approach for combining cached retrieved passages. It is the natural baseline for separating the value of choosing refresh tokens from the value of correcting the tokens left unrefreshed.

ProphetKV: User-Query-Driven Selective Recomputation for Efficient KV Cache Reuse in Retrieval-Augmented Generation. <https://arxiv.org/abs/2602.02579>. A prior query-driven selection approach. It changes which tokens are recomputed, while AgentKVShift also estimates a correction for the reused entries.

A-MEM: Agentic Memory for LLM Agents. <https://arxiv.org/abs/2502.12110>. A prior structured-memory system that organizes enriched notes and links. It supplies one of the memory workloads that AgentKVShift accelerates. A-MEM itself addresses memory organization and is not the same kind of inference optimization.

Main paper: AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems. <https://arxiv.org/abs/2607.21604v1>
