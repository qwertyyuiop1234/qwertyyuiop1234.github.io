---
layout: post
title: "AgentKVShift: Correcting Reused KV"
date: 2026-09-24 00:00:00 +0900
description: "How a small probe set can repair a much larger KV cache—and where the evidence stops."
tags: ["agentic-memory","kv-cache","inference-optimization"]
toc:
  sidebar: left
---

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/agentkvshift-correcting-reused-kv/f5cf10148eb2d57cf899be544b40fdf6237071dca4a678b1b113b71a15cf4577.png" class="img-fluid rounded z-depth-1" zoomable=true alt="An editorial illustration of sample tabs guiding misaligned memory cards into registration." %}
  </div>
</div>

## Problem &amp; Background {#problem}

An agent can retrieve the right memory and still answer poorly if it reuses the wrong representation of that memory. Stored text may be unchanged, but its transformer keys and values depend on the context in which it was encoded. Reusing a note after earlier context, retrieved neighbors, ordering, or positions change can therefore introduce a representation mismatch. In a causal decoder, the relevant influence is the preceding context; a later query does not retroactively change earlier token states merely by being appended.

The engineering pressure comes from repeated prefill. Before decoding an answer, the model processes the retrieved tokens through its layers. An agent may repeatedly retrieve overlapping notes across a long interaction history, making full re-encoding expensive. KV reuse saves this computation only if the reused states remain useful. It does not improve which memories are retrieved, repair a missing fact, or replace the memory-management policy.

Raw-passage RAG retrieves text largely as stored. Agentic memory systems instead curate summaries, keywords, tags, links, or graph nodes. A-MEM exemplifies note-based organization; LiCoMemory uses a graph organization. These methods improve the information available to the model, but structured retrieval still leaves the cost of encoding those units. AgentKVShift asks whether their reuse errors have a structure that is cheaper to estimate than to recompute token by token.

| Approach | What it solves | What remains |
| --- | --- | --- |
| Full recomputation | Current-context KV for every token | Repeated prefill cost |
| Prefix caching | Reuses identical shared prefixes | Changed preceding context or ordering reduces hits |
| CacheBlend | Refreshes high-divergence tokens | Other token states remain stale |
| ProphetKV | Uses query relevance for refresh selection | Selection alone leaves unrefreshed states unchanged |
| AgentKVShift | Uses probes to correct an entire memory unit | Chunk-local correction may miss reasoning dependencies |

<div class="caption">The progression from exact reuse to selective refresh to correction of unrefreshed states.</div>

The crucial change is the role of a recomputed token. In selective reuse it is one repaired entry. In AgentKVShift it is also a measurement: its fresh-minus-cached residual helps estimate how other entries should move. This is useful when many errors share a direction. It is less useful when the errors are heterogeneous, the probes are unrepresentative, or the expensive part of serving lies elsewhere.

## Core Idea &amp; Method {#method}

For one memory chunk containing n tokens, consider one layer and one attention head of dimension d. The residual is the vector difference between the desired current-context state and the reused state. The same construction is applied separately to keys and values; a layer index makes explicit that every layer needs its own correction.

$$
R_K^{\ell}=K_{\mathrm{fresh}}^{\ell}-K_{\mathrm{reuse}}^{\ell},\qquad R_V^{\ell}=V_{\mathrm{fresh}}^{\ell}-V_{\mathrm{reuse}}^{\ell}
$$

The spectral observation is more specific than simply saying that the error is low rank. Subtracting the mean residual vector from every token greatly reduces the dominant singular mode, especially for keys. This motivates a shared mean plus a token-specific remainder. A dominant singular value alone would not establish a common offset: a general rank-one matrix can have different token coefficients.

$$
R=\mathbf{1}\mu^{\mathrm{T}}+\Xi,\qquad r_j=\mu+\xi_j
$$

Here the transpose is denoted by T, the all-ones vector repeats the offset across tokens, and the centered remainder records what the shared correction cannot explain. The appendix reports that oracle and random-probe corrections explain roughly 64–86% of residual energy in the examined blocks. These are diagnostic observations on sampled settings, not a guarantee for every memory or layer.

1. At check layer 1, compute separate fresh-versus-reused divergence scores for keys and values across the chunk.
2. Choose the top b tokens for each score, where b is the ceiling of r times n. Fix these probe indices and clipped per-token weights for subsequent layers.
3. At each layer, recompute KV for the union of key and value probe sets. Estimate the key mean from key probes and the value mean from value probes.
4. Keep fresh K on key probes and fresh V on value probes. Add the corresponding weighted mean to each remaining state, then continue through the layers.

$$
\begin{aligned}b&=\operatorname{ceil}(rn),\\ d_{K,j}^{1}&=\Vert K_{j,\mathrm{fresh}}^{1}-K_{j,\mathrm{reuse}}^{1}\Vert_2,\\ w_{K,j}&=\min(d_{K,j}^{1},1)\end{aligned}
$$

$$
\widehat{\mu}_{K,C}^{\ell}=\frac{1}{b}\sum_{j\in S_C^K}\left(K_{j,\mathrm{fresh}}^{\ell}-K_{j,\mathrm{reuse}}^{\ell}\right)
$$

$$
K_{j,\mathrm{corr}}^{\ell}=\begin{cases}K_{j,\mathrm{fresh}}^{\ell},&j\in S_C^K,\\ K_{j,\mathrm{reuse}}^{\ell}+w_{K,j}\widehat{\mu}_{K,C}^{\ell},&j\notin S_C^K.\end{cases}
$$

Values follow the same equations with their own scores, set, and mean. The mean itself is an ordinary average of probe residuals; the token-specific weight is applied when correcting a reused entry. Probe locations and weights remain fixed across layers, while the mean is re-estimated at each layer. A single global offset across all chunks and layers would be a different algorithm.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/agentkvshift-correcting-reused-kv/b95bb9272a47ccdd3fef77187f1c316b6bf3bebb57ecc08108f2d8fb522135e8.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Separate key and value probes estimate layer-specific offsets, which correct non-probes while probe states remain fresh." caption="One chunk, one layer: two estimators and two correction paths. Probe indices are selected once; offsets are estimated at every layer." %}
  </div>
</div>

> ##### A nominal 10% budget is not automatically 10% of total work
>
> Key and value probe sets can differ. Their union has between b and min(2b, n) distinct tokens. Check-layer scoring, loading cached states, moving tensors, and correcting non-probes also cost time. The reported recompute ratio must be distinguished from actual refreshed positions, FLOPs, and wall-clock latency.
{: .block-tip }

A small numerical example shows what the mechanism does. Consider one scalar coordinate of a value vector. Suppose two selected probe residuals are 2.2 and 1.8, giving an estimated offset of 2.0. This example is explanatory and is not a reported experiment; different coordinates are corrected together as a vector in the real method.

| Non-probe token | Reused value | Fresh target | Weight | Corrected value |
| --- | --- | --- | --- | --- |
| A | 10\.0 | 12\.1 | 1\.0 | 12\.0 |
| B | 20\.0 | 21\.0 | 0\.5 | 21\.0 |
| C | 30\.0 | 30\.2 | 0\.1 | 30\.2 |

<div class="caption">Illustrative scalar correction: reused value plus weight times 2.0. Weights were fixed from the check layer, not fitted to this example’s current-layer targets.</div>

Token A retains a residual of 0.1 rather than 2.1; B and C happen to match their targets exactly. The probes themselves use their fresh values. The example also exposes the assumption: if an unseen token needed a negative shift, a positive shared estimate could make it worse. The algorithm does not know the fresh targets of non-probes at runtime.

The idealized analysis samples probes uniformly with replacement and applies an unweighted mean correction. If a token residual is mean plus fluctuation and the estimated mean differs from the true mean by e, subtraction gives a simple identity: the shared bias disappears and the remaining error is fluctuation minus estimation error.

$$
e=\widehat{\mu}-\mu,\qquad r_j^{\mathrm{corr}}=\xi_j-e
$$

Under suitable independent sampling and concentration assumptions, the mean-estimation term decreases at the familiar inverse-square-root rate in probe count. A tighter bound becomes plausible when shared bias is larger than the cost of estimating it. That argument motivates the method, but a smaller upper bound does not establish a smaller realized error, better F1, or a guarantee for the weighted top-b implementation. The printed attention-bound constant also requires correction, discussed below.

## Evidence &amp; Results {#evidence}

LoCoMo evaluates long conversational memory using 10 multi-session conversations. Three models are tested with A-MEM and LiCoMemory at r = 0.1: Qwen2.5-3B-Instruct, Qwen3-4B-Instruct, and Mistral-7B-Instruct-v0.3. AMA-Bench uses Qwen3-32B with A-MEM at r = 0.3 and covers six domains and four memory capabilities. The implementation uses each memory system’s default ingestion and retrieves the top 10 units. Results are therefore not a full factorial evaluation of every model, memory system, and benchmark.

| Model / memory | Full F1 | CacheBlend | ProphetKV | AgentKVShift |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B / A-MEM | 0\.339 | 0\.178 | 0\.125 | 0\.319 |
| Qwen3-4B / A-MEM | 0\.444 | 0\.360 | 0\.315 | 0\.429 |
| Mistral-7B / A-MEM | 0\.305 | 0\.208 | 0\.170 | 0\.282 |
| Qwen2.5-3B / LiCo | 0\.390 | 0\.286 | 0\.329 | 0\.384 |
| Qwen3-4B / LiCo | 0\.446 | 0\.360 | 0\.385 | 0\.435 |
| Mistral-7B / LiCo | 0\.509 | 0\.290 | 0\.357 | 0\.491 |

<div class="caption">LoCoMo F1 at nominal r = 0.1; LiCo abbreviates LiCoMemory. Values are reported measurements, not a reproduction.</div>

AgentKVShift leads the two reuse baselines in all six LoCoMo F1 configurations. For Qwen2.5-3B with A-MEM, 0.319 versus 0.339 is a 0.020 absolute gap, or about a 5.90% relative reduction. The full six-row range is approximately 1.54–7.54%, because Mistral with A-MEM reaches 0.282 versus 0.305. Thus the repeated 1.5–6% summary does not cover every entry. Absolute F1 points and relative percent changes should not be interchanged.

<div class="row mt-3 justify-content-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/reviews/agentkvshift-correcting-reused-kv/83aca860ad4359c4bcebe29853f3b5ef5042a0b8107edf6a44c1afa1c5732971.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Grouped bar chart of LoCoMo F1 for six configurations, showing full recomputation and three reuse methods." caption="All six LoCoMo F1 comparisons at r = 0.1. The shared zero baseline avoids exaggerating small gaps to full recomputation." %}
  </div>
</div>

The recompute sweep with Qwen2.5-3B indicates similar quality at roughly 0.10 refresh for AgentKVShift and 0.45–0.55 for the selective baselines. The often quoted approximately fivefold saving is a comparison of nominal refresh ratios near a quality target. It is not a fivefold end-to-end speedup, nor a result established for every model and task.

| AMA capability | Full F1 | AKS F1 | Full judge | AKS judge |
| --- | --- | --- | --- | --- |
| Recall | 0\.296 | 0\.284 | 0\.287 | 0\.279 |
| Causal inference | 0\.284 | 0\.252 | 0\.277 | 0\.218 |
| State updating | 0\.320 | 0\.263 | 0\.252 | 0\.182 |
| State abstraction | 0\.311 | 0\.235 | 0\.176 | 0\.118 |
| All, weighted | 0\.302 | 0\.263 | 0\.257 | 0\.213 |

<div class="caption">AMA-Bench at r = 0.3 with Qwen3-32B and A-MEM. AKS means AgentKVShift; judge accuracy is binary GPT-4o evaluation.</div>

Recall retains approximately 95.95% of full F1 and 97.21% of judge accuracy. Across all capabilities, retention drops to approximately 87.09% and 82.88%. State abstraction is a concrete exception to a universal ranking: ProphetKV has F1 0.254 versus AgentKVShift’s 0.235, and both have judge accuracy 0.118. On Recall’s Web domain, AgentKVShift also trails both reuse baselines in F1. The stronger aggregate result should not hide these failures.

Speed depends on the operating point. At 4,096 tokens and batch 16, r = 0.1 yields 3.33× prefill speedup for the 3B model and 3.65× for the 32B model. At r = 0.3 the narrative reports roughly 1.5–1.6× in that profiling regime. These are prefill measurements; retrieval, memory construction, decoding, and complete agent runtime are not captured by the same number.

| 3B setting, r = 0.1 | CacheBlend | ProphetKV | AgentKVShift |
| --- | --- | --- | --- |
| 2,048 tokens, batch 1 | 1\.19× | 1\.13× | 0\.94× |
| 4,096 tokens, batch 1 | 2\.55× | 2\.28× | 1\.88× |
| 4,096 tokens, batch 16 | 3\.69× | 3\.21× | 3\.33× |
| 16,384 tokens, batch 8 | 2\.66× | 2\.17× | 2\.39× |

<div class="caption">Reported prefill speedups over full recomputation. Values below 1 mean a slowdown; at equal r, correction can be slower than selective refresh.</div>

The A100 measurements cover smaller models, whereas the 32B accuracy and speed runs use H200. The text is inconsistent about A100 memory capacity: the throughput caption says 40GB while the appendix says 80GB. No precise capacity claim is warranted without clarification. All methods reach out-of-memory at 16,384 tokens and batch 16 in the reported throughput table.

Quantization changes storage precision, while reuse changes how much computation is repeated. Their combination is attractive because a fresh-minus-quantized-cache probe residual contains both contextual drift and quantization error. A shared correction can absorb an average component, but cannot remove arbitrary token-specific quantization noise.

| Quantization / r | CacheBlend F1 | ProphetKV F1 | AgentKVShift F1 |
| --- | --- | --- | --- |
| KIVI 2-bit / 0.1 | 0\.076 | 0\.012 | 0\.203 |
| KIVI 4-bit / 0.1 | 0\.190 | 0\.102 | 0\.277 |
| OTT 2-bit / 0.1 | 0\.080 | 0\.038 | 0\.202 |
| OTT 4-bit / 0.1 | 0\.196 | 0\.100 | 0\.317 |
| KIVI 2-bit / 0.5 | 0\.255 | 0\.169 | 0\.244 |

<div class="caption">Selected quantized LoCoMo results for Qwen2.5-3B with A-MEM, including a high-refresh counterexample.</div>

At KIVI 2-bit and r = 0.1, 0.203 is about 2.67 times CacheBlend’s F1. But KIVI 4-bit gives only about 1.46 times CacheBlend, so “over 2×” is not universal across bit widths and comparators. At r = 0.5, CacheBlend slightly wins with KIVI 2-bit and 4-bit. More refresh is not always monotonically better under aggressive quantization: AgentKVShift’s KIVI 2-bit F1 falls from 0.203 at r = 0.1 to 0.184 at r = 0.3.

## Limitations &amp; Confidence {#limitations}

> ##### Strong experimental direction; qualified theoretical support
>
> The reported comparisons support useful quality recovery at low refresh ratios in the evaluated settings. They do not establish universal dominance, statistically significant improvements in every cell, or a guarantee that the practical algorithm approaches full recomputation on arbitrary agent workloads.
{: .block-tip }

There are two different theory gaps. First, uniform random sampling supports a mean-estimation argument, whereas top-b divergence selection can bias the estimator. The practical weighted update also differs from applying one unweighted shift to all entries. Fixed-query, single-layer attention analysis does not directly control the hidden-state evolution of a multilayer transformer or downstream generation.

Second, the proof uses a 1/2 Lipschitz constant from logit infinity norm to probability L1 norm. That mixed-norm claim fails even with two logits. For logits (0, 0) perturbed to (0.1, −0.1), the L1 probability change is about 0.099668, greater than 0.05. A constant of 1 is valid for this norm pair. The displayed lemma statement and later proof also use different coefficients. The residual-estimation intuition remains useful, but the printed constants should not be treated as verified guarantees.

$$
\begin{aligned}z&=(0,0),\quad z^{\prime}=(0.1,-0.1),\\ \Vert\operatorname{softmax}(z^{\prime})-\operatorname{softmax}(z)\Vert_1&\approx0.099668>0.05.\end{aligned}
$$

A useful further distinction is common key shift versus common value shift. Adding exactly the same key vector to every visible key adds one common scalar to every attention logit, which softmax cancels. A per-chunk key shift can still matter by changing relative mass between chunks, and token-dependent weights break the global invariance. A common value shift directly changes the attention output. Therefore shared residual energy alone is not a complete measure of task-relevant error.

The residual diagnostics fit keys more cleanly than values. Shallow-layer value residuals show more non-Gaussian behavior, and projection plots cannot prove the vector concentration assumptions needed by a theorem. The authors also identify cross-chunk reasoning as a remaining weakness. Resolving a changed state across multiple notes can require more than making each note locally resemble its fresh representation.

Reported tables provide point estimates without confidence intervals that establish the reliability of small differences. GPT-4o judge scores depend on the selected reference and judge protocol. Full recomputation is a comparison reference, not a mathematical upper bound on answer metrics: a reuse method occasionally scores higher. A-MEM’s unquantized CacheBlend F1 is also inconsistent between the main LoCoMo table (0.178) and quantization tables (0.150); the values above preserve their own table context rather than silently reconciling them.

## Reusable Research Insights {#insights}

The reusable research move is to model the error of reuse before deciding what to refresh. A cache is not merely correct or stale. Its residual can contain shared bias, a few structured directions, and a token-specific remainder. Different error shapes suggest different uses of a limited measurement budget. A mean shift is especially attractive when the mean explains much of the consequential error and the remaining correction is bandwidth-efficient.

1. Diagnose structure with full-versus-reused states on a small held-out workload. Compare residual norms, singular spectra, and spectra after mean-centering separately for K, V, layer, and memory type.
2. Separate selection from estimation. Compare random probes, top-divergence probes, query-based probes, and an oracle chunk mean at matched actual compute. Measure held-out-token error as well as probe error.
3. Separate correction from recomputation. Ablate K-only, V-only, both, unweighted shifts, and clipped weights; keep token selections fixed where possible. This tests what produces the quality gain.
4. Compare quality at equal wall-clock cost and latency at equal quality. Record the K/V probe union, check-layer work, cache transfer, correction kernels, batch size, and context length.
5. Stress state changes and cross-chunk reasoning. Track Recall and state abstraction separately and report repeated-run uncertainty before designing an adaptive fallback.

A potential extension is asymmetric budgeting: allocate different effort to keys and values, or to layers where the shared component is weak. Another is state-conditioned correction for memories whose relevance changes during reasoning. These are research proposals, not demonstrated improvements. Extra predictors, low-rank correction, or more frequent probing must earn their complexity through quality gains after their own latency is counted.

A practical acceptance criterion is a quality–latency frontier, not one best-looking speedup. If the workload has stable exact prefixes, ordinary prefix reuse may already solve much of the problem. If retrieval or decoding dominates total time, prefill gains may have limited impact. If a small batch makes correction overhead dominant, full recomputation or selective reuse may be preferable. The decision should follow the measured workload and acceptable quality loss.

The broader lesson extends beyond KV caches: expensive recomputation can serve both as repair and as a sensor for a larger population. That strategy succeeds only when the measured error structure transfers to unmeasured items. Testing the transfer is as important as choosing the probes.

## Related Reading {#related-reading}

CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. <https://arxiv.org/abs/2405.16444> — An earlier selective-refresh approach for combining cached retrieved passages. It is the natural baseline for distinguishing the value of choosing refresh tokens from the value of correcting tokens left unrefreshed.

ProphetKV: User-Query-Driven Selective Recomputation for Efficient KV Cache Reuse in Retrieval-Augmented Generation. <https://arxiv.org/abs/2602.02579> — A prior query-driven selection approach. It changes which tokens are recomputed; AgentKVShift additionally estimates a correction for reused entries.

A-MEM: Agentic Memory for LLM Agents. <https://arxiv.org/abs/2502.12110> — A prior structured-memory system that organizes enriched notes and links. It supplies one of the memory workloads accelerated by AgentKVShift, addressing memory organization rather than the same inference optimization.

Main paper: AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems. <https://arxiv.org/abs/2607.21604v1>
