---
layout: post
title: "LLM inference: core concepts and terms"
date: 2026-11-01 00:00:00 +0900
description: "A map of the terms in LLM serving: prefill and decode, batching, KV cache, metrics, sampling, engines and libraries."
tags: [llm-inference, kv-cache, vllm, sglang]
categories: [LLM]
toc:
  sidebar: left
related_posts: false
---

LLM inference is a pipeline. The server receives requests and groups them into batches. It reads each prompt in one pass, which is called prefill. Then it makes the output one token at a time, which is called decode. This post collects the abbreviations and feature names that attach to each part of this pipeline.

Almost every term serves one of two goals: keep the GPU busy, or get the answer to the user sooner. With these two goals in mind, each new term is easier to place.

<blockquote class="block-note" markdown="1">

##### About this post

This post is based on the Zhihu article [大模型推理核心概念与术语总结](https://zhuanlan.zhihu.com/p/1983137653336585901) ("Key concepts and terms of large model inference") by kaiyuan, from the [InfraTech](https://github.com/CalvinXKY/InfraTech) series. I put the terms in a new order, added worked examples, and marked the places where the article is unclear or differs from common usage. In this post, "the source article" means that article. All figures are new drawings.

</blockquote>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/posts/llm-inference-key-terms/fig01-term-map.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Map of where each inference term sits on the path of one request." caption="<b>Figure 1.</b> The term map for this post. The boxes on the left show the path of one request, and the terms next to each box attach to that stage. The curved arrow is the decode loop: the model makes one token and goes back to decode until it makes EOS. The brackets on the right show the part of the path that each latency metric measures. The dashed box lists the ways to split all of these stages across many GPUs. Come back to this map when you lose your place." %}
  </div>
</div>

## Two stages to know first: prefill and decode {#prefill-and-decode}

The source article starts with a list of terms. But almost every term targets one of two stages, so this post explains these stages first.

Prefill (prompt processing)
: The model runs the full prompt through all layers in one pass and makes the first output token. It computes N tokens at the same time, so the GPU compute units stay full. Prefill is **compute-bound**.

Decode (generation)
: After prefill, the model makes one new token per step. Each step computes only one token, but it must read all the model weights and the full KV cache from memory. Decode is **memory-bandwidth-bound**. The source article calls it "storage-bound".

KV cache (key-value cache)
: The memory that keeps the key and value vectors of the tokens that attention already processed. Without it, the model must compute all the earlier text again for each new token.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig02-prefill-vs-decode.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Prefill multiplies a token matrix by the weights, while decode multiplies one vector by the weights and also reads the KV cache." caption="<b>Figure 2.</b> The two stages go through the same layer but behave differently. On the left, prefill reads the weights W once and uses them for N tokens, so computation is the bottleneck. On the right, decode reads all of W and the full KV cache to make one token, so the memory read speed is the bottleneck. This difference is the reason for PD disaggregation and chunked prefill. It is also why it pays to batch many decode requests together." %}
  </div>
</div>

<blockquote class="block-note" markdown="1">

##### Analogy

Prefill is like a student who reads the whole exam paper once at the start. Decode is like a student who writes the answer one letter at a time. Before each letter, the student opens the full textbook (the weights) and the answer so far (the KV cache) again. Most of the time goes to reading, so the hand that writes (the compute units) often waits.

</blockquote>

<blockquote class="block-note" markdown="1">

##### In short

Prefill runs out of compute first, and decode runs out of memory bandwidth first. Most techniques in this post answer one question: how do we mix or separate these two bottlenecks so that the GPU does not sit idle?

</blockquote>

## Core features: what an engine offers {#core-features}

The source article lists about 20 features of inference engines such as vLLM and SGLang. This post puts related features into groups.

### Batching: keep the GPU busy

Continuous batching (iteration-level batching)
: The engine does not run a fixed batch to the end. At each decode step, it removes the finished requests and adds new requests at once. This is the most basic way to keep the GPU busy.

Static batching waits until the longest request in the batch is done. When a short request ends early, its slot stays empty. Continuous batching gives that slot to a waiting request at once.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig03-static-vs-continuous-batching.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Static batching leaves slots empty until the longest request ends; continuous batching fills each free slot with a waiting request." caption="<b>Figure 3.</b> Static batching and continuous batching. Each row is one batch slot, and each column T1 to T8 is one engine step (iteration). P cells are prefill, S cells are decode, and END marks the end of a request. With static batching, slots stay empty after their requests end, until the longest request ends. With continuous batching, a waiting request (S5, S6, …) takes a slot right after an END, so no slot stays empty." %}
  </div>
</div>

<blockquote class="block-note" markdown="1">

##### Analogy

Static batching is like a four-seat ride that waits until all four riders get off before it takes new riders. Continuous batching gives a free seat to the next person in line as soon as one rider gets off.

</blockquote>

### KV cache memory management

PagedAttention
: The engine divides the KV cache into blocks of fixed size, for example 16 tokens per block. A block table maps the logical blocks of each request to physical blocks in GPU memory. The idea is the same as virtual memory paging in an operating system, and it reduces memory fragmentation.

Copy-on-write (CoW)
: Several requests can share the blocks of a common prefix. The engine copies a shared block only when one request must write different content into it. Until that write, no block is copied.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig04-paged-attention.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Logical KV blocks of one request map through a block table to scattered physical blocks in GPU memory." caption="<b>Figure 4.</b> PagedAttention with blocks of 4 tokens. The prompt of request A, &quot;Alice opened the old wooden door and&quot;, fills logical blocks 0 and 1 in order. In GPU memory, these are physical blocks 7 and 1, which are not next to each other. The block table in the middle records the mapping and the number of filled slots. The circled numbers show the time order. The new token &quot;saw&quot; goes into the free slot of block 1. The next token &quot;a&quot; needs a new block, so the engine allocates physical block 3. The idea is from Kwon et al. (2023)." %}
  </div>
</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig05-copy-on-write.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Two samples share physical blocks until one must write a different token, which triggers a copy." caption="<b>Figure 5.</b> Copy-on-write. Samples A1 and A2 come from the same prompt, so they share physical blocks 7 and 1 (reference count 2). At the last slot, A1 must write &quot;saw&quot; and A2 must write &quot;heard&quot;. At that moment, the engine copies block 1 into a new block 3 for A1. A2 keeps block 1, and the reference count of block 1 drops from 2 to 1." %}
  </div>
</div>

RadixAttention (SGLang)
: SGLang keeps the token sequences of requests in a radix tree, so requests with the same prefix share their KV cache automatically. The KV tensors sit in non-contiguous pages, and by default one page holds one token (`--page-size 1`). When memory is full, SGLang evicts leaf nodes, by default with an LRU (least recently used) policy. The source article spells the name "Radix Attetnion" by mistake.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig06-radix-tree-lru.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Six snapshots of a radix tree that grows, splits, evicts a leaf and reuses a path." caption="<b>Figure 6.</b> A radix tree at six moments. In (2) and (3), one chat continues and its branch grows. In (4), a new chat shares the system prompt &quot;You are a helpful assistant.&quot;, so a new branch starts below that node. In (5), memory is full, and the least recently used leaf is evicted (dashed, with an X). In (6), chat B finishes, and its answer becomes a new leaf below its question. Green nodes are new, and blue nodes are reused. For readability, each message has its own node here. A real radix tree merges a chain of single-child nodes into one edge, so in (4) it splits that edge after the system prompt." %}
  </div>
</div>

APC (automatic prefix caching)
: A new request can start with the same prefix as an earlier request. The engine then reuses the stored KV cache for that prefix. Prefill has less to compute, so the time to first token (TTFT) goes down. APC in vLLM has the same goal as RadixAttention but uses hash-based block matching.

### Disaggregation: different work on different machines

PD disaggregation (prefill-decode disaggregation)
: The engine puts the compute-bound prefill (P) and the memory-bound decode (D) on different GPUs or instances. The prefill side makes the KV cache and sends it to the decode side, and the decode side continues the generation. The two stages no longer interrupt each other. Each side can use its own GPU count, parallelism and batch size. So the engine can tune TTFT and ITL separately, and more requests meet their latency targets on each GPU. PD disaggregation alone does not raise the raw throughput.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig07-pd-disaggregation.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="PD aggregation runs prefill and decode in one instance; PD disaggregation runs them in two instances joined by a KV cache transfer." caption="<b>Figure 7.</b> On the left, PD aggregation: prefill and decode run in one model instance and share one KV cache. On the right, PD disaggregation: a prefill instance and a decode instance run apart, and a KV cache transfer connects them. The loop on the decode side makes one token per step." %}
  </div>
</div>

AFD (attention-FFN disaggregation)
: AFD splits the work more finely than PD disaggregation. Inside one layer, the attention part and the FFN part (in particular the MoE experts) run on different devices. You tune the ratio of attention devices (A) to FFN devices (F) so that both sides stay busy.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig08-afd.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Attention instances send hidden states to FFN instances, which send expert outputs back, once per layer." caption="<b>Figure 8.</b> Attention-FFN disaggregation for one MoE layer. The attention instances run Norm, attention, Norm and the router, then send the hidden states to the FFN instances in fp8. The FFN side scatters the tokens to their experts (EP), computes, and sends the outputs back in bf16. The attention side then combines the results with the top-k weights of the router. Each layer makes one round trip between the two device groups." %}
  </div>
</div>

### Faster decode

Speculative decoding
: A small, fast draft model guesses several tokens first. The large target model then checks all the guesses in one forward pass. The engine keeps the guesses up to the first wrong one, and from that point it uses the token of the large model. This is the rule for greedy decoding. With sampling, the engine accepts each guess with the probability min(1, p/q). Here p and q are the probabilities from the large and the small model. If it rejects a guess, it samples a replacement from a corrected distribution. So the output follows the distribution of the large model exactly. This works because decode is memory-bound: the large model needs about the same time to check one token or five tokens.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig09-speculative-decoding.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="A draft model proposes three tokens and the target model accepts two and corrects the third in one pass." caption="<b>Figure 9.</b> Speculative decoding. (1) After &quot;The sky is&quot;, the small draft model proposes &quot;blue&quot;, &quot;and&quot; and &quot;cold&quot;, one token at a time. (2) The large target model checks the guesses in one forward pass. &quot;The&quot; and &quot;sky&quot; are already in the KV cache. So the pass runs only the 4 new positions and gives its own token at each one. (3) The engine accepts &quot;blue&quot; and &quot;and&quot;, rejects &quot;cold&quot;, and uses the target token &quot;clear&quot; in its place. One pass of the target model gives three new tokens." %}
  </div>
</div>

Chunked prefill
: The engine cuts a long prefill into small chunks along the sequence. Each chunk goes into the same batch as the decode requests of that step. A long prefill can then no longer take the whole GPU and stop the decode of other requests. The chunks also use the compute that decode alone leaves idle.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig10-sarathi-chunked-prefill.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Two pipeline timelines; the chunked version has smaller bubbles and ends earlier." caption="<b>Figure 10.</b> Chunked prefill in a two-stage pipeline, after the idea of SARATHI (Agrawal et al., 2023). (a) Without chunking, each prefill (Ap, Bp, …) runs whole, and idle gaps (bubbles) appear between the pipeline stages. (b) With chunking, each prefill splits into chunks (Ap1, Ap2, …), and decode tokens ride along in the same batch. The bubbles get much smaller, and the work ends earlier. Small bubbles stay at the end, where the last decode must wait until its prefill leaves stage 2. In the paper, SARATHI reduces the bubbles by 6.29× on GPT-3 with pipeline parallelism." %}
  </div>
</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig11-vllm-chunked-prefill-budget.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="A token budget of 10 per step split across the prefill and decode tokens of three requests." caption="<b>Figure 11.</b> Chunked prefill in vLLM. On the left, R1 is in decode when R2, R3 and R4 arrive, and their prefills join the same steps. On the right, a budget of 10 tokens per step splits the prompts of R1 (3 tokens), R2 (5) and R3 (12). Step 0 is {R1: 3, R2: 5, R3: 2}. In step 1, R1 and R2 decode one token each, and R3 gets 8 prefill tokens. Step 2 gives R3 its last 2 prefill tokens. From step 3, each request decodes one token per step." %}
  </div>
</div>

<blockquote class="block-note" markdown="1">

##### Worked example: the token budget

Check the right side of Figure 11 with numbers. The budget is 10 tokens per step. In step 0, 3 + 5 + 2 = 10. In step 1, R1 and R2 have finished prefill, so each gets one decode token. R3 has 12 − 2 = 10 prefill tokens left, and the budget allows 8 of them (1 + 1 + 8 = 10). In step 2, R3 gets its last 2 prefill tokens, plus the two decode tokens (1 + 1 + 2 = 4). From step 3, all three requests decode one token per step.

The vLLM docs describe the rule like this: put the decode tokens in first, then fill the rest of the budget with prefill chunks. The current V1 scheduler is a little different. It serves the running requests first, in queue order, and a running request can also be an unfinished prefill. New requests then use the rest of the budget. In this example, both rules give the same steps. Smaller chunks make the decode latency (ITL) more stable. But one prefill then spans more steps, so its TTFT can go up.

</blockquote>

### Quantization, LoRA and output control

W8A16 (8-bit weights, 16-bit activations)
: The weights (W) are stored in 8 bits (INT8 or FP8), and the activations (A) keep 16-bit floating-point precision, such as FP16. Related formats are W8A8 and W4A16. Sometimes the data type follows the name, as in `w8a8_int8`.

<blockquote class="block-note" markdown="1">

##### Why shrink only the weights?

In a dense model at a small batch size, decode reads all the weights at each step. So the weight size in bytes sets most of the step time. A 7B (7 billion parameter) model needs about 14 GB in FP16, at 2 bytes per parameter. In INT8 (1 byte) it needs about 7 GB, and in 4-bit about 3.5 GB. When the engine reads half the bytes, memory-bound decode can become up to twice as fast. W8A8 also makes the activations 8-bit, so the matrix multiplications can run on INT8 units. The risk of accuracy loss is higher.

</blockquote>

Multi-LoRA (serving many LoRA adapters at once)
: One base model carries many small LoRA adapters, and each adapter is fine-tuned for one task. For each request, the engine selects the adapter that the request needs. All requests share the weights of the base model.

Guided decoding
: At each step, the engine removes the tokens that would break a given format or grammar, such as JSON, SQL or a regular expression. So the output that the model has made so far is always valid. The output is complete unless generation stops early, for example at `max_tokens`. The source article calls this feature "Guided Decoder".

Function call / tool call
: The model is trained to output a function call in a fixed format, and the engine parses the call from the output. This lets the LLM call external tools. Guided decoding can also force the call to match the argument schema of the function.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig12-function-calling.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Five steps of a function call between an app and a model." caption="<b>Figure 12.</b> The flow of a function call. (1) The app sends the tool definition <code>get_weather(city)</code> and the user question to the model. (2) The model does not answer yet. It returns a call, <code>get_weather(city=&quot;Seoul&quot;)</code>. Guided decoding can keep this format valid (in vLLM, for a named tool or <code>tool_choice=&quot;required&quot;</code>). (3) The app runs the function. (4) The app sends the result <code>{&quot;temp_c&quot;: 9}</code> back to the model. (5) The model writes the final answer." %}
  </div>
</div>

### Parallelism and overlap

DP / SP / TP / PP (data, sequence, tensor and pipeline parallelism)
: These are the four basic ways to split a model or its requests across GPUs. DP splits the batch of requests. SP splits the token sequence. TP splits the hidden dimension, which means the weight matrices. PP splits the layers.

A small figure in the source article maps DP, SP and TP to the three axes of the activation tensor `[bs, seq_len, hidden_size]`. Figure 13 adds PP to that picture.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig13-dp-sp-tp-pp.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Four panels show which tensor axis or model part DP, SP, TP and PP split between two GPUs." caption="<b>Figure 13.</b> What each kind of parallelism splits. Yellow is the share of GPU0, and blue is the share of GPU1. DP copies the full model and splits only the requests, so it needs almost no communication, but each GPU must hold the full model. TP splits the matrices of one layer, so you use it when the model does not fit on one GPU. But TP must combine the results of every layer with an all-reduce. The SP panel shows the ring style, in which the GPUs pass KV blocks around a ring. Other SP methods use all-to-all or all-gather instead." %}
  </div>
</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig14-hybrid-parallelism.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="A dense model uses TP inside hosts and PP across hosts; an MoE model uses DP for attention and EP for experts." caption="<b>Figure 14.</b> Hybrid parallelism. On the left is a dense model of the Llama 3 405B class. In each host, 8 devices (TP0 to TP7) split one layer, and the two hosts split the layers as PP stage 0 and stage 1. On the right is an MoE model of the DeepSeek-V3 class. Each device holds one attention replica (DP) and a different set of experts (EP). The crossing lines are all-to-all communication: each token goes to the devices that hold its experts." %}
  </div>
</div>

TBO / DBO (two-batch overlap / dual-batch overlap)
: The engine splits the requests into two micro-batches and runs them out of phase. While one micro-batch does attention, the other does the dispatch and combine communication for the experts. The compute stream and the communication stream run at the same time, so each hides the wait time of the other.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig15-tbo.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Compute and communication streams overlap for two micro-batches." caption="<b>Figure 15.</b> Two-batch overlap. The CPU launches the kernels in order. In prefill, a dispatch blocks the CPU until it gets metadata from the other GPUs. So the CPU queues the next compute kernel before each dispatch. On the compute stream, attention and MLP run back to back. On the communication stream, dispatch and combine for the other micro-batch run at the same time. Where a communication block sits under a compute block, its time is hidden." %}
  </div>
</div>

### Attention kernels

FA (FlashAttention)
: An algorithm that computes attention fast. FlashAttention-1 uses the online softmax method to fuse QKᵀ, softmax and ×V into one kernel. So it never writes the large attention matrix to HBM (the global memory of the GPU) and reads it back. FlashAttention-2 divides the work better and uses fewer operations that are not matrix multiplications. This helps on NVIDIA A100 and H100 class GPUs, where the units outside the tensor cores are slow.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig16-flashattention-online-softmax.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="FlashAttention processes two key-value blocks and rescales the first partial output when it adds the second." caption="<b>Figure 16.</b> FlashAttention processes K and V in two blocks. Solid boxes are data in HBM. Dashed boxes are intermediate values that stay in on-chip SRAM and never go to HBM. As in FlashAttention-2, the partial output O⁽¹⁾ also stays on chip, and only the final O⁽²⁾ goes to HBM. (FlashAttention-1 writes O, l and m back to HBM after each block.) The step that makes this work is on the right. For block 2, the kernel rescales the earlier output O⁽¹⁾ by l⁽¹⁾/l⁽²⁾ and adds the new part. The result is the exact softmax, without the full matrix." %}
  </div>
</div>

$$
\begin{aligned}
l^{(2)} &= l^{(1)} + \sum_i \exp\!\big(S^{(2)}_i\big) \\
O^{(2)} &= \frac{l^{(1)}}{l^{(2)}}\,O^{(1)} + \frac{A^{(2)}}{l^{(2)}}\,V^{(2)}
\end{aligned}
$$

Here $l$ is the sum of the exponentials so far (the softmax denominator), and $A = \exp(S)$ is the weight before normalization. Real kernels also update the row maximum $m$ for numerical stability. The figure leaves $m$ out.

Flash-Decoding
: In decode, the query has only one token. FlashAttention splits the work over the batch, the heads and blocks of the query. In decode the query is one token, so with a small batch and a long context, most of the GPU sits idle. Flash-Decoding splits the long KV sequence into parts instead and computes the parts in parallel. At the end, it combines the parts with a softmax correction.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig17-flash-decoding.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="A long key-value sequence split into five parts that one query processes in parallel." caption="<b>Figure 17.</b> Flash-Decoding. The long key and value sequence splits into 5 parts. The same query computes a partial output with each part, in parallel. Each part also keeps its softmax statistics (the row maximum m and the sum l). A last step rescales the partial outputs, as in Figure 16, and sums them into the final output." %}
  </div>
</div>

FlashInfer
: An attention kernel library that inference frameworks use under the hood. The shapes of the query and the KV differ between inference stages, so FlashInfer has optimized FlashAttention kernels for prefill, append and decode. Append uses the same API as prefill. The source article writes "FIA (Flash Infer Attention)", but FlashInfer itself does not use the name FIA. On Huawei Ascend NPUs, FIA means Fused Infer Attention, a fused operator that also picks a prefill or decode kernel by the query shape. The figure in the source article matches the FlashInfer material, so this post follows FlashInfer.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig18-flashinfer-shapes.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Attention masks for decode, append or verify, and prefill." caption="<b>Figure 18.</b> Attention shapes for three cases, over the same five keys (columns) and queries (rows). In decode (left), one new query sees every key, which is a thin strip. In append or verify (middle), a few new queries (draft tokens or added prompt) see all earlier keys and each other. In prefill (right), 5 prompt tokens see each other through a full causal triangle. The bars show the work: 5, 12 and 15 score cells." %}
  </div>
</div>

Linear attention
: The cost of softmax attention grows with the square of the sequence length. Linear attention models replace it with the update of a state of fixed size, so the cost grows linearly with the length. They need less memory and compute faster. The source article says that hybrid models with some linear attention layers keep getting more accurate. It adds that in some areas they reach the level of standard (full) attention.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig19-ssm-timeline.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Timeline of state space models from linear attention in 2020 to Gated DeltaNet in 2024." caption="<b>Figure 19.</b> A short history of state space models (SSMs). The update hₜ = A·hₜ₋₁ + B·xₜ keeps one state of fixed size, so there is no KV cache that grows with the length. Linear attention (2020) turns attention into a running state. S4 (2021) is linear in length but weak at selective copying. In Mamba (2023), A, B and C depend on the input at each step. Mamba-2 (2024) shows that an SSM is a form of linear attention, so it can use tensor cores. Gated DeltaNet (2024) adds gating and a delta-rule update." %}
  </div>
</div>

### Load balancing

DPLB (data-parallel load balancing)
: A scheduler that spreads requests evenly over the engine instances of a DP deployment. Each engine reports its running and waiting requests. In vLLM, a coordinator collects these reports and publishes them to the API server. The API server then sends the new request to a less busy engine.

EPLB (expert-parallelism load balancer)
: In an MoE model, many tokens can go to the same few experts. Then only the GPUs that hold those experts are busy. EPLB copies or moves the popular experts so that the load is even across devices.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig20-dplb.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Four engines report their load to a DP coordinator, which publishes it to the API server; the load balancer in the API server sends a request to engine 0." caption="<b>Figure 20.</b> DPLB, in the way vLLM does it. Engines 0 to 3 report their load (waiting and running requests, KV cache usage) to a DP coordinator. The coordinator publishes these stats to the API server. The load balancer in the API server then sends the new request to the least busy engine (here, engine 0)." %}
  </div>
</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig21-dplb-eplb-pd.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="A PD-disaggregated deployment with request load balancers in the API server and an expert placement planner." caption="<b>Figure 21.</b> PD disaggregation with both DPLB and EPLB. In the API server, a prefill load balancer and a decode load balancer spread requests over the instances from their load stats (the DPLB layer). EPLB is not in the path of the tokens. It is a planner that reads the expert load stats and decides where each expert goes. Here it puts the hot expert E0 on two GPUs in each instance, so that the GPU load is even (the EPLB layer). For simplicity, all four instances show the same placement. In practice, each instance gets a plan from its own load stats. The KV transfer arrow moves the KV cache from a prefill instance to a decode instance. The external KV cache storage at the bottom is optional. It stores and shares KV outside the instances." %}
  </div>
</div>

The feature list of the source article also names multi-step scheduling: the engine plans several decode steps in one scheduler call, which cuts CPU overhead. The source article gives no more detail, so the name is enough for now. vLLM had this feature in its older V0 engine (`--num-scheduler-steps`). The current V1 engine does not have it.

## Performance metrics: what to measure {#metrics}

Inference metrics measure the quality of a service by how fast it makes tokens. They fall into two groups. The first group is the latency that one request sees (TTFT, ITL, E2EL). The second group is the throughput of the whole system (TPS, QPS).

TTFT (time to first token)
: The time from the request to the first output token. Most of it is prefill time, so TTFT is the main prefill metric.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig22-ttft-range.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Request timeline with TTFT marked from the request to the first token." caption="<b>Figure 22.</b> The range of TTFT: tokenization, then prefill on the GPU, which makes the first token, then detokenization of that token. Each later decode step (d1 to d4) makes one more token. In real measurements, the wait in the queue also counts toward TTFT." %}
  </div>
</div>

E2EL (end-to-end latency)
: The time from the prompt input until the system returns the full result.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig23-e2el-range.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Request timeline with E2EL marked from the request to the last token." caption="<b>Figure 23.</b> E2EL also covers all the decode steps after prefill. The only difference from Figure 22 is the number of decode steps inside the range." %}
  </div>
</div>

ITL (inter-token latency)
: The (average) time to make one token in the decode stage. The common formula, from the GenAI-Perf definition, is below.

$$
\text{ITL} = \frac{\text{E2EL} - \text{TTFT}}{n - 1}
$$

Here $n$ is the number of output tokens. TTFT already contains the first token, so the formula divides by the $n - 1$ gaps that remain.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig24-itl-range.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Request timeline with the gaps between consecutive tokens marked as ITL." caption="<b>Figure 24.</b> Each gap between consecutive tokens (itl-1 to itl-4) is one TBT in the terms of this post, and ITL is their average. The gaps should be equal for all tokens. When they vary, the user sees the text arrive in bursts." %}
  </div>
</div>

TBT (time between tokens)
: The time between two consecutive tokens. It usually means the time of one specific token: TBTᵢ = latencyᵢ − latencyᵢ₋₁.

TPOT (time per output token)
: In the definition of the source article, the average time per token, with the first token included: TPOT = E2EL / n.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig25-latency-example.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Timeline example: TTFT 200 ms, four decode gaps of 80 ms, E2EL 520 ms." caption="<b>Figure 25.</b> An example with numbers that I chose (they are not from the source article). The request has n = 5 output tokens. The first token comes at 200 ms, and each later token takes 80 ms. The top arrow is E2EL, the arrow below it is TTFT, and each short gap is one TBT." %}
  </div>
</div>

<div class="table-responsive" markdown="1">

| Metric                | Formula                               | Value in Figure 25      |
| --------------------- | ------------------------------------- | ----------------------- |
| TTFT                  | time of first token − time of request | 200 ms                  |
| E2EL                  | time of last token − time of request  | 520 ms                  |
| ITL                   | (E2EL − TTFT) / (n − 1)               | (520 − 200) / 4 = 80 ms |
| TBT₃                  | latency₃ − latency₂                   | 360 − 280 = 80 ms       |
| TPOT (source article) | E2EL / n                              | 520 / 5 = 104 ms        |

</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig26-sequence-diagram.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Sequence diagram between user and server with TTFT, ITL and end-to-end latency." caption="<b>Figure 26.</b> The same metrics as a message flow between user and server. TTFT runs from the query to token 0. ITL_0 and ITL_1 are single gaps between consecutive tokens (TBTs in the terms of this post; vLLM reports each gap as ITL). The end-to-end latency runs from the query to token n − 1. Each bar on the server side is one decode step." %}
  </div>
</div>

<blockquote class="block-note" markdown="1">

##### The names and formulas differ between tools

The source article notes that some tools, such as vLLM, compute ITL the way this post defines TBT: one gap per token. (More exactly, `vllm bench serve` measures one gap per streamed chunk, which is usually one token.) They compute TPOT the way this post defines ITL: (E2EL − TTFT) / (n − 1). Most benchmarks today define TPOT as the average without the first token. So it is safer to treat TPOT = E2EL / n, with the first token included, as a minority definition. In the example above, the two definitions give 104 ms and 80 ms, which is a large difference. Before you compare benchmark numbers, always check which formula they use.

</blockquote>

### Throughput and service targets

QPS (queries per second)
: The number of requests that the system completes per second. The source article gives the formula QPS = T / Σ latency(i).

TPS (tokens per second)
: The total number of output tokens per second: TPS = n_tokens / (T_y − T_x), in tok/s. Here T_x is the time when the first request is sent, and T_y is the time when the last response arrives.

QPM (queries per minute)
: The number of requests that the system completes per minute.

TP90 / TP99 (top percentile, often called P90 / P99)
: The value that 90% (or 99%) of requests meet. For example, when the TP99 of TTFT is 1 second, 99 of 100 requests got their first token within 1 second. TP50 (the median) is similar.

RPS (requests per second)
: In a load test, the number of requests that the test sends per second. It is a main setting of a throughput test. Some tools, such as those from NVIDIA, use RPS for the requests completed per second, which is QPS in this post. Check which one a report means.

Ramp-up test
: A test that changes the RPS step by step and records how the service performance changes.

SLO (service level objective)
: The quality target that the service must meet. For example: "In the last period, did at least 99% of the requests get 20 tok/s or more each?" This example uses the output speed of each request (TPS per user, close to 1 / ITL for long outputs), not the system TPS above. 20 tok/s per user means an ITL of 50 ms or less.

MFU (model FLOPs utilization)
: The fraction of the theoretical compute of the GPU that the model actually uses.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig27-tps.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Overlapping request bars that all lie inside a measurement window from Tx to Ty." caption="<b>Figure 27.</b> How to compute TPS, after the definition of NVIDIA. Tx is the time when the first request is sent, and Ty is the time when the last response of the last request arrives. All requests R1 to Rm lie inside this window, and all of their output tokens count. TPS is the total, n_tokens, divided by Ty − Tx. The benchmark setup before Tx and the teardown after Ty do not count. (Some tools, such as LLMPerf, divide by T_end − T_start instead.)" %}
  </div>
</div>

<blockquote class="block-note" markdown="1">

##### A closer look at QPS

The QPS formula of the source article is ambiguous, because T also appears in the range of the sum. A common definition in practice is the number of requests completed in the window, divided by the length of the window in seconds. A server runs many requests at the same time, so a division by the sum of latencies underestimates the throughput.

Latency and throughput also pull against each other. A larger batch raises TPS, but it also raises the ITL of each request. An SLO sets the limit in this trade-off, for example: "the highest throughput that keeps latency under this line".

</blockquote>

## Sampling: rules for the next token {#sampling}

At each step, the model gives a score (a logit) for every token in the vocabulary. Sampling turns these scores into probabilities, keeps some candidates, and picks one of them. The source article puts sampling at the end of the metrics section. This post gives it a section of its own.

Temperature
: The engine divides the logits by T and then applies softmax, which changes the shape of the whole distribution. With T < 1, the distribution becomes sharper, and the top token becomes more likely. With T > 1, it becomes flatter, and the answers vary more.

Top-k
: Sort the tokens by probability and keep only the top k. The range is 1 to vocab_size. In vLLM, 0 (the default) or −1 turns the filter off. In Hugging Face Transformers, 0 turns it off.

Top-p (nucleus sampling)
: Sort the probabilities from high to low as $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(V)}$. Keep the first k tokens, where k is the first point at which the cumulative sum $C_k = \sum_{i=1}^{k} p_{(i)}$ reaches P or more. P is between 0 and 1.

Min-p
: Keep only the tokens with a probability of at least min-p times the top probability. The source article gives the range as "> 0", but in practice the value is between 0 and 1.

$$
p_i \ge \text{min-p} \times p_{\max}
$$

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig28-sampling-rules.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Five bar charts of one distribution after temperature, top-k, top-p and min-p." caption="<b>Figure 28.</b> Five tokens A to E with the probabilities 0.40, 0.25, 0.15, 0.12 and 0.08 (values that I chose). Each panel applies one rule. Dashed bars are tokens that the rule removes. As panel 3 shows, the engine scales the remaining probabilities so that they add up to 1. Panels 4 and 5 keep the original values, so that you can see the cumulative sums and the threshold. The engine scales those survivors in the same way. In panel 5, the dashed line is the threshold 0.25 × 0.40 = 0.10." %}
  </div>
</div>

<blockquote class="block-note" markdown="1">

##### Worked example: follow the numbers

With T = 0.5: softmax(z/T) is the same as $p_i^{1/T}$ after normalization, so here the engine normalizes $p_i^2$. The squares are 0.16, 0.0625, 0.0225, 0.0144 and 0.0064, and their sum is 0.2658. After the division, they become 0.60, 0.24, 0.08, 0.05 and 0.02. So A goes from 0.40 to 0.60.

With top-k = 3, keep A, B and C and divide by their sum, 0.80. The result is 0.50, 0.31 and 0.19.

With top-p = 0.75, the cumulative sums are 0.40, 0.65 and 0.80. The engine stops at C₃ = 0.80 ≥ 0.75 and keeps three tokens. The result is the same as with top-k only by chance. Top-p is an adaptive cut: it keeps one or two tokens when the distribution is sharp, and many tokens when it is flat.

With min-p = 0.25, D has 0.12 ≥ 0.10, so D stays, and only E goes. Min-p is also adaptive, because the threshold scales with p_max. The order of these steps in real engines (penalties, then temperature, then top-k, top-p and min-p, then the sample) differs a little between implementations.

</blockquote>

Frequency penalty
: Lowers the logit of each token that already appeared, in proportion to the number of times it appeared. More appearances give a larger cut.

Presence penalty
: Subtracts a fixed value from the logit of each token that appeared at least once. The cut happens once per token, whatever the count.

Repetition penalty
: Reduces tokens that already appeared. The source article compares it to the frequency penalty. But vLLM and Hugging Face apply it once per token, whatever the count, and they also count prompt tokens. They scale the logit: they divide a positive logit by the penalty and multiply a negative one by it.

Beam search
: At each step, the search keeps only the k candidate sequences with the highest cumulative probability (k is the beam width) and prunes the rest. You can think of it as top-k plus pruning, applied to whole sequences.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig29-beam-search.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Beam search tree with beam width 2 that prunes the branch of I like cats." caption="<b>Figure 29.</b> Beam search with beam width 2. At step 1, the search keeps the top two words after &quot;I&quot;: &quot;like&quot; (0.6) and &quot;love&quot; (0.3). At each step, it keeps only the two sequences with the highest cumulative probability. At step 3, the upper branch gets a ✗. Its sequences score lower than the two on the lower branch, so the search prunes it." %}
  </div>
</div>

<blockquote class="block-note" markdown="1">

##### Worked example: beam search

At step 2, "I like cats" = 0.6 × 0.3 = 0.18 and "I love to" = 0.3 × 0.7 = 0.21 stay. At step 3, the best extension of "I like cats" gives 0.18 × 0.3 = 0.054. The extensions of "I love to" give "read" with 0.21 × 0.3 = 0.063 and "cook" with 0.21 × 0.4 = 0.084. The top two (0.084 and 0.063) both come from the lower branch, so the search prunes the upper branch.

A greedy search takes only the top token at each step, so it would take "like" (0.6) at step 1. Beam search finds that the "love" path wins in the end, although it was second at first. The cost is extra computation that grows with the beam width.

</blockquote>

## Engines and frameworks {#engines}

vLLM
: A widely used inference framework from the Sky Computing Lab at UC Berkeley. It started from PagedAttention. Its strengths are many features and a large ecosystem.

SGLang
: Another main framework, from UC Berkeley and Stanford. It started from RadixAttention, and its main claim is speed.

Dynamo
: The distributed inference system of NVIDIA. It runs on top of engines such as vLLM and SGLang and adds PD disaggregation, KV-aware routing and KV offloading.

AIBrix
: A cloud-native LLM inference solution from ByteDance, based on Kubernetes.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig30-vllm-multi-dp-flow.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Request path through the vLLM API server to several engine core processes and back." caption="<b>Figure 30.</b> The vLLM execution flow with several DP engines. (1) The user sends a request to the API server. (2) AsyncLLM tokenizes it, and the Processor does the pre-processing. (3) DPAsyncMPClient sends the requests over zmq to several engine_core processes, also on other nodes. They go from its ROUTER socket to the DEALER socket of each engine. (4) Each engine_core sends its results back on a separate path (zmq PUSH to PULL), and the OutputProcessor collects them. (5) The server sends the response to the user. The class names follow vLLM v0.8 to v0.9. Newer versions use DPLBAsyncMPClient for this case and call the Processor InputProcessor." %}
  </div>
</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig31-dynamo.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Dynamo components: router, prefill and decode workers, NIXL, KV cache manager and planner." caption="<b>Figure 31.</b> The parts of Dynamo. Below the API server, the smart router sends each request to the place that has its KV cache (KV-cache-aware routing). Disaggregated serving splits the work into prefill workers and decode workers, and NIXL moves the KV cache between them. The KV cache manager offloads KV to host memory or object storage. The planner adjusts resources in real time." %}
  </div>
</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig32-aibrix.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="AIBrix gateway, multi-LoRA services on vLLM pods, control plane and KV cache runtime." caption="<b>Figure 32.</b> The structure of AIBrix. The API gateway at the top gives an OpenAI-compatible API, fairness and TPM/RPM limits, and routing that knows the prefix cache and the load. The services in the middle (base and LoRA-1 to LoRA-3) are multi-LoRA serving: the LoRA adapters are spread over vLLM engines in several pods. The control plane has a cold start manager and an autoscaler for LLMs. A distributed KV cache runtime is at the bottom." %}
  </div>
</div>

### Main serving parameters in vLLM

<div class="table-responsive" markdown="1">

| Parameter                     | Meaning                                                                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max-num-seqs`                | The maximum number of sequences in one step (iteration)                                                                                                                     |
| `max-num-batched-tokens`      | The maximum number of tokens in one step. It controls the total token count (the token budget in Figure 11)                                                                 |
| `max-model-len`               | The context length of the model (the maximum of input plus output tokens). If you do not set it, vLLM reads it from the model config                                        |
| `max-input-length`            | The limit on the input length. The source article lists it, but current vLLM has no such flag; `max-model-len` limits the context                                           |
| `load-format`                 | The format to load the weights. `"dummy"` fills the weights with random values, mainly for performance analysis                                                             |
| `dtype`                       | The data type of weights and activations: auto, half, float16, bfloat16, float, float32                                                                                     |
| `gpu-memory-utilization`      | The fraction of the total GPU memory that this vLLM instance can use. vLLM fills the unused part of this budget with KV cache, so the value is a budget, not the real load. |
| `block-size`                  | The number of consecutive tokens in one KV cache block                                                                                                                      |
| `enforce-eager`               | Always run PyTorch in eager mode, with no CUDA graph capture                                                                                                                |
| `pipeline-parallel-size` (pp) | The number of pipeline stages                                                                                                                                               |
| `tensor-parallel-size` (tp)   | The tensor-parallel size: the number of GPUs that share each layer                                                                                                          |
| `data-parallel-size`          | The number of data-parallel replicas                                                                                                                                        |
| `enable-expert-parallel`      | Use expert parallelism instead of tensor parallelism for the MoE layers. EP = DP × TP                                                                                       |

</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig33-gpu-memory-utilization.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="An 80 GB GPU memory bar split into weights, activations, a KV block pool and space for other processes." caption="<b>Figure 33.</b> The numbers are an example. Llama-2-7b-hf (7 billion parameters) in FP16 needs about 14 GB for its weights. The engine takes its share (here 90%, so 72 GB). It subtracts the weights, the peak activations, the CUDA graphs and other memory outside PyTorch (such as the CUDA context and NCCL buffers). It reserves all the rest as KV cache blocks in advance: 72 − 14 − 4 − 2 = 52 GB. The 4 GB and 2 GB are rough example values. Recent vLLM versions count the CUDA graph memory inside this budget. So this value is the upper limit for the engine, not a measured usage. Leave room for the other processes on the same GPU." %}
  </div>
</div>

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig34-moe-dp2-tp2-ep4.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Four GPUs with two DP attention groups of TP 2 and an MoE layer with EP 4." caption="<b>Figure 34.</b> An MoE model with DP = 2, TP = 2 and EP = 4. At the top, two DP attention groups each use TP = 2 for batch0 and batch1 and combine their results with an all-reduce. In the MoE layer, AllGather collects both batches, and each of the 4 GPUs holds different experts (EP = 2 × 2 = 4). ReduceScatter then returns the results to each batch. A dense FFN layer (not MoE) uses only TP inside each DP group." %}
  </div>
</div>

### Main serving parameters in SGLang (memory and scheduling)

The source article shows a table from the SGLang documentation as an image, in Chinese. The table below gives its content.

<div class="table-responsive" markdown="1">

| Parameter                   | Description                                                                                                                                                               | Default  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `mem_fraction_static`       | The fraction of GPU memory for static memory, such as model weights and the KV cache. Increase it if the KV cache cannot be built. Decrease it if CUDA runs out of memory | None     |
| `max_running_requests`      | The maximum number of requests that run at the same time                                                                                                                  | None     |
| `max_total_tokens`          | The maximum number of tokens that the KV cache can hold. Mainly for debugging                                                                                             | None     |
| `chunked_prefill_size`      | Run prefill in chunks of this size. A larger value makes prefill faster but uses more VRAM. Decrease it if CUDA runs out of memory                                        | None     |
| `max_prefill_tokens`        | The token budget of one prefill batch. The real limit is the larger of this value and `context_length`                                                                    | 16384    |
| `schedule_policy`           | The scheduling policy: the order in which one engine processes the waiting prefill requests                                                                               | `"fcfs"` |
| `schedule_conservativeness` | How cautious the engine is when it accepts new prefill requests. A high value can cause starvation, and a low value can slow down decode                                  | 1.0      |
| `cpu_offload_gb`            | The RAM (in GB) to reserve for offloading model parameters to the CPU                                                                                                     | 0        |

</div>

The parameters change between engine versions. Before you use them, check the official vLLM and SGLang documentation.

### Main modules of the vLLM engine

The source article describes the vLLM engine as three modules: the scheduler, KV cache management and the engine core. Figure 35 shows how they work together in one loop.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig35-vllm-engine-step.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="One vLLM engine step between the API server process and the EngineCore process." caption="<b>Figure 35.</b> One vLLM engine step, simplified. The API server process and the EngineCore process talk through zmq sockets. Inside EngineCore, the scheduler decides which requests compute how many tokens in this step, and it gets blocks from the KV manager. The model runs and samples by that plan (the scheduler output), and the results go back. That is one loop." %}
  </div>
</div>

Scheduler
: The scheduler looks at the system resources (the token budget and the KV blocks) and at the state of the requests. At each step, it decides which tokens of which requests to compute.

A large flowchart in the source article, in Chinese, shows how the scheduler builds one step. In text, the steps are:

1. The scheduler starts with the running queue. It goes through the running requests from the front while the token budget is above 0. For each request, it counts the new tokens to compute in this step. If the count is 0, it skips the request.
2. It asks the KV manager for KV blocks. If there are not enough blocks, it preempts the last request in the running list. (This is the default FCFS policy. The priority policy preempts the request with the lowest priority.) It moves that request back to the waiting queue, frees its blocks, and tries again.
3. When the blocks are available, it adds the request to the list for this step. It subtracts the tokens of the request from the budget and goes to the next request.
4. Next comes the waiting queue. If a preemption just happened, resources are short, so the scheduler skips the waiting queue for this step.
5. Otherwise, the scheduler takes requests from the front of the waiting queue. Some requests must wait. For example, the FSM for structured output is not compiled yet, or the batch already has the maximum number of LoRA adapters. These requests go to a skipped list. The scheduler gives the other requests tokens and blocks and moves them to running. A request in the state WAITING counts as new, and a request in the state PREEMPTED counts as resumed.
6. At the end, the scheduler puts the skipped requests back into the waiting queue. It then builds the scheduler output from the new requests and the existing ones (resumed and running).

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig36-scheduler-flowchart.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Flowchart of one vLLM scheduling step with running, waiting and finish zones." caption="<b>Figure 36.</b> The scheduler flowchart for the six steps above. The blue zone handles the running queue, the yellow zone handles the waiting queue, and the green zone handles the skipped requests and the output. The small inset shows a preemption: the last request of the running list moves to the front of the waiting queue." %}
  </div>
</div>

KV cache management
: Built on PagedAttention. It has a logical layer (the blocks that a request sees) and a physical layer (the real blocks in GPU memory). This is similar to virtual memory management in an operating system.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig37-kv-manager-model-runner.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="KV manager, scheduler and model runner with block table and slot mapping." caption="<b>Figure 37.</b> KV management in vLLM. The KV manager on the left manages the logical blocks in the KV pool and gives block IDs to the scheduler. The scheduler passes its output to the model runner on the right. The block table of the model runner (request to block indices) points to the physical blocks. At the bottom, slot_mapping tells where in physical memory to write the K and V of each new token. Here, the new token of request A goes to slot 3 × 4 + 0 = 12." %}
  </div>
</div>

Engine core (execution unit)
: Runs the work of one step that the scheduler gives it. This includes the KV transfer and the model execution.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig38-engine-core-threads.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Input thread, busy loop and output thread inside the vLLM engine core." caption="<b>Figure 38.</b> The three threads inside the engine core. process_input_sockets puts the requests from zmq into input_queue. run_busy_loop drains the queue (add_request, abort_requests and others) and repeats scheduler.schedule() and execute_model(scheduler_output) in _process_engine_step. The results go through output_queue, and process_output_sockets sends them out with zmq PUSH. The I/O runs in separate threads, so the GPU loop does not wait for the network." %}
  </div>
</div>

## Software and hardware basics {#basics}

CoT (chain of thought)
: A prompting method that makes the model show its reasoning steps before the answer. It also matters for inference systems, because it adds many output tokens and so puts more load on decode.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig39-chain-of-thought.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Standard prompting gives a wrong answer; chain-of-thought prompting shows steps and gives the right answer." caption="<b>Figure 39.</b> Standard prompting and chain-of-thought prompting. On the left, the example shows only the answer, and the model gets the apple question wrong (21). On the right, the example shows the reasoning steps. The model also writes its steps, &quot;15 − 8 = 7. 7 + 6 = 13.&quot;, and gets the right answer, 13. The method is from Wei et al. (2022). The example is new." %}
  </div>
</div>

FCFS (first-come, first-served)
: A scheduling policy that serves the earliest request first. It is the default `schedule_policy` in SGLang.

GPTQ
: A one-shot weight quantization method for GPT-style models, with no retraining. It uses approximate second-order information (the Hessian) to correct the quantization error.

AWQ (activation-aware weight quantization)
: A quantization method that looks at the activation distribution and protects the important weight channels. Before quantization, it multiplies each input channel of the weights by its own scale, which it picks from activation statistics. It then quantizes the weights in groups, often of 128 weights.

Llama-2-7b-hf
: A naming rule: a Llama 2 model with 7 billion parameters, stored in the Hugging Face format.

Round robin
: Each task runs for a fixed time slice (for example 10 ms) in turn. If A, B and C arrive in that order, each runs for 10 ms in turn: A, B, C, then A again. Every task gets the same share, so the policy is fair, and no task starves (waits a long time without a turn).

EOS (end of sequence)
: A special token that marks the end of a sequence. When the model generates EOS, the output stops. EOS is the exit condition of the decode loop in Figure 1.

logprobs (log probabilities)
: The engine works with ln(p) instead of the probability p. A product of many probabilities quickly comes close to 0 and falls below the range of floating-point numbers (underflow). Logarithms change the product into a sum, so this problem does not occur.

<blockquote class="block-note" markdown="1">

##### Worked example: why logs

Take the example of the source article, 0.9²⁰⁰. ln(0.9) ≈ −0.105, so after 200 multiplications the log value is about −21.1. This means that 0.9²⁰⁰ ≈ 7 × 10⁻¹⁰. FP32 can still hold this value. But some token probabilities are near 0.01, and a sequence can be thousands of tokens long. Then the product easily drops below the FP32 minimum (about 10⁻³⁸, or about 10⁻⁴⁵ with subnormal numbers). In log space, the same result stays an ordinary number such as −21.1. Beam search also computes its cumulative probabilities as sums of logprobs.

</blockquote>

Reasoning parser
: Separates the output of a reasoning model into the thinking part (for example, the text inside think tags) and the final answer.

Stream reasoning (streaming output)
: The system sends the results one by one as it generates them. Non-streaming output is the opposite: the system returns everything at once when it is done. With streaming, the user feels TTFT and ITL more than E2EL.

Structured output
: A feature that makes the output follow a structure, such as a JSON format. Guided decoding (above) implements it. The source article says that the "input" follows the structure, but the constraint applies to the output.

The source article gives an example that applies guided decoding with an SQL grammar. Here it is as code instead of an image.

```python
# Guided decoding with a grammar
simplified_sql_grammar = """
    ?start: select_statement
    ?select_statement: "SELECT " column_list " FROM " table_name
    ?column_list: column_name ("," column_name)*
    ?table_name: identifier
    ?column_name: identifier
    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""
guided_decoding_params = GuidedDecodingParams(grammar=simplified_sql_grammar)
```

The first rule, `start`, says that the full output must be one SELECT statement. The rule `select_statement` forces the order "SELECT columns FROM table". The rule `column_list` allows several column names with commas between them. The rule `identifier` uses a regular expression for the name rule: the first character is a letter or an underscore. The last line passes this grammar to `GuidedDecodingParams` in vLLM. At each step, the engine then sets the probability of every token that the grammar does not allow to 0. Newer vLLM versions removed the `guided_*` API in v0.12.0. There, the same grammar goes into `StructuredOutputsParams(grammar=...)`.

RPC (remote procedure call)
: A protocol that lets a program call a function on another computer across the network, as if it were a local function. You do not need to know the network details below it.

Prometheus
: An open-source monitoring and alerting system, built on a time-series database. Inference servers often use it to collect metrics such as TTFT and throughput.

ORCA (Open Request Cost Aggregation)
: A standard for how backend servers report their real-time load metrics. A load balancer in front of them, such as an Envoy proxy, reads these reports and spreads the traffic better. The source article names Envoy as an example of a server that reports its load. In ORCA, Envoy is usually the load balancer that reads the reports.

FlashMLA (multi-head latent attention)
: A library of attention kernels from DeepSeek. It started in February 2025 as an MLA decode kernel for Hopper GPUs. Later releases added prefill and sparse attention kernels.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig40-flashmla.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Two warp groups in a FlashMLA kernel that overlap memory loads and matrix math." caption="<b>Figure 40.</b> Inside the first FlashMLA decode kernel (February 2025 release, for Hopper GPUs). Two warp groups of 128 threads share the work. Warp group 0 computes the QKᵀ scores, applies the mask and softmax, and stores the probabilities p in shared memory (SMEM). Warp group 1 loads the next K block in advance and waits for p. It then computes the second half of the multiplication with V, while group 0 computes the first half. Grey boxes are tensor-core GEMMs, green boxes are shared-memory traffic, and dashed lines are sync points. The two groups overlap memory loads and math. In April 2025, DeepSeek replaced this schedule with a &quot;seesaw&quot; schedule, in which both warp groups compute QKᵀ and softmax on alternating K blocks." %}
  </div>
</div>

## Python libraries common in inference {#libraries}

triton
: A language and compiler for writing GPU kernels in Python at the block level. Older versions included a few ready-made kernels (`triton.ops`). Triton 3.2 removed them, and kernels such as `triton_kernels` now come as a separate package.

<div class="row mt-3 justify-content-center">
  <div class="col-12 mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/posts/llm-inference-key-terms/fig41-cuda-vs-triton.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="CUDA programs each thread in a block; Triton programs a block as one range of data." caption="<b>Figure 41.</b> The host launches a kernel, and the device runs a grid of blocks. The difference is inside a block. In CUDA (left), you program each thread. In Triton (right), you program the range of data that a block handles, and the compiler maps the threads. So Triton kernels are short and easier to write." %}
  </div>
</div>

ray
: A distributed execution library with flexible resource scheduling. It was first designed for reinforcement learning. In inference, it combines the GPUs of several machines for parallel execution.

lmcache
: A library that stores and transfers KV cache. Its focus is a lower TTFT through reuse.

nixl (NVIDIA Inference Xfer Library)
: The data transfer library of NVIDIA for inference. In Dynamo (Figure 31), it moves the KV cache between prefill and decode.

mooncake
: A library for distributed KV cache storage and transfer. Its TransferEngine is a common choice for cache transfer.

pytorch
: A general-purpose deep learning framework.

punica
: A library for multi-LoRA serving. Its SGMV kernel batches requests for many different LoRA adapters on one shared copy of the base model. The project is now archived.

asyncio
: The standard Python library for concurrent code on a single thread, with an event loop. While one task waits for a network request or for file I/O, the loop switches to another task. Inference servers use it for asynchronous work on the CPU side.

```python
async def main():
    outputs = await llm.async_generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")
```

This code comes from an image in the source article. `async def` defines a coroutine. While `await llm.async_generate(...)` waits for the generation to finish, the event loop can serve other requests. When the results arrive, the code prints each prompt with its generated text.

uvloop
: A faster replacement for the asyncio event loop.

tqdm
: A library that makes progress bars easy. The name comes from the Arabic word "taqaddum" (progress).

zmq (ZeroMQ)
: A high-performance messaging library for distributed and concurrent applications. The vLLM API server and EngineCore use it to talk to each other (Figures 30, 35 and 38).

fastapi
: A web framework that people often use to build API servers. Its APIRouter groups routes, so the code is easier to read and maintain.

## Summary {#summary}

Material that goes deeper into PagedAttention, PD disaggregation, speculative decoding, FlashAttention and MoE parallelism uses the abbreviations in this post without explanation. When you meet one, first find where it attaches on the map in Figure 1. Then ask which bottleneck it serves: prefill (compute) or decode (memory).

The source article ends with the same kind of advice. When you first learn a concept, search results and LLM answers can be wrong. If two sources disagree, check the original paper and the official documentation first. The different definitions of ITL and TPOT between tools are a good example.

A quick self-check. After this post, you should be able to:

- explain with Figure 2 why prefill is compute-bound and decode is memory-bound;
- say in one sentence which waste each of continuous batching, PagedAttention, APC, chunked prefill and PD disaggregation removes;
- mark TTFT, ITL, TBT, TPOT and E2EL on a time axis and compute them from given numbers;
- apply temperature, top-k, top-p and min-p to one distribution by hand and find the tokens that remain;
- say what DP, TP, PP, SP and EP each split, and what EP = DP × TP means.

## Sources {#sources}

- kaiyuan, [大模型推理核心概念与术语总结](https://zhuanlan.zhihu.com/p/1983137653336585901) (Zhihu), the source article of this post. Code and notebooks: [CalvinXKY/InfraTech](https://github.com/CalvinXKY/InfraTech).
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), NeurIPS 2024.
- Agrawal et al., [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369), 2023.
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), 2022, and Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023.
- Dao et al., [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), 2023.
- FlashInfer team, [Accelerating Self-Attentions for LLM Serving with FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html), 2024.
- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), 2022.
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323), 2022, and Lin et al., [AWQ](https://arxiv.org/abs/2306.00978), 2023.
- Chen et al., [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547), 2023.
- The AIBrix team, [AIBrix](https://arxiv.org/abs/2504.03648), 2025, and [NVIDIA Dynamo](https://developer.nvidia.com/dynamo).
