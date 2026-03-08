---
layout: post
title: "LLM from scratch - 1.3 Multi Head Self Attention"
date: 2026-03-08 13:00:00
description: "Why Multi Head Attention is important, mathematical formulations, and implementation."
tags: [llm-project, nlp, transformer, attention]
categories: [project, study]
---

# 1. What Is Multi Head Self Attention?

Multi-head self-attention is an extension of single-head self-attention. 

$$ \begin{aligned}
X&: \text{Input sequence shape } (B, T, D_{model})\\
n_{head}&: \text{Number of attention heads}=3 \\
D_{head} &= D_{model} / n_{head}: \text{Dimension of each head}
\end{aligned}
$$

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/llm_from_scratch/Pasted image 20260308230405.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Instead of performing a single attention operation on the whole dimension, we project the queries ($Q$), keys ($K$), and values ($V$) into $n_{head}$ different representation subspaces. We then divide the feature dimension $D_{model}$ into $n_{head}$ pieces, and perform self-attention independently on **each subspace**. 

---

# 2. Why Multi Head?

There is a fundamental question:

> "Why is multi-head attention needed instead of just one large self-attention head?"

- **Capturing Different Representation Subspaces**: By separating the representation into $n_{head}$ distinct heads, each head can learn to focus on different types of relationships between tokens. For example, one head might attend to grammatical structures (like subject-verb relationships), another might focus on semantic meaning, and another on positional relevance.
  
- **Preventing Meaning Mixing**: If we just use one large attention head, the attention distribution (softmax weights) will be forced to average out all these different relationships into a single weighted sum. This **mixes the meanings of the tokens** $\rightarrow$ resulting in a lower resolution of information. Multi-head enables multiple simultaneous attention distributions without this dilution.

---

# 3. Mathematical Formulation

In Multi-Head Self-Attention, the operations can be summarized as:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O $$
$$ \text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) $$

- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{D_{model} \times D_{head}}$ are the projection matrices for head $i$.
- $W^O \in \mathbb{R}^{D_{model} \times D_{model}}$ is the final output projection matrix.

---

# 4. Code Implementation

```python
# from attn_mask import causal_mask
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
  
class MultiHeadSelfAttention(nn.Module):
	"""1.4 Multi-head attention with explicit shape tracing.
	
	Dimensions (before masking):
		x: (B, T, d_model)
		qkv: (B, T, 3*d_model)
		view→ (B, T, 3, n_head, d_head) where d_head = d_model // n_head
		split→ q,k,v each (B, T, n_head, d_head)
		swap→ (B, n_head, T, d_head)
		scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
		weights:(B, n_head, T, T) = softmax(scores)
		ctx: (B, n_head, T, d_head) = weights @ v
		merge: (B, T, n_head*d_head) = (B, T, d_model)
	"""
	def __init__(self, n_head: int, d_model: int, dropout: float = 0.0, trace_shapes: bool = True):
		super().__init__()
		assert d_model % n_head == 0, "d_model must be divisible by n_head"
		self.n_head = n_head
		self.d_model = d_model
		self.d_head = d_model // n_head
		self.qkv = nn.Linear(d_model, 3 * d_model, bias = False)
		self.proj = nn.Linear(d_model, d_model, bias = False)
		self.dropout = nn.Dropout(dropout)
		self.trace_shapes = trace_shapes
	
	def forward(self, x: torch.tensor):
		B, T, D = x.shape
		qkv = self.qkv(x)
		qkv = qkv.view(B, T, 3, self.n_head, self.d_head)
		if self.trace_shapes:
			print("qkv view:", qkv.shape)
			
		q, k, v = qkv.unbind(dim = 2) # (B, T, head, dim)
		q = q.transpose(1, 2) # (B, head, T, dim)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)
		
		if self.trace_shapes:
			print("q:", q.shape, "k:", k.shape, "v:", v.shape)
		
		scale = 1.0 / math.sqrt(k.shape[-1])
		attn = torch.matmul(q, k.transpose(-2, -1)) * scale # (B,heads,T,T)
		
        # We assume causal_mask is defined separately
        # mask = causal_mask(T, device = x.device)
		# attn = attn.masked_fill(mask, float('-inf'))
		w = F.softmax(attn, dim = -1)
		w = self.dropout(w)
		ctx = torch.matmul(w, v) # (B,heads,T,dim)
		
		if self.trace_shapes:
			print("weights:", w.shape, "ctx:", ctx.shape)
		
		out = ctx.transpose(1, 2).contiguous().view(B, T, D) # (B,T,d_model)
		out = self.proj(out)
		
		if self.trace_shapes:
			print("out:", out.shape)
		return out, w
```

**Key Implementation Details:**
- **Fused QKV Projection**: You might notice that `q, k, v` are derived from a single `nn.Linear(d_model, 3 * d_model)` rather than three separate layers. This is treated as one combined tensor primarily for **the advantage of computation**, allowing better GPU utilization through parallel matrix multiplication rather than being a theoretical necessity.
- **Final Projection Layer (`self.proj`)**: After multiplying the weights and values, the outputs of all heads are concatenated together (via `.view(B, T, D)`) and passed through a final linear layer `self.proj`. This allows the model to **mix** the features gathered from the independent heads back together.

---

_Reference_ :
- [LLMs from Scratch – Practical Engineering from Base Model to PPO RLHF](https://www.youtube.com/watch?v=p3sij8QzONQ&t=1304s)
- [Stanford CS231N Spring 2025 Lecture 8: Attention and Transformers](https://www.youtube.com/watch?v=RQowiOF_FvQ&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=8)
