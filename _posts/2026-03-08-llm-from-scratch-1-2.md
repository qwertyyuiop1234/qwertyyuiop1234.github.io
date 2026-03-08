---
layout: post
title: "LLM from scratch - 1.2 Single Head Self Attention"
date: 2026-03-08 12:00:00
description: "Implementation details of single head self attention and causal masks."
tags: [llm-project, nlp, transformer, attention]
categories: [project, study]
---

# 1. Causal Mask

```python
import torch

def causal_mask(T: int, device = None):
	"""Returns a bool mask where True means *masked* (disallowed).
	Shape: (1, 1, T, T) suitable for broadcasting with (B, heads, T, T).
	"""
	mask = torch.triu(torch.ones((T,T), dtype = torch.bool, device = device), diagonal = 1)
	return mask.view(1, 1, T, T)
```

- `torch.triu` stands for 'triangle upper'.  
- `diagonal = 1` sets all elements above the main diagonal to `True` to mask future tokens.

**Why use a Causal Mask?**
In auto-regressive models (like GPT), predicting the next word relies only on the past and present tokens. The causal mask ensures that the model cannot **look ahead** or **cheat** by giving attention to future tokens that haven't been generated yet.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/llm_from_scratch/Pasted image 20260308214938.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


---

# 2. What Is Single Head Self Attention?

<div class="row mt-3 justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/llm_from_scratch/Pasted image 20260309002814.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Self-attention allows the model to weigh the importance of different tokens in a sequence relative to a specific token. It achieves this using three learned vectors for each token:
- **Query (Q)**: "What kind of information am I looking for?" (The current token asking questions to context)
- **Key (K)**: "What kind of information do I contain?" (Other tokens holding up their identity name tags)
- **Value (V)**: "Here is my actual content." (The underlying representation or meaning of the token)

---

# 3. The Attention Formula

$$Attention(Q, K, V ) = softmax( \frac{QK^T}{\sqrt{d_k}} )V $$

**Step-by-step breakdown:**
1. **$QK^T$ (Calculate Scores):** Take the dot product between the Query of the current token and the Keys of all other tokens. A high dot product means the Key closely matches what the Query is looking for (high relevance).
2. **$\frac{1}{\sqrt{d_k}}$ (Scale):** If the dimension $d_k$ is large, the dot products can grow extremely large or small. This causes the gradients of the Softmax function to **vanish** (become flat). Dividing by $\sqrt{d_k}$ keeps **the variance stable**.
3. **$softmax(\dots)$ (Get Weights):** Converts the raw scores into probabilities (Attention Weights) that sum to 1.
4. **$\times V$ (Get Context):** Multiplies the Attention Weights by the Value vectors. The result is a weighted sum representing the newly contextualized token.

---

# 4. Code Implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from attn_mask import causal_mask

class SingleHeadSelfAttention(nn.Module):
	"""1.3 Single-head attention (explicit shapes).
	
	Dimensions summary (single head):
		X: (B, T, d_model)
		q, k, v: (B, T, d_k)
		scores: (B, T, T)
		Weights: (B, T, T)
		Output: (B, T, d_k)
	"""
	def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shapes: bool = False):
		super().__init__()
		self.q = nn.Linear(d_model, d_k, bias=False)
		self.k = nn.Linear(d_model, d_k, bias=False)
		self.v = nn.Linear(d_model, d_k, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.trace_shapes = trace_shapes
	
	def forward(self, x: torch.Tensor): # x: (B, T, d_model)
		B, T, _ = x.shape
		q = self.q(x) # (B,T,d_k)
		k = self.k(x) # (B,T,d_k)
		v = self.v(x) # (B,T,d_k)
		if self.trace_shapes:
			print(f"q {q.shape} k {k.shape} v {v.shape}")
			
		scale = 1.0 / math.sqrt(q.size(-1))
		attn = torch.matmul(q, k.transpose(-2, -1)) * scale # (B,T,T)
		mask = causal_mask(T, device=x.device)
		attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
		w = F.softmax(attn, dim=-1)
		w = self.dropout(w)
		out = torch.matmul(w, v) # (B,T,d_k)
		
		if self.trace_shapes:
			print(f"weights {w.shape} out {out.shape}")
		return out, w
```

---

_Reference_ :
- [LLMs from Scratch – Practical Engineering from Base Model to PPO RLHF](https://www.youtube.com/watch?v=p3sij8QzONQ&t=1304s)
- [Stanford CS231N Spring 2025 Lecture 8: Attention and Transformers](https://www.youtube.com/watch?v=RQowiOF_FvQ&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=8)
