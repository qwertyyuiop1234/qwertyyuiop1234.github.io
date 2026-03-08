---
layout: post
title: "LLM from scratch - 1.1 Positional Encoding"
date: 2026-03-08 11:00:00
description: "Positional encoding explanations and code implementations, including learned and sinusoidal encoding."
tags: [llm-project, nlp, transformer]
categories: [project, study]
---

# 1. Positional Encoding

<div class="row mt-3 justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/llm_from_scratch/Pasted image 20260308021220.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

According to the transformer architecture, the positional encoding is used for indexing the order on tokens.

Let the input $x: (\text{B, T, D})$ 

which:
- $B$: batch
- $T$: timestep / sequence length
- $D$: dimension

## 1.1 Learned Postional Encoding

```python
import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
	def __init__(self, max_len : int, d_model : int):
		super().__init__()
		self.emb = nn.Embedding(max_len, d_model)
		
	def forward(self, x : torch.tensor):
		# B: Batch size, T: Time steps / sequence length
		B, T, _ = x.shape
		pos = torch.arange(T, device = x.device)
		pos_emb = self.emb(pos) # get pos_emb about pos (T, d_model)
		return x + pos_emb.unsqueeze(0) # fit the dimension and broadcast over the batch
```

You might be wondering why we use `nn.Embedding` instead of `nn.Linear`, since both can conceptually serve as a learned linear mapping. The key difference lies in **how they access data** under the hood:

- **`nn.Linear`**: Requires the input to be a one-hot encoded vector, and performs a full matrix multiplication.
- **`nn.Embedding`**: Acts as a simple **Lookup Table**. It takes an integer index and directly retrieves the corresponding vector from memory.

Since position IDs are just discrete integers ($0, 1, 2, ...$), using one-hot encoding followed by matrix multiplication (`nn.Linear`) would be highly inefficient and wasteful. `nn.Embedding` provides a much faster and memory-efficient way to grab the specific positional vector we need.

> **💡 What does `device=x.device` mean?**
>
> In PyTorch, data can reside on either the CPU or a specific GPU. Operations can only be performed if all involved tensors are on the **same device**. By passing `device=x.device`, we ensure that our newly created position indices (`pos`) are generated on the exact same hardware accelerator where our input tensor `x` currently lives, preventing device mismatch errors.


## 1.2 Sinusoidal  Postional Encoding

This method was proposed in the _Attention Is All You Need_ paper.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/llm_from_scratch/Pasted image 20260308025153.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

```python
class SinusoidalPositionalEncoding(nn.Module):
	def __init__(self, max_len : int, d_model : int):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		pos = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
		denom = torch.exp((-math.log(10000.0) / d_model) * torch.arange(0, d_model, 2).float())
		pe[:, 0::2] = torch.sin(pos * denom)
		pe[:, 1::2] = torch.cos(pos * denom)
		self.register_buffer('pe', pe) # register pe at buffer for some reasons
  
	def forward(self, x : torch.tensor):
	    B, T, _ = x.shape
	    return x + self.pe[:T].unsqueeze(0)
```

> **💡 Why implement the denominator using $\log$ and $\exp$?**
> 
> The formula for the denominator in the paper is $10000^{2i / d_{model}}$.
> Directly calculating $10000^{x}$ for large exponents can cause severe **floating-point precision issues** (overflow/underflow) in computers.  
> By utilizing the mathematical property $x = \exp(\log(x))$, we can rewrite $10000^{2i / d} $ as: 
> $\exp(\frac{2i}{d} \cdot \log(10000))$. 
> This computes the same exact values but remains highly **numerically stable** during tensor operations.

### 1.2.1 Deep Dive Into Sinusoidal Positional Encoding

Look at the image above and you'll notice it uses $\sin$ and $\cos$ functions. But why exactly these functions?

Sinusoidal positional encoding identifies a token’s position by mapping it to a **unique pattern** of sine and cosine values across multiple dimensions. 

Think of it like the **hands of a clock**:
- The first few dimensions (high frequency) move very fast, like the second hand, capturing exact local positions.
- The later dimensions (low frequency) move very slowly, like the hour hand, giving a sense of the broader, long-range position in the sequence.

Because each dimension pulses at a **different frequency**, the overall combination acts as a **distinct, continuous signature** for that specific position. This continuous nature is crucial because it allows the model to easily calculate **relative distances** between tokens. Because of trigonometric properties (like $\sin(\alpha + \beta)$), a positional offset can be computed as a simple linear transformation, making it incredibly easy for the attention mechanism to learn relative positions.

The constant $10000$ is simply a design choice (hyperparameter) that sets the maximum wavelength, ensuring the "hour hand" doesn't complete a full cycle even for very long sequences.

The term $2i / d_{model}$ spaces these frequencies smoothly on a **logarithmic scale**, ensuring the model captures positional patterns at various resolutions from short-range to long-range.

---

_Reference_ :
- [LLMs from Scratch – Practical Engineering from Base Model to PPO RLHF](https://www.youtube.com/watch?v=p3sij8QzONQ&t=1304s)
