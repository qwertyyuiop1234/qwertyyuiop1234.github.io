---
layout: post
title: "Deep Dive into MicroGPT by Karpathy"
date: 2026-02-18 02:00:00 +0900
categories: [AI, NLP]
tags: [microgpt, karpathy, autograd, backpropagation, nlp]
description: A detailed walkthrough of Karpathy's MicroGPT, covering dataset preparation, character-level tokenization, a minimal autograd engine (the Value class), Python special methods, and backpropagation via topological sort.
toc:
  beginning: true
---

In this post, I'm reviewing [**MicroGPT by Andrej Karpathy**](https://karpathy.github.io/2026/02/12/microgpt/){:target="\_blank"}, which is an essential resource for anyone interested in understanding how language models work from scratch.

---

# 1. Getting Datasets & Tokenizer

## 1.1 Download Dataset

```python
# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")  # total number of names is 32,033
```

`urllib.request` is a Python standard library module for fetching data from URLs.

The dataset consists of 32,033 human names, each on a separate line. After loading, the names are stripped of whitespace and shuffled randomly.

<br>

## 1.2 Tokenizer

```python
# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs)))  # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars)  # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1  # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")
```

- `uchars` is the sorted set of all unique characters that appear across the names in `input.txt`. Each character is assigned a token ID from `0` to `n-1`.
- `vocab_size` is `len(uchars) + 1` because we need one additional ID for the **BOS** token. Every token — including special tokens — requires its own unique ID in the vocabulary.

> **What is BOS (Beginning of Sequence)?**
>
> BOS is a special token that marks the start of a sequence. During **generation**, the model receives the BOS token as its initial input to begin producing the first real character. Without BOS, the model would have no starting signal and could not initiate generation.
>
> In this character-level model, BOS is the *only* special token — and it **doubles as the EOS (End of Sequence) token**. Each name is wrapped like this:
>
> `[BOS] e m m a [BOS]`
>
> The same token appears at both the beginning and the end. During generation, when the model predicts BOS as the next token, that signals the name is complete and generation stops. This is an elegant design: a single special token handles both roles, keeping the vocabulary minimal.

For a deeper understanding of tokenization strategies, see: [How Tokenizers Work in AI Models (Nebius)](https://nebius.com/blog/posts/how-tokenizers-work-in-ai-models){:target="\_blank"}

---

# 2. Computation Graph

```python
# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
```

<br>

## 2.1 Why `__slots__`?

In a standard Python class, instance attributes are stored in a per-instance `__dict__` dictionary. This is flexible but comes with overhead: each dictionary maintains a hash table, keys, values, and internal bookkeeping.

**Without `__slots__`** (default):
```
instance
  └── __dict__  (a full hash-table dictionary)
        ├── key → value
        ├── key → value
        └── ...
```

**With `__slots__`**:
```
instance
  └── fixed attribute slots  (C-struct-like, no dictionary)
        ├── data
        ├── grad
        ├── _children
        └── _local_grads
```

By declaring `__slots__`, Python allocates a **fixed-size struct** for the listed attributes instead of a dictionary. This:
- **Saves memory** — no dictionary overhead per instance (important when creating thousands of `Value` nodes in a computation graph).
- **Speeds up attribute access** — direct offset-based lookup instead of hash-table lookup.
- **Prevents accidental attribute creation** — assigning to a non-declared attribute raises `AttributeError`.

<br>

## 2.2 Special Methods

```python
def __init__(self, data, children=(), local_grads=()):
    self.data = data                # scalar value of this node (computed during forward pass)
    self.grad = 0                   # derivative of loss w.r.t. this node (computed during backward pass)
    self._children = children       # children of this node in the computation graph
    self._local_grads = local_grads # local derivatives of this node w.r.t. its children

def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))

def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
def log(self):  return Value(math.log(self.data), (self,), (1/self.data,))
def exp(self):  return Value(math.exp(self.data), (self,), (math.exp(self.data),))
def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
def __neg__(self):      return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other):  return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other):  return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

Each operation creates a new `Value` node that records:
1. The **forward result** (`data`)
2. Its **parent operands** (`_children`)
3. The **local gradients** (`_local_grads`) — partial derivatives of this operation's output with respect to each child

This builds a **computation graph** during the forward pass that will later be traversed in reverse for backpropagation.

<br>

### 2.2.1 Normal Methods vs. Reflected Methods

Consider `__add__` and `__radd__`. Why do we need the reflected (reverse) version?

When Python evaluates `3 + Value(5)`:
1. Python first tries `int.__add__(3, Value(5))` → `int` doesn't know how to add a `Value`, so it returns `NotImplemented`.
2. Python then falls back to `Value.__radd__(Value(5), 3)` → this works because `__radd__` calls `self + other`, which triggers `Value.__add__`.

Without `__radd__`, expressions like `3 + Value(5)` or `2 * Value(3)` would raise a `TypeError`.

<br>

### 2.2.2 Why `__sub__` Uses `+ (-other)` Instead of Direct Subtraction

```python
def __sub__(self, other): return self + (-other)
```

This might look roundabout, but it's a deliberate design choice:
- **Reuses existing operations**: Subtraction is decomposed into negation (`__neg__`) and addition (`__add__`), both of which are already implemented with correct gradient tracking.
- **Avoids redundant gradient logic**: If subtraction were implemented as a separate operation, we would need to define and maintain additional local gradient formulas.
- **Keeps the computation graph simple**: Every operation in the graph maps to one of a small set of primitives (`add`, `mul`, `pow`, `log`, `exp`, `relu`), making the backward pass straightforward.

<br>

---

*This post will be updated as I continue studying the remaining sections of MicroGPT.*
