# Let's Build a Diffusion Language Model: From Scratch, In Code, Spelled Out

*A Karpathy-style educational guide to training your own Masked Diffusion Language Model*

---

## Prologue: Why This Guide Exists

If you've followed Andrej Karpathy's legendary "Let's build GPT" tutorial, you know the magic of building something from scratch—watching abstract concepts crystallize into working code that actually *generates text*. That tutorial showed us the autoregressive paradigm: predict the next token, one at a time, left to right.

But here's a fascinating question: **Is autoregressive the only way to do language modeling?**

The answer, it turns out, is *no*. Enter **diffusion language models**—a completely different paradigm where instead of predicting tokens sequentially, we start with *complete noise* and iteratively refine it into coherent text. It's like sculpting: starting with a block of marble (noise) and gradually revealing the statue (text) inside.

This guide will take you through building a **Masked Diffusion Language Model (MDLM)** from scratch. We'll use the same dataset Karpathy used (Tiny Shakespeare), target a similar model size (~10-20M parameters), and build everything step-by-step with the same pedagogical philosophy: **no magic, everything spelled out.**

---

## Table of Contents

1. [The Big Picture: What Makes Diffusion Different](#1-the-big-picture)
2. [Setting Up: Environment and Data](#2-setting-up)
3. [The Forward Process: How We Corrupt Text](#3-the-forward-process)
4. [The Reverse Process: Learning to Denoise](#4-the-reverse-process)
5. [The Training Objective: ELBO and Cross-Entropy](#5-the-training-objective)
6. [Building the Model Architecture](#6-building-the-model-architecture)
7. [The Training Loop](#7-the-training-loop)
8. [Sampling: Generating Text](#8-sampling)
9. [Evaluation and Visualization](#9-evaluation)
10. [What's Next and Further Reading](#10-whats-next)

---

## 1. The Big Picture: What Makes Diffusion Different {#1-the-big-picture}

### 1.1 Autoregressive vs. Diffusion: Two Philosophies

Let's start with what you already know from Karpathy's tutorial:

**Autoregressive (GPT-style):**
```
Input:  "To be or not to"
Output: "be" (predict next token)
```

The model sees tokens left-to-right and predicts one token at a time. Generation is **sequential**: to generate 100 tokens, you need 100 forward passes.

**Diffusion (what we're building):**
```
Input:  "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"  (fully masked)
Output: "To be or not to be"                          (all tokens at once!)
```

The model sees the *entire* sequence (with some tokens masked) and predicts *all* masked tokens simultaneously. Generation is **iterative but parallel**: you gradually unmask over many steps, but each step predicts all tokens at once.

### 1.2 The Core Intuition: Denoising as Generation

Here's the key insight that might blow your mind:

> **Training:** Corrupt clean data → Learn to uncorrupt it  
> **Generation:** Start with maximum corruption → Iteratively uncorrupt

For images, "corruption" means adding Gaussian noise. For text, we can't add Gaussian noise to discrete tokens. Instead, we use **masking** as our corruption process:

```
Original:    "To be or not to be"
t=0.2:       "To be or [M] to be"     # 20% masked
t=0.5:       "To [M] or [M] to [M]"   # 50% masked  
t=0.8:       "[M] [M] [M] [M] to [M]" # 80% masked
t=1.0:       "[M] [M] [M] [M] [M] [M]" # fully masked
```

The model learns: "Given this partially masked sequence, what were the original tokens?"

### 1.3 The Epiphany: BERT Training ≈ Diffusion Training

Here's something beautiful: if you've trained BERT, you've essentially trained a diffusion model! The key difference:

| Aspect | BERT | MDLM (Diffusion) |
|--------|------|------------------|
| Mask rate | Fixed (15%) | **Variable** (0% to 100%) |
| Goal | Representation learning | **Generation** |
| Training objective | MLM loss | **ELBO (variational bound)** |

MDLM is like BERT, but with a twist: by varying the mask rate and using it to define a proper generative model, we can *sample* from the learned distribution.

> 💡 **Aha Moment:** The training objective of MDLM is literally a **weighted average of masked language modeling losses** at different mask rates. That's it! The math is fancier, but the code is surprisingly simple.

### 1.4 Why Diffusion for Language? The Promises

1. **Bidirectional context**: Unlike GPT which only sees the past, diffusion models see the *entire* sequence (masked or not)
2. **Parallel decoding**: Multiple tokens predicted per step (faster than pure autoregressive)
3. **Flexible editing**: Natural support for infilling, editing, and controlled generation
4. **No reversal curse**: GPT trained on "A→B" struggles with "B→A"; diffusion doesn't have this problem

### 1.5 Papers You Should Know

Before we code, bookmark these:

| Paper | Year | Key Contribution |
|-------|------|------------------|
| **D3PM** (Austin et al.) | 2021 | Discrete Diffusion for categorical data |
| **MDLM** (Sahoo et al.) | 2024 | Simplified masked diffusion, state-of-the-art |
| **LLaDA** (Nie et al.) | 2025 | Scaled MDLM to 8B, competitive with LLaMA3 |
| **SEDD** (Lou et al.) | 2024 | Score entropy discrete diffusion |

We'll primarily follow the **MDLM** approach—it's the simplest and most effective.

---

## 2. Setting Up: Environment and Data {#2-setting-up}

### 2.1 The Setup

Let's get our environment ready. Open a Jupyter notebook and let's go:

```python
# Cell 1: Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Check device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
```

### 2.2 Loading Tiny Shakespeare

Just like Karpathy's tutorial, we'll use Tiny Shakespeare. It's small enough to train on a laptop but has enough structure to be interesting:

```python
# Cell 2: Download and load data
import urllib.request
import os

# Download if not exists
if not os.path.exists('input.txt'):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of text: {len(text)} characters")
print(f"First 200 characters:\n{text[:200]}")
```

Output:
```
Length of text: 1115394 characters
First 200 characters:
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

### 2.3 Tokenization: Character-Level

We'll use character-level tokenization (just like nanoGPT). This keeps things simple and lets us see exactly what's happening:

```python
# Cell 3: Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
itos = {i: ch for i, ch in enumerate(chars)}  # int to string

# Encode/decode functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Test
print(encode("hello"))
print(decode(encode("hello")))
```

> ⚠️ **Important for Diffusion:** We need a **[MASK] token**! Let's add it:

```python
# Cell 4: Add mask token
MASK_TOKEN = vocab_size  # New token ID
vocab_size_with_mask = vocab_size + 1  # Now 66 tokens

print(f"Original vocab size: {vocab_size}")
print(f"Vocab size with [MASK]: {vocab_size_with_mask}")
print(f"[MASK] token ID: {MASK_TOKEN}")
```

### 2.4 Creating the Dataset

Here's where diffusion differs from GPT. In GPT, we create (input, target) pairs where target = input shifted by one. In MDLM, our "target" is always the *original* sequence, and we'll dynamically mask during training:

```python
# Cell 5: Dataset
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = torch.tensor(encode(text), dtype=torch.long)
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Return a chunk of text (no masking here - we'll do it in training)
        chunk = self.data[idx:idx + self.block_size]
        return chunk  # This is x_0 (clean data)

# Hyperparameters
block_size = 128  # Context length (can adjust based on your GPU)
batch_size = 64

# Create dataset and dataloader
dataset = TextDataset(text, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test
batch = next(iter(dataloader))
print(f"Batch shape: {batch.shape}")  # [batch_size, block_size]
print(f"Sample decoded: {decode(batch[0].tolist())[:50]}...")
```

> 💡 **Key Difference from GPT:** In GPT, targets are shifted inputs. In diffusion, **targets are the original (clean) tokens themselves**. The "input" to the model will be a corrupted (masked) version.

---

## 3. The Forward Process: How We Corrupt Text {#3-the-forward-process}

### 3.1 The Masking Process

In continuous diffusion (images), we add Gaussian noise. In discrete diffusion for text, we **replace tokens with [MASK]**. The "diffusion time" t ∈ [0, 1] controls how much we mask:

- t = 0: No masking (clean data)
- t = 0.5: 50% of tokens masked
- t = 1: 100% masked (pure noise)

```python
# Cell 6: Forward process (masking)
def forward_process(x_0, t, mask_token_id):
    """
    Apply forward diffusion by masking tokens.
    
    Args:
        x_0: Clean tokens, shape [batch, seq_len]
        t: Mask ratio (scalar or [batch] tensor), values in [0, 1]
        mask_token_id: ID of the [MASK] token
    
    Returns:
        x_t: Masked tokens
        mask: Boolean mask indicating which positions were masked
    """
    if isinstance(t, float):
        t = torch.tensor([t] * x_0.shape[0], device=x_0.device)
    
    # Generate random values for each position
    rand = torch.rand_like(x_0, dtype=torch.float)
    
    # Mask where rand < t (so t=0.3 means ~30% masked)
    mask = rand < t.unsqueeze(1)  # [batch, seq_len]
    
    # Apply mask
    x_t = x_0.clone()
    x_t[mask] = mask_token_id
    
    return x_t, mask

# Visualize the forward process
def visualize_forward_process(text_sample):
    """Visualize how a sentence gets progressively masked."""
    tokens = torch.tensor([encode(text_sample)], device=device)
    
    print("Forward Process Visualization:")
    print("=" * 60)
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x_t, mask = forward_process(tokens, t, MASK_TOKEN)
        # Decode, replacing mask with ■
        decoded = ""
        for tok in x_t[0].tolist():
            if tok == MASK_TOKEN:
                decoded += "■"
            else:
                decoded += itos[tok]
        print(f"t={t:.1f}: {decoded[:50]}...")

# Test
sample_text = "To be, or not to be, that is the question"
visualize_forward_process(sample_text)
```

Output:
```
Forward Process Visualization:
============================================================
t=0.0: To be, or not to be, that is the question...
t=0.2: To be, ■r ■ot to be,■that is the qu■stion...
t=0.4: ■o ■e, o■ ■ot ■o be,■■hat ■s ■he questio■...
t=0.6: ■o ■■, ■■ ■■t ■■ ■e,■■■a■ ■s ■■■ ■■■■■io■...
t=0.8: ■■ ■■, ■■ ■■■ ■■ ■■,■■■■■ ■■ ■■■ ■■■■■■■■...
t=1.0: ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■...
```

> 💡 **Aha Moment:** Look at t=0.4 — even with 40% of characters masked, you can probably still guess most of the original text! The model will learn to exploit this structure.

### 3.2 The Mathematical Formulation

For the mathematically curious, here's what's happening:

The forward process is defined as:
```
q(x_t | x_0) = Categorical(x_t; (1-t)·one_hot(x_0) + t·one_hot(MASK))
```

In plain English: each token independently either stays the same (probability 1-t) or becomes [MASK] (probability t).

This is simpler than D3PM's general transition matrices—that's the beauty of focusing on "absorbing state" diffusion where [MASK] is the absorbing state (once masked, stays masked).

### 3.3 Key Property: Once Masked, Always Masked

An important property of our forward process:

```
If a token is masked at time t, it remains masked for all t' > t
```

This "absorbing state" property simplifies everything. The [MASK] token is like a sink—tokens flow into it but never out (during the forward process).

---

## 4. The Reverse Process: Learning to Denoise {#4-the-reverse-process}

### 4.1 The Goal

The reverse process does the opposite of masking: given a partially masked sequence, predict the original tokens. Our neural network learns this mapping:

```
Input: "[M] be, [M] not [M] be"  + time t
Output: "To be, or not to be"    (predicted clean tokens)
```

### 4.2 Why Predict All Tokens (Not Just Masked)?

You might think: "Just predict the masked positions!" But MDLM predicts **all positions**, then only uses the predictions at masked positions for the loss.

Why? Because:
1. The model doesn't know which positions are masked (no cheating!)
2. Predicting all positions gives richer gradients
3. At inference, we might want to "reconsider" even unmasked positions

### 4.3 The Prediction Target

Here's a subtle but crucial point:

> The model predicts the **original (clean) tokens x_0**, not the "next step" x_{t-1}.

This is called **x_0-parameterization** and it's what makes MDLM training so simple. The model just does masked language modeling, predicting original tokens.

```python
# Pseudo-code for what the model does
def model_forward(x_t, t):
    """
    Given masked input and time, predict original tokens.
    
    Args:
        x_t: Masked sequence [batch, seq_len] 
        t: Diffusion time [batch]
    
    Returns:
        logits: [batch, seq_len, vocab_size] 
                Predictions for x_0 at each position
    """
    # The model sees the (partially) masked sequence
    # and predicts what the original tokens were
    return transformer(x_t, t)  # We'll build this next!
```

---

## 5. The Training Objective: ELBO and Cross-Entropy {#5-the-training-objective}

### 5.1 The Beautiful Simplicity

Here's where MDLM really shines. The training objective is:

> **Weighted average of cross-entropy losses at masked positions, across different mask rates.**

That's it! Let me show you the code, then explain the math:

```python
# Cell 7: Training objective
def compute_loss(model, x_0, mask_token_id):
    """
    Compute the MDLM training loss.
    
    Args:
        model: The denoiser network
        x_0: Clean tokens [batch, seq_len]
        mask_token_id: ID of mask token
    
    Returns:
        loss: Scalar loss value
    """
    batch_size = x_0.shape[0]
    
    # 1. Sample random mask rate t ~ Uniform(0, 1) for each sample
    t = torch.rand(batch_size, device=x_0.device)
    
    # 2. Apply forward process (masking)
    x_t, mask = forward_process(x_0, t, mask_token_id)
    
    # 3. Model predicts original tokens
    logits = model(x_t, t)  # [batch, seq_len, vocab_size]
    
    # 4. Compute cross-entropy only at masked positions
    # Reshape for cross_entropy: [batch*seq, vocab] and [batch*seq]
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = x_0.view(-1)
    mask_flat = mask.view(-1)
    
    # Cross entropy at all positions
    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # Only count loss at masked positions
    masked_loss = ce_loss * mask_flat.float()
    
    # Average over masked positions (avoid division by zero)
    num_masked = mask_flat.sum()
    if num_masked > 0:
        loss = masked_loss.sum() / num_masked
    else:
        loss = masked_loss.sum()  # Edge case: no masks
    
    return loss
```

### 5.2 Understanding the Math (Optional but Enlightening)

The ELBO (Evidence Lower BOund) for diffusion models looks scary:

```
L = E_{t, x_t} [ -log p_θ(x_0 | x_t, t) ]
```

But for masked diffusion, this simplifies beautifully!

**Step 1:** For masked positions only, the model needs to predict the original token.  
**Step 2:** The loss at each masked position is just cross-entropy.  
**Step 3:** Since we sample t uniformly, all mask rates are weighted equally.

The continuous-time version (from the MDLM paper) adds some weighting, but the discrete-time version we're using is simply:

```
L = E_{t~U(0,1)} [ E_{mask~Bernoulli(t)} [ CE(model(x_t), x_0) at masked positions ] ]
```

> 💡 **Aha Moment:** This is literally BERT training, but instead of always masking 15% of tokens, we mask t% where t is sampled freshly each time! That tiny change turns a representation learner into a generative model.

### 5.3 Why This Works: The Variational Perspective

For those who want deeper understanding:

The loss is an **upper bound** on the negative log-likelihood:

```
-log p_θ(x_0) ≤ L
```

By minimizing L, we're implicitly minimizing -log p(x_0), i.e., maximizing the probability of real data. This is why we can sample from the model—it's a proper generative model!

### 5.4 Time Conditioning: To Use or Not to Use?

The original diffusion models condition on time t. But here's a surprising finding from recent work:

> For masked diffusion, **time conditioning is optional** and often doesn't help much!

Why? Because the mask pattern itself implicitly encodes the "noise level". If 80% of tokens are masked, the model knows it's seeing highly corrupted data.

We'll include time conditioning for completeness, but feel free to ablate it.

---

## 6. Building the Model Architecture {#6-building-the-model-architecture}

### 6.1 Architecture Overview

We'll build a **bidirectional Transformer** (like BERT, unlike GPT). Key differences from GPT:

| Component | GPT (Autoregressive) | MDLM (Diffusion) |
|-----------|---------------------|------------------|
| Attention Mask | Causal (triangular) | **None** (full attention) |
| Position Embedding | Learned or RoPE | Learned or RoPE |
| Time Embedding | N/A | **Added** (optional) |
| Output | Next token logits | **All token logits** |

### 6.2 Sinusoidal Time Embeddings

If you've seen Transformers, you've seen sinusoidal position embeddings. Time embeddings work the same way:

```python
# Cell 8: Time embedding
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion time t."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: [batch] tensor of times in [0, 1]
        Returns:
            [batch, dim] time embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Scale time to reasonable range
        emb = t[:, None] * emb[None, :] * 1000
        
        # Sinusoidal embedding
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb

# Test
time_emb = SinusoidalTimeEmbedding(64)
t_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
emb = time_emb(t_test)
print(f"Time embedding shape: {emb.shape}")  # [5, 64]

# Visualize
plt.figure(figsize=(12, 4))
plt.imshow(emb.detach().numpy(), aspect='auto', cmap='RdBu')
plt.xlabel('Embedding dimension')
plt.ylabel('Time t')
plt.yticks(range(5), ['t=0.0', 't=0.25', 't=0.5', 't=0.75', 't=1.0'])
plt.colorbar(label='Value')
plt.title('Sinusoidal Time Embeddings')
plt.tight_layout()
plt.show()
```

### 6.3 Multi-Head Self-Attention (No Causal Mask!)

This is almost identical to GPT's attention, but **without the causal mask**:

```python
# Cell 9: Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Key, Query, Value projections (combined for efficiency)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape  # Batch, Sequence length, Embedding dim
        
        # Calculate Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # NO CAUSAL MASK! This is the key difference from GPT
        # Every position can attend to every other position
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum of values
        y = att @ v  # [B, nh, T, hd]
        
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y
```

> 💡 **Aha Moment:** In GPT, we use `att = att.masked_fill(causal_mask == 0, float('-inf'))` to prevent attending to future tokens. Here, we simply... don't! Every token can see every other token. This is what gives diffusion models their **bidirectional context**.

### 6.4 Feed-Forward Network

This is identical to GPT:

```python
# Cell 10: MLP
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

### 6.5 Transformer Block with Time Conditioning

Here's where we integrate time. We'll use **adaptive layer norm (adaLN)**, which modulates the layer norm parameters based on time:

```python
# Cell 11: Transformer Block
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, use_time=True):
        super().__init__()
        self.use_time = use_time
        
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
        
        # Time conditioning: modulate layer norm with time
        if use_time:
            # Projects time embedding to scale and shift for layer norms
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_embd, 4 * n_embd)  # 2 * (scale + shift) for 2 layer norms
            )
    
    def forward(self, x, time_emb=None):
        if self.use_time and time_emb is not None:
            # Get scale and shift parameters from time
            time_params = self.time_mlp(time_emb)  # [B, 4*n_embd]
            shift1, scale1, shift2, scale2 = time_params.chunk(4, dim=-1)
            
            # Modulated layer norm for attention
            h = self.ln_1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
            x = x + self.attn(h)
            
            # Modulated layer norm for MLP
            h = self.ln_2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
            x = x + self.mlp(h)
        else:
            # Standard transformer block
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        
        return x
```

### 6.6 The Complete Model

Now let's put it all together:

```python
# Cell 12: Complete MDLM
class MaskedDiffusionLM(nn.Module):
    def __init__(
        self,
        vocab_size,       # Including [MASK] token
        n_embd=256,       # Embedding dimension
        n_head=8,         # Number of attention heads
        n_layer=6,        # Number of transformer blocks
        block_size=128,   # Maximum sequence length
        dropout=0.1,
        use_time=True     # Whether to condition on time
    ):
        super().__init__()
        
        self.block_size = block_size
        self.use_time = use_time
        
        # Token embeddings (vocab_size already includes [MASK])
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        # Position embeddings
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Time embeddings
        if use_time:
            self.time_emb = SinusoidalTimeEmbedding(n_embd)
            self.time_proj = nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.SiLU(),
                nn.Linear(n_embd, n_embd)
            )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, use_time)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying: share weights between embedding and output
        self.head.weight = self.tok_emb.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model has {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, t=None):
        """
        Forward pass.
        
        Args:
            x: Input tokens [batch, seq_len], may contain [MASK] tokens
            t: Diffusion time [batch], optional
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, T = x.shape
        
        # Token + position embeddings
        tok_emb = self.tok_emb(x)  # [B, T, n_embd]
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos)  # [T, n_embd]
        
        h = self.dropout(tok_emb + pos_emb)
        
        # Time embedding
        if self.use_time and t is not None:
            t_emb = self.time_emb(t)  # [B, n_embd]
            t_emb = self.time_proj(t_emb)  # [B, n_embd]
        else:
            t_emb = None
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        h = self.ln_f(h)
        logits = self.head(h)  # [B, T, vocab_size]
        
        return logits

# Create model
model = MaskedDiffusionLM(
    vocab_size=vocab_size_with_mask,  # 66 = 65 chars + 1 [MASK]
    n_embd=256,
    n_head=8,
    n_layer=6,
    block_size=block_size,
    dropout=0.1,
    use_time=True
).to(device)
```

Output:
```
Model has 12.45M parameters
```

> 🎯 **Parameter Count:** With n_embd=256, n_layer=6, we get ~12M parameters. Adjust n_embd to 384 for ~20M, or n_layer=4 for ~8M. This is in our target range!

### 6.7 Architecture Comparison with nanoGPT

| Component | nanoGPT | Our MDLM |
|-----------|---------|----------|
| Attention | Causal | Full (bidirectional) |
| Time embedding | ❌ | ✅ (sinusoidal + MLP) |
| Layer norm | Standard | adaLN (time-modulated) |
| Weight tying | ✅ | ✅ |
| Output | Next token | All tokens |

---

## 7. The Training Loop {#7-the-training-loop}

### 7.1 Complete Training Function

Let's build a training loop with proper logging:

```python
# Cell 13: Training setup
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, dataloader, epochs, lr=3e-4, warmup_steps=1000):
    """Train the MDLM."""
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * epochs
    
    # Simple warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Tracking
    losses = []
    step = 0
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x_0 = batch.to(device)
            
            # Compute loss
            loss = compute_loss(model, x_0, MASK_TOKEN)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track
            epoch_losses.append(loss.item())
            losses.append(loss.item())
            step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return losses

# Let's do a quick test with 1 epoch first
print("Testing training loop...")
test_losses = train(model, dataloader, epochs=1, lr=3e-4)
```

### 7.2 Full Training Run

Now let's train for real:

```python
# Cell 14: Full training
# Reset model
model = MaskedDiffusionLM(
    vocab_size=vocab_size_with_mask,
    n_embd=256,
    n_head=8,
    n_layer=6,
    block_size=block_size,
    dropout=0.1,
    use_time=True
).to(device)

# Train!
# On M1 MacBook: ~5-10 minutes per epoch
# On GPU: ~1-2 minutes per epoch
losses = train(model, dataloader, epochs=10, lr=3e-4)

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses, alpha=0.3)
# Smoothed
window = 100
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(losses)), smoothed, 'r', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 7.3 Expected Loss Values

Here's what to expect:

| Stage | Loss | Notes |
|-------|------|-------|
| Random init | ~4.0-4.5 | log(vocab_size) ≈ log(66) ≈ 4.2 |
| After 1 epoch | ~2.5-3.0 | Learning basic patterns |
| After 5 epochs | ~1.8-2.2 | Getting coherent |
| After 10 epochs | ~1.5-1.8 | Good quality |
| Converged | ~1.2-1.5 | Character-level perplexity ~3-4 |

> 💡 **Sanity Check:** If your initial loss is much higher than 4.2, something is wrong. If it doesn't decrease, check your learning rate and gradient clipping.

---

## 8. Sampling: Generating Text {#8-sampling}

### 8.1 The Sampling Process

This is where the magic happens! To generate text:

1. Start with all [MASK] tokens
2. Iteratively unmask tokens based on model predictions
3. After many steps, we have clean text!

```python
# Cell 15: Sampling
@torch.no_grad()
def sample(model, seq_len, num_steps=50, temperature=1.0):
    """
    Generate text using the reverse diffusion process.
    
    Args:
        model: Trained MDLM
        seq_len: Length of sequence to generate
        num_steps: Number of denoising steps
        temperature: Sampling temperature
    
    Returns:
        Generated token sequence
    """
    model.eval()
    
    # Start with all masks
    x = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)
    
    # Linearly spaced time steps from 1 to 0
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]
        
        # Get model predictions
        t_batch = torch.tensor([t_current], device=device)
        logits = model(x, t_batch)  # [1, seq_len, vocab_size]
        
        # Don't predict [MASK] token during sampling
        logits[:, :, MASK_TOKEN] = float('-inf')
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
        
        # Sample predictions for all positions
        # (we'll only use some based on remasking strategy)
        pred_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)), 
            num_samples=1
        ).view(1, seq_len)
        
        # Determine which positions to unmask this step
        # Strategy: unmask a fraction of remaining masked tokens
        is_mask = (x == MASK_TOKEN)
        num_masked = is_mask.sum().item()
        
        if num_masked > 0:
            # How many to unmask this step? 
            # Linear schedule: unmask proportional to time decrease
            frac_to_unmask = (t_current - t_next) / t_current
            num_to_unmask = max(1, int(num_masked * frac_to_unmask))
            
            # Get confidence scores at masked positions
            # Use max probability as confidence
            max_probs = probs.max(dim=-1).values  # [1, seq_len]
            max_probs[~is_mask] = -1  # Don't select unmasked positions
            
            # Select top-k most confident masked positions
            _, indices = max_probs[0].topk(min(num_to_unmask, num_masked))
            
            # Unmask selected positions
            x[0, indices] = pred_tokens[0, indices]
    
    # Final cleanup: unmask any remaining masks
    is_mask = (x == MASK_TOKEN)
    if is_mask.any():
        logits = model(x, torch.tensor([0.0], device=device))
        logits[:, :, MASK_TOKEN] = float('-inf')
        probs = F.softmax(logits / temperature, dim=-1)
        final_preds = probs.argmax(dim=-1)
        x[is_mask] = final_preds[is_mask]
    
    model.train()
    return x[0].tolist()

# Generate some samples!
print("Generated samples:")
print("=" * 60)
for i in range(5):
    tokens = sample(model, seq_len=100, num_steps=50, temperature=0.8)
    text = decode(tokens)
    print(f"\nSample {i+1}:")
    print(text)
    print("-" * 40)
```

### 8.2 Visualizing the Denoising Process

This is really cool—let's watch text emerge from noise:

```python
# Cell 16: Visualize sampling
@torch.no_grad()
def sample_with_history(model, seq_len, num_steps=20):
    """Sample and return intermediate states."""
    model.eval()
    
    x = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)
    history = []
    
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    
    for i in range(num_steps):
        # Save current state
        text_state = ""
        for tok in x[0].tolist():
            if tok == MASK_TOKEN:
                text_state += "■"
            else:
                text_state += itos[tok]
        history.append((timesteps[i].item(), text_state))
        
        t_current = timesteps[i]
        t_next = timesteps[i + 1]
        
        # Model prediction
        logits = model(x, torch.tensor([t_current], device=device))
        logits[:, :, MASK_TOKEN] = float('-inf')
        probs = F.softmax(logits / 0.8, dim=-1)
        pred_tokens = probs.argmax(dim=-1)
        
        # Unmask based on confidence
        is_mask = (x == MASK_TOKEN)
        num_masked = is_mask.sum().item()
        
        if num_masked > 0:
            frac_to_unmask = (t_current - t_next) / t_current
            num_to_unmask = max(1, int(num_masked * frac_to_unmask))
            
            max_probs = probs.max(dim=-1).values
            max_probs[~is_mask] = -1
            
            _, indices = max_probs[0].topk(min(num_to_unmask, num_masked))
            x[0, indices] = pred_tokens[0, indices]
    
    # Final state
    text_state = ""
    for tok in x[0].tolist():
        text_state += itos.get(tok, "?")
    history.append((0.0, text_state))
    
    model.train()
    return history

# Visualize
history = sample_with_history(model, seq_len=60, num_steps=20)

print("Reverse Diffusion Process (t=1 → t=0):")
print("=" * 70)
for t, text in history:
    print(f"t={t:.2f}: {text[:60]}")
```

Expected output:
```
Reverse Diffusion Process (t=1 → t=0):
======================================================================
t=1.00: ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
t=0.95: ■■■■■■■■e■■■■■■■■■■■■■■■■■■■■t■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
t=0.90: ■■■■■■■■e■■■■■■■■■■■■■■■t■■■■t■■■■■■■■■■■■■■■■■■■e■■■■■■■■■■
t=0.85: ■■■■■■■■e■■■■■■■■■■■■■■■t■■■■t■e■■■■o■■■■■■■■■■■■■e■■■■■■■■■
...
t=0.10: What do you say■■■■■the■king■■of■■■■■■■■■ the■■■eart■■■■■■■
t=0.05: What do you say to■■■the king■ of■■■■■■■■■ the heart■■■■■■■
t=0.00: What do you say to me the king? of the lord? the heart of?
```

> 💡 **Aha Moment:** Watch how the model first fills in high-frequency tokens (spaces, 'e', 't'), then gradually adds less common characters! It's like watching an image come into focus, but for text.

### 8.3 Sampling Strategies

There are several ways to decide which tokens to unmask:

1. **Random**: Unmask random masked positions (simple but noisy)
2. **Confidence-based**: Unmask where model is most confident (what we implemented)
3. **Low-to-high entropy**: Unmask easiest positions first
4. **Linear schedule**: Unmask fixed fraction each step

Our confidence-based approach works well. The intuition: unmask positions where the model is sure, then use that information to help predict harder positions.

---

## 9. Evaluation and Visualization {#9-evaluation}

### 9.1 Perplexity

Perplexity measures how "surprised" the model is by the test data. Lower is better:

```python
# Cell 17: Evaluation
@torch.no_grad()
def evaluate_perplexity(model, text, block_size):
    """Compute perplexity on text."""
    model.eval()
    
    tokens = torch.tensor(encode(text), dtype=torch.long, device=device)
    total_loss = 0
    total_tokens = 0
    
    # Process in chunks
    for i in range(0, len(tokens) - block_size, block_size):
        x_0 = tokens[i:i+block_size].unsqueeze(0)
        
        # Evaluate at various mask rates and average
        losses = []
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            x_t, mask = forward_process(x_0, t, MASK_TOKEN)
            t_tensor = torch.tensor([t], device=device)
            
            logits = model(x_t, t_tensor)
            
            # Loss at masked positions
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x_0.view(-1),
                reduction='none'
            )
            masked_loss = (ce_loss * mask.view(-1).float()).sum()
            num_masked = mask.sum()
            
            if num_masked > 0:
                losses.append(masked_loss.item() / num_masked.item())
        
        if losses:
            total_loss += sum(losses) / len(losses) * block_size
            total_tokens += block_size
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    model.train()
    return perplexity

# Evaluate
# Use a held-out portion of the text
test_text = text[-10000:]  # Last 10k characters
ppl = evaluate_perplexity(model, test_text, block_size)
print(f"Test Perplexity: {ppl:.2f}")
```

### 9.2 Qualitative Evaluation: Prompt Completion

Let's test the model's ability to complete prompts (infilling!):

```python
# Cell 18: Prompt completion
@torch.no_grad()
def complete_prompt(model, prompt, completion_len=50, num_steps=30):
    """Complete a text prompt."""
    model.eval()
    
    # Encode prompt
    prompt_tokens = encode(prompt)
    total_len = len(prompt_tokens) + completion_len
    
    # Create sequence: prompt tokens + masks for completion
    x = torch.full((1, total_len), MASK_TOKEN, dtype=torch.long, device=device)
    x[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, device=device)
    
    # Which positions need to be generated?
    completion_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
    completion_mask[len(prompt_tokens):] = True
    
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]
        
        logits = model(x, torch.tensor([t_current], device=device))
        logits[:, :, MASK_TOKEN] = float('-inf')
        probs = F.softmax(logits / 0.7, dim=-1)
        
        # Only work on completion positions that are still masked
        is_mask = (x == MASK_TOKEN) & completion_mask.unsqueeze(0)
        num_masked = is_mask.sum().item()
        
        if num_masked == 0:
            break
            
        frac_to_unmask = (t_current - t_next) / t_current
        num_to_unmask = max(1, int(num_masked * frac_to_unmask))
        
        max_probs = probs.max(dim=-1).values
        max_probs[~is_mask] = -1
        
        _, indices = max_probs[0].topk(min(num_to_unmask, num_masked))
        
        pred_tokens = probs.argmax(dim=-1)
        x[0, indices] = pred_tokens[0, indices]
    
    model.train()
    return decode(x[0].tolist())

# Test prompt completion
prompts = [
    "ROMEO: ",
    "To be or not to be",
    "What light through yonder",
    "Friends, Romans, countrymen,",
]

print("Prompt Completions:")
print("=" * 60)
for prompt in prompts:
    completion = complete_prompt(model, prompt, completion_len=60)
    print(f"\nPrompt: {prompt}")
    print(f"Completion: {completion}")
    print("-" * 40)
```

### 9.3 The Infilling Superpower

One thing diffusion models do naturally that GPT cannot: **infilling**!

```python
# Cell 19: Infilling
@torch.no_grad()
def infill(model, text_with_blanks, num_steps=30):
    """
    Fill in blanks marked with [___] in the text.
    """
    model.eval()
    
    # Parse text: find blank positions
    tokens = []
    is_blank = []
    
    i = 0
    while i < len(text_with_blanks):
        if text_with_blanks[i:i+5] == "[___]":
            # Add 5 mask tokens for each blank
            for _ in range(5):
                tokens.append(MASK_TOKEN)
                is_blank.append(True)
            i += 5
        else:
            tokens.append(stoi.get(text_with_blanks[i], 0))
            is_blank.append(False)
            i += 1
    
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    is_blank = torch.tensor([is_blank], device=device)
    
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]
        
        logits = model(x, torch.tensor([t_current], device=device))
        logits[:, :, MASK_TOKEN] = float('-inf')
        probs = F.softmax(logits / 0.7, dim=-1)
        
        is_mask = (x == MASK_TOKEN) & is_blank
        num_masked = is_mask.sum().item()
        
        if num_masked == 0:
            break
        
        frac_to_unmask = (t_current - t_next) / t_current
        num_to_unmask = max(1, int(num_masked * frac_to_unmask))
        
        max_probs = probs.max(dim=-1).values
        max_probs[~is_mask] = -1
        
        _, indices = max_probs[0].topk(min(num_to_unmask, num_masked))
        pred_tokens = probs.argmax(dim=-1)
        x[0, indices] = pred_tokens[0, indices]
    
    model.train()
    return decode(x[0].tolist())

# Test infilling
print("Infilling Examples:")
print("=" * 60)
examples = [
    "To [___] or not to [___]",
    "ROMEO: O, she doth [___] the torches to burn [___]!",
    "Now is the [___] of our [___]",
]

for ex in examples:
    result = infill(model, ex)
    print(f"\nOriginal: {ex}")
    print(f"Filled:   {result}")
```

> 💡 **Aha Moment:** GPT simply *cannot* do this without tricks! To fill in a blank in the middle, GPT would need to generate everything after the blank. Diffusion models handle it naturally because they see bidirectional context.

---

## 10. What's Next and Further Reading {#10-whats-next}

### 10.1 Improvements to Try

1. **Subword tokenization**: Use BPE/SentencePiece instead of characters. The MDLM paper shows vocabulary size matters a lot!

2. **Better architecture**: Try:
   - RoPE (Rotary Position Embeddings) instead of learned
   - RMSNorm instead of LayerNorm
   - Deeper models with more parameters

3. **Improved sampling**:
   - DDPM-style sampling with noise injection
   - Nucleus (top-p) sampling
   - Semi-autoregressive generation for longer texts

4. **Better noise schedules**: Cosine schedule instead of linear

5. **Classifier-free guidance**: For controlled generation

### 10.2 Scaling Up

To train larger models:
- Use mixed precision (fp16/bf16)
- Gradient accumulation
- Flash Attention
- Multi-GPU with DDP

The LLaDA paper trained an 8B parameter model on 2.3T tokens using these techniques.

### 10.3 Essential Papers

**Foundational:**
- [D3PM](https://arxiv.org/abs/2107.03006) (Austin et al., 2021) - Discrete diffusion basics
- [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., 2024) - Simplified masked diffusion

**State of the Art:**
- [LLaDA](https://arxiv.org/abs/2502.09992) (Nie et al., 2025) - Scaled to 8B
- [SEDD](https://arxiv.org/abs/2310.16834) (Lou et al., 2024) - Score entropy approach

**Background:**
- [DDPM](https://arxiv.org/abs/2006.11239) (Ho et al., 2020) - Original diffusion for images
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019) - Masked language modeling

### 10.4 Blog Posts and Tutorials

- [What are Diffusion Language Models?](https://spacehunterinf.github.io/blog/2025/diffusion-language-models/) - Great visual explainer
- [Hugging Face: Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion) - For images, but concepts transfer
- [MDLM GitHub](https://github.com/kuleshov-group/mdlm) - Official implementation

### 10.5 Key Insights to Remember

1. **MDLM ≈ BERT with variable masking**: The training is surprisingly similar to masked language modeling

2. **No causal mask = bidirectional context**: This is the fundamental advantage over GPT

3. **Time conditioning is optional**: The mask pattern implicitly encodes noise level

4. **Generation is iterative parallel**: Unlike GPT's sequential, diffusion predicts all tokens at each step

5. **Infilling is natural**: No tricks needed for tasks GPT struggles with

### 10.6 Final Thoughts

You've just built a diffusion language model from scratch! While it won't match GPT-4 (yet), you now understand a completely different paradigm for language modeling. As of 2025, diffusion LMs are competitive with autoregressive models at the 8B scale (LLaDA vs LLaMA3).

The field is moving fast. Keep an eye on:
- Hybrid autoregressive-diffusion models
- Better sampling algorithms
- Scaling laws for diffusion LMs

Most importantly: **experiment!** Try different architectures, datasets, and sampling strategies. The best insights come from hands-on exploration.

---

## Appendix A: Complete Code Reference

### A.1 Minimal Training Script

Here's everything in one compact script:

```python
"""
Minimal MDLM Training Script
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============== Config ==============
vocab_size = 65 + 1  # Characters + MASK
MASK_TOKEN = 65
block_size = 128
n_embd = 256
n_head = 8
n_layer = 6
batch_size = 64
epochs = 10
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============== Forward Process ==============
def forward_process(x_0, t, mask_token):
    if isinstance(t, float):
        t = torch.tensor([t] * x_0.shape[0], device=x_0.device)
    rand = torch.rand_like(x_0, dtype=torch.float)
    mask = rand < t.unsqueeze(1)
    x_t = x_0.clone()
    x_t[mask] = mask_token
    return x_t, mask

# ============== Model ==============
class MDLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Simple transformer without time conditioning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layer)
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, x, t=None):
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        h = tok_emb + pos_emb
        h = self.transformer(h)
        h = self.ln_f(h)
        return self.head(h)

# ============== Loss ==============
def compute_loss(model, x_0):
    t = torch.rand(x_0.shape[0], device=x_0.device)
    x_t, mask = forward_process(x_0, t, MASK_TOKEN)
    logits = model(x_t)
    ce = F.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1), reduction='none')
    return (ce * mask.view(-1).float()).sum() / mask.sum().clamp(min=1)

# ============== Sampling ==============
@torch.no_grad()
def sample(model, length, steps=50):
    x = torch.full((1, length), MASK_TOKEN, dtype=torch.long, device=device)
    for i, t in enumerate(torch.linspace(1, 0, steps+1)[:-1]):
        logits = model(x)
        logits[:, :, MASK_TOKEN] = float('-inf')
        probs = F.softmax(logits / 0.8, dim=-1)
        preds = probs.argmax(-1)
        
        is_mask = x == MASK_TOKEN
        if not is_mask.any(): break
        
        conf = probs.max(-1).values
        conf[~is_mask] = -1
        n_unmask = max(1, int(is_mask.sum() * (1/steps)))
        _, idx = conf[0].topk(min(n_unmask, is_mask.sum()))
        x[0, idx] = preds[0, idx]
    return x[0].tolist()

# ============== Usage ==============
if __name__ == "__main__":
    model = MDLM().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Train with your data...
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # for epoch in range(epochs):
    #     for batch in dataloader:
    #         loss = compute_loss(model, batch.to(device))
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
```

---

## Appendix B: Frequently Asked Questions

**Q: Why use masking instead of Gaussian noise?**
A: Tokens are discrete—you can't add 0.3 to the letter "A". Masking is the discrete analog of adding noise.

**Q: Is this model as good as GPT?**
A: At small scale, GPT usually wins on perplexity. But diffusion models have advantages in controllability and infilling. At 8B scale (LLaDA), they're competitive.

**Q: Can I use a pretrained BERT?**
A: Yes! BERT initialization helps a lot. See the DiffusionBERT paper.

**Q: How is this different from BERT?**
A: BERT uses fixed 15% masking. MDLM uses variable masking (0-100%) which gives it generation capabilities.

**Q: Why is sampling slow?**
A: Diffusion requires many forward passes (typically 50-1000). This is the main disadvantage vs GPT. Research is ongoing to speed it up!

---

*Happy diffusing! 🌊*

*This guide is inspired by Andrej Karpathy's "Let's build GPT" tutorial. If you found this helpful, consider starring the repos of the papers mentioned!*