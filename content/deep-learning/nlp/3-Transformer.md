+++
title = "Transformer Architecture"
date = 2025-11-22T12:00:00+05:30
draft = false
weight = 3
description = "Comprehensive guide to the Transformer architecture introduced in 'Attention is All You Need'. Covers encoder-decoder architecture, self-attention mechanisms, multi-head attention, positional encoding, and training processes."
+++

## Introduction

The Transformer architecture, introduced in the paper **"Attention is All You Need"** (Vaswani et al., 2017), revolutionized NLP by eliminating the need for recurrence or convolutions in sequence modeling. It relies entirely on **self-attention mechanisms**.

The Transformer (referred to as "vanilla Transformer" to distinguish it from enhanced versions) has an encoder-decoder architecture, commonly used in many Neural Machine Translation (NMT) models. Later, simplified versions achieved great performance in language modeling tasks:
- **Encoder-only**: BERT
- **Decoder-only**: GPT

**Reference:** [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)

---

## High-Level Overview

### Key Ideas

- **Parallel Processing**: Processes entire sequences at once (unlike RNNs)
- **Self-Attention**: Each token attends to all other tokens in the sequence
- **No Recurrence**: No sequential dependencies, enabling parallelization
- **Position Encoding**: Explicit positional information since there's no inherent order

### Architecture Flow

```
Input Sequence
    ↓
[Embedding + Positional Encoding]
    ↓
Encoder Stack (× N layers)
    ↓
[Contextual Representations]
    ↓
Decoder Stack (× N layers) ← [Attends to Encoder Output]
    ↓
[Linear + Softmax]
    ↓
Output Sequence
```

---

## Encoder-Decoder Architecture

The Transformer uses a **stacked encoder-decoder** architecture, where both encoder and decoder consist of multiple identical layers.

### Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        ENCODER                                │
│                                                               │
│  Input → [Embedding + Positional Encoding]                   │
│    ↓                                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Encoder Block × N (N=6 in original paper)          │    │
│  │  ┌───────────────────────────────────────────────┐  │    │
│  │  │  Multi-Head Self-Attention                    │  │    │
│  │  │  Add & LayerNorm                              │  │    │
│  │  │  Feed Forward Network                         │  │    │
│  │  │  Add & LayerNorm                              │  │    │
│  │  └───────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│    ↓                                                          │
│  Encoder Output (Contextual Embeddings)                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                        DECODER                                │
│                                                               │
│  Output → [Embedding + Positional Encoding]                  │
│    ↓                                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Decoder Block × N (N=6 in original paper)          │    │
│  │  ┌───────────────────────────────────────────────┐  │    │
│  │  │  Masked Multi-Head Self-Attention             │  │    │
│  │  │  Add & LayerNorm                              │  │    │
│  │  │  Multi-Head Cross-Attention (Encoder-Decoder) │  │    │
│  │  │  Add & LayerNorm                              │  │    │
│  │  │  Feed Forward Network                         │  │    │
│  │  │  Add & LayerNorm                              │  │    │
│  │  └───────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│    ↓                                                          │
│  [Linear + Softmax] → Output Probabilities                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Encoder Block

The encoder generates an **attention-based representation** with the capability to locate specific pieces of information from a potentially infinitely-large context.

### Encoder Specifications

- **Stack of N=6 identical layers** (in original paper)
- Each layer has:
  1. Multi-head self-attention layer
  2. Position-wise fully connected feed-forward network
- Each sub-layer has:
  - Residual connection
  - Layer normalization
- All sub-layers output data of the same dimension (typically 512)

### Encoder Block Architecture

```
Input Embeddings + Positional Encodings
            ↓
  ┌─────────────────────────────────────┐
  │  Multi-Head Self-Attention          │
  │  (Q, K, V all from same sequence)   │
  └─────────────────────────────────────┘
            ↓
      [Add & LayerNorm]
            ↓
  ┌─────────────────────────────────────┐
  │  Feed Forward Network (FFN)          │
  │  FFN(x) = max(0, xW₁ + b₁)W₂ + b₂    │
  └─────────────────────────────────────┘
            ↓
      [Add & LayerNorm]
            ↓
        Output → Next Encoder Layer
```

### Components Explained

#### 1. Multi-Head Self-Attention

- Computes attention over **all positions** in the sequence
- Uses **multiple attention heads** to capture different representations
- Each head learns different aspects of relationships between tokens

#### 2. Feed Forward Network (FFN)

Applied to each position **independently** (position-wise):

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

- Two linear transformations with ReLU activation
- Typically: input dimension → 4× dimension → input dimension
- Example: 512 → 2048 → 512

#### 3. Residual Connections & Layer Normalization

- **Residual Connection**: `x + Sublayer(x)`
- **Layer Normalization**: Applied after residual connection
- Formula: `LayerNorm(x + SelfAttention(x))`

**Why Residual Connections?**
- Helps with gradient flow during training
- Allows model to learn identity mappings when needed

---

## Decoder Block

The decoder is able to **retrieve information from the encoded representation** and generate the output sequence.

### Decoder Specifications

- **Stack of N=6 identical layers** (in original paper)
- Each layer has:
  1. Masked multi-head self-attention
  2. Multi-head cross-attention (encoder-decoder attention)
  3. Position-wise fully connected feed-forward network
- Each sub-layer has:
  - Residual connection
  - Layer normalization
- **Masked attention** prevents positions from attending to subsequent positions

### Decoder Block Architecture

```
Decoder Input Embeddings + Positional Encodings
            ↓
  ┌─────────────────────────────────────────────┐
  │  Masked Multi-Head Self-Attention          │
  │  (Can only attend to previous tokens)       │
  └─────────────────────────────────────────────┘
            ↓
      [Add & LayerNorm]
            ↓
  ┌─────────────────────────────────────────────┐
  │  Multi-Head Cross-Attention                 │
  │  Q: Decoder, K & V: Encoder Output          │
  └─────────────────────────────────────────────┘
            ↓
      [Add & LayerNorm]
            ↓
  ┌─────────────────────────────────────────────┐
  │  Feed Forward Network (FFN)                 │
  └─────────────────────────────────────────────┘
            ↓
      [Add & LayerNorm]
            ↓
        Output → Next Decoder Layer or Softmax
```

### Components Explained

#### 1. Masked Multi-Head Self-Attention

**Purpose:** Prevents the decoder from peeking ahead at future tokens (ensures causality).

**How it works:**
- Computes self-attention like the encoder
- Applies a **causal mask** to prevent "seeing ahead"
- Only tokens at positions ≤ t are visible to token at position t

**Mask Example:**
```
For sequence of length 4:
[1, 0, 0, 0]  ← token 1 can only see itself
[1, 1, 0, 0]  ← token 2 can see tokens 1-2
[1, 1, 1, 0]  ← token 3 can see tokens 1-3
[1, 1, 1, 1]  ← token 4 can see all tokens 1-4
```

**Formula:**
```
MaskedAttention(Q, K, V) = softmax(QK^T / √d_k + M) × V
```
Where M is the mask (set to -∞ for masked positions).

#### 2. Multi-Head Cross-Attention (Encoder-Decoder Attention)

**Purpose:** Allows decoder to attend to encoder output, learning alignment between input and output.

**How it works:**
- **Queries (Q)**: Come from decoder's previous layer
- **Keys (K) and Values (V)**: Come from encoder output
- This is where encoder outputs are used as input to the decoder

**Formula:**
```
CrossAttention(Q_decoder, K_encoder, V_encoder) = softmax(QK^T / √d_k) × V
```

#### 3. Feed Forward Network & Layer Normalization

Same as in encoder block - applied position-wise with residual connections.

---

## Attention Mechanisms

### Self-Attention vs. Cross-Attention

| Type | Queries (Q) | Keys & Values (K, V) | Purpose |
|------|------------|---------------------|---------|
| **Self-Attention** (Encoder or Decoder) | Same sequence | Same sequence | Learn internal relationships within sequence |
| **Cross-Attention** (Decoder only) | Decoder | Encoder outputs | Bring in external context from input sequence |

### Attention Formula

The core attention mechanism is:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Components:**
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide?"
- **√d_k**: Scaling factor to prevent softmax saturation

**Steps:**
1. Compute similarity scores: `QK^T`
2. Scale by `√d_k`
3. Apply softmax to get attention weights
4. Weighted sum of values: `Attention × V`

---

## Multi-Head Attention

Instead of one attention mechanism, use **multiple attention heads in parallel**.

### Architecture

```
Input
  ↓
[Linear Projections] → Q₁, K₁, V₁, Q₂, K₂, V₂, ..., Qₕ, Kₕ, Vₕ
  ↓
[Head 1: Attention(Q₁, K₁, V₁)]
[Head 2: Attention(Q₂, K₂, V₂)]
...
[Head h: Attention(Qₕ, Kₕ, Vₕ)]
  ↓
[Concatenate] → [Linear Projection] → Output
```

### Why Multiple Heads?

- Each head can learn to attend to **different aspects** of relationships
- Head 1 might focus on syntax, Head 2 on semantics, etc.
- Allows model to capture **diverse patterns** simultaneously

### Formula

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

**Typical Configuration:**
- Number of heads (h): 8
- Dimension per head (d_k): 64
- Total dimension: h × d_k = 512

---

## Positional Encoding

Since Transformers have no recurrence or convolution, they need **explicit positional information**.

### Methods

1. **Sinusoidal Positional Encoding** (Original Transformer)
   - Fixed, non-learnable encodings
   - Uses sine and cosine functions of different frequencies
   - Formula:
     ```
     PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
     ```

2. **Learned Positional Embeddings** (BERT, GPT)
   - Learnable parameters
   - Similar to token embeddings
   - Often performs better in practice

### Usage

```
Final Input = Token Embedding + Positional Encoding
```

---

## Feed Forward Network

The FFN is a **2-layer MLP** applied independently to each position.

### Architecture

```
Input (d_model = 512)
  ↓
Linear(d_model → d_ff = 2048) + ReLU
  ↓
Linear(d_ff → d_model)
  ↓
Output (d_model = 512)
```

### Formula

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Typical Dimensions:**
- Input/Output: 512
- Hidden: 2048 (4× expansion)

**Why this design?**
- Provides non-linearity
- Allows model to transform representations
- Applied position-wise (no interaction between positions)

---

## Training Process

### Training Steps

#### 1. Inputs

- **Encoder**: Gets full source sentence → outputs contextual embeddings
- **Decoder**: Gets partial target sentence → tries to predict next token

#### 2. Forward Pass

1. Encoder creates contextual token representations
2. Decoder uses masked self-attention (prevents peeking at future tokens)
3. Decoder uses cross-attention to attend to encoder outputs
4. Final output: probability distribution over vocabulary via Softmax

#### 3. Loss Computation

**Categorical Cross-Entropy Loss:**

```
L = -Σ log(P(y_t | y_{<t}, x))
```

Where:
- `y_t`: Target token at position t
- `y_{<t}`: All previous tokens
- `x`: Input sequence

**Training Objective:**
- Maximize likelihood of correct next token
- Minimize negative log-likelihood

### Teacher Forcing

During training, decoder receives **ground truth tokens** as input (not its own predictions). This is called "teacher forcing" and speeds up training.

---

## Key Innovations

### 1. Parallelization

- **RNNs**: Sequential processing (slow)
- **Transformers**: Parallel processing of entire sequence (fast)

### 2. Long-Range Dependencies

- **RNNs**: Struggle with long sequences (vanishing gradients)
- **Transformers**: Direct attention to any position (constant path length)

### 3. Interpretability

- Attention weights show which tokens the model focuses on
- Can visualize attention patterns

### 4. Scalability

- Can stack many layers
- Can process longer sequences (with modifications)
- Foundation for large language models (GPT, BERT, T5)

---

## Summary

The Transformer architecture revolutionized NLP by:

1. **Eliminating recurrence** - enables parallel processing
2. **Using self-attention** - captures long-range dependencies
3. **Stacking encoder-decoder blocks** - learns hierarchical representations
4. **Enabling transfer learning** - foundation for pretrained models

**Key Components:**
- Multi-head self-attention
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Positional encodings

**Variants:**
- **Encoder-only**: BERT (classification, NER)
- **Decoder-only**: GPT (generation)
- **Encoder-decoder**: T5, BART (translation, summarization)

For more details on specific implementations:
- [BERT](./9.%20BERT.md) - Encoder-only Transformer
- [GPT](./10.%20GPT.md) - Decoder-only Transformer
