+++
title = "GPT: Generative Pre-trained Transformer"
date = 2025-11-22T12:00:00+05:30
draft = false
weight = 5
description = "Comprehensive guide to GPT architecture covering decoder-only transformers, autoregressive generation, GPT variants (GPT-1 through GPT-4), ChatGPT, RLHF, context length, and applications of generative language models."
+++

## Introduction

**GPT (Generative Pre-trained Transformer)** is a family of autoregressive language models developed by OpenAI. Unlike BERT, which uses bidirectional context, GPT processes text **unidirectionally (left-to-right)**, making it ideal for text generation tasks.

**Key Innovation:** GPT demonstrated that large-scale pretraining on diverse text corpora, followed by task-specific fine-tuning, could achieve state-of-the-art results across many NLP tasks.

---

## GPT Architecture

### Overview

GPT uses **only the Transformer Decoder stack** (no encoder). It is:
- **Unidirectional**: Processes text left-to-right
- **Causal**: Cannot see future tokens (no information leakage)
- **Autoregressive**: Generates text token by token

### Architecture Diagram

```
Input Tokens
    ↓
[Token Embeddings + Position Embeddings]
    ↓
┌─────────────────────────────────────┐
│  Transformer Decoder Stack           │
│  ┌───────────────────────────────┐  │
│  │  Decoder Block × N            │  │
│  │  (N = 12 for GPT-1)           │  │
│  │  (N = 24 for GPT-2)           │  │
│  │  (N = 96 for GPT-3)           │  │
│  │                                │  │
│  │  Each block contains:          │  │
│  │  - Masked Multi-Head           │  │
│  │    Self-Attention              │  │
│  │  - Feed Forward Network        │  │
│  │  - Layer Normalization         │  │
│  │  - Residual Connections        │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
[Final Hidden States]
    ↓
[Linear Layer (tied with embeddings)]
    ↓
[Softmax]
    ↓
[Next Token Probabilities]
```

### Key Characteristics

- **No Encoder**: Unlike original Transformer, GPT has no encoder
- **Masked Attention**: Prevents looking at future tokens
- **Autoregressive**: Predicts next token given previous tokens
- **Weight Tying**: Output embedding matrix tied with input embedding matrix

---

## Input Representation

### Token Embeddings

- Each token is mapped to a dense vector using an embedding matrix
- Vocabulary size varies by model:
  - GPT-1: 40,000 tokens
  - GPT-2: 50,257 tokens (BPE)
  - GPT-3: 50,257 tokens (BPE)
  - GPT-4: Uses byte-level BPE

### Positional Embeddings

- **Learnable positional embeddings** (not sinusoidal)
- Added to token embeddings to encode position information
- Maximum sequence length varies:
  - GPT-1: 512 tokens
  - GPT-2: 1024 tokens
  - GPT-3: 2048 tokens
  - GPT-4: Up to 128K tokens (depending on variant)

### Final Input

```
x = TokenEmbedding(token) + PositionalEmbedding(position)
```

### Tokenization

- **GPT-1, GPT-2**: Byte-Pair Encoding (BPE)
- **GPT-3, GPT-4**: Byte-level BPE
- Handles out-of-vocabulary words through subword units

---

## Decoder Block Details

### GPT Decoder Block Architecture

```
Input Embeddings + Positional Encodings
            ↓
  ┌─────────────────────────────────────┐
  │  Masked Multi-Head Self-Attention   │
  │  (Causal mask - only previous tokens)│
  └─────────────────────────────────────┘
            ↓
      [Add & LayerNorm]
            ↓
  ┌─────────────────────────────────────┐
  │  Feed Forward Network (FFN)         │
  │  FFN(x) = GELU(xW₁ + b₁)W₂ + b₂      │
  └─────────────────────────────────────┘
            ↓
      [Add & LayerNorm]
            ↓
        Output → Next Decoder Layer
```

### Components

#### 1. Masked Multi-Head Self-Attention

**Purpose:** Allows each token to attend only to previous tokens (causal attention).

**Causal Mask:**
```
For sequence: ["I", "love", "NLP"]

Attention matrix (masked):
[1, 0, 0]  ← "I" can only see itself
[1, 1, 0]  ← "love" can see "I" and itself
[1, 1, 1]  ← "NLP" can see all previous tokens
```

**Formula:**
```
MaskedAttention(Q, K, V) = softmax(QK^T / √d_k + M) × V
```

Where `M` is the causal mask (set to -∞ for future positions).

**Key Difference from BERT:**
- **BERT**: Can see all tokens (bidirectional)
- **GPT**: Can only see previous tokens (unidirectional)

#### 2. Feed Forward Network (FFN)

Applied independently to each position:

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

**Differences from original Transformer:**
- Uses **GELU activation** instead of ReLU
- Typical dimensions: 768 → 3072 → 768 (for GPT-2 base)

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) = x × Φ(x)
```
Where Φ(x) is the CDF of standard normal distribution.

#### 3. Residual Connections & Layer Normalization

- **Residual Connection**: `x + Sublayer(x)`
- **Layer Normalization**: Applied after residual connection
- **Pre-norm**: LayerNorm applied before sublayer (in GPT-2 and later)

---

## Training Objective

### Autoregressive Language Modeling

GPT is trained to predict the **next token** in a sequence given all previous tokens.

### Loss Function

**Categorical Cross-Entropy Loss:**

```
L = -Σ log P(x_t | x_{<t})
```

Where:
- `x_t`: Token at position t
- `x_{<t}`: All previous tokens (x₁, x₂, ..., x_{t-1})

**Full Sequence Loss:**

```
L = -Σ_{t=1}^T log P(x_t | x_1, ..., x_{t-1})
```

### Training Process

1. **Input**: Sequence of tokens `[x₁, x₂, ..., x_T]`
2. **Forward Pass**: 
   - Process tokens through decoder stack
   - Get hidden states for each position
3. **Prediction**: 
   - Predict next token at each position
   - Position t predicts token at position t+1
4. **Loss**: 
   - Compare predictions with actual next tokens
   - Average cross-entropy loss across all positions

### Example

```
Input:  "I love machine"
Target: "love machine learning"

At position 1 ("I"):     Predict "love"     ✓
At position 2 ("love"): Predict "machine"  ✓
At position 3 ("machine"): Predict "learning" ✓
```

---

## Autoregressive Generation

### Generation Process

GPT generates text **autoregressively** - one token at a time.

#### Step-by-Step

1. **Start**: Given initial prompt (or empty)
2. **Predict**: Model outputs probability distribution over vocabulary
3. **Sample**: Select next token (using sampling strategy)
4. **Append**: Add token to sequence
5. **Repeat**: Use new sequence to predict next token
6. **Stop**: When end-of-sequence token is generated or max length reached

### Sampling Strategies

#### 1. Greedy Decoding

- Always selects token with highest probability
- Fast but can be repetitive
- Formula: `next_token = argmax(P(token | context))`

#### 2. Top-k Sampling

- Samples from top k most likely tokens
- Introduces randomness while maintaining quality
- Typical k: 40-50

#### 3. Top-p (Nucleus) Sampling

- Samples from tokens whose cumulative probability exceeds p
- Dynamic vocabulary size
- Typical p: 0.9-0.95

#### 4. Temperature Sampling

- Adjusts probability distribution sharpness
- Temperature < 1: More focused (deterministic)
- Temperature > 1: More diverse (random)
- Formula: `P'(token) = P(token)^(1/T) / Σ P(token)^(1/T)`

### Generation Example

```
Prompt: "The future of AI is"
    ↓
Step 1: Predict next token → "bright" (P=0.3), "uncertain" (P=0.25), ...
       Sample: "bright"
    ↓
Sequence: "The future of AI is bright"
    ↓
Step 2: Predict next token → "and" (P=0.4), "with" (P=0.2), ...
       Sample: "and"
    ↓
Sequence: "The future of AI is bright and"
    ↓
... (continues until stop token or max length)
```

---

## GPT Variants

### GPT-1 (2018)

**Specifications:**
- **Parameters**: 117M
- **Layers**: 12
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Context Length**: 512 tokens
- **Training Data**: BooksCorpus (7,000 books)

**Key Contribution:**
- Demonstrated effectiveness of transfer learning
- Fine-tuned for various tasks with task-specific heads

---

### GPT-2 (2019)

**Specifications:**
- **Parameters**: 1.5B (largest variant)
- **Layers**: 48 (largest variant)
- **Hidden Size**: 1600 (largest variant)
- **Attention Heads**: 25 (largest variant)
- **Context Length**: 1024 tokens
- **Training Data**: WebText (40GB, 8M web pages)

**Key Innovations:**
- **Zero-shot learning**: No fine-tuning needed for many tasks
- **Larger scale**: 10× more parameters than GPT-1
- **Better tokenization**: Improved BPE with byte-level encoding

**Variants:**
- Small: 117M parameters
- Medium: 345M parameters
- Large: 762M parameters
- XL: 1.5B parameters

---

### GPT-3 (2020)

**Specifications:**
- **Parameters**: 175B
- **Layers**: 96
- **Hidden Size**: 12,288
- **Attention Heads**: 96
- **Context Length**: 2048 tokens (can be extended)
- **Training Data**: Common Crawl, WebText2, Books, Wikipedia (570GB)

**Key Innovations:**
- **In-context learning**: Few-shot learning without fine-tuning
- **Massive scale**: 100× more parameters than GPT-2
- **Emergent abilities**: Shows capabilities not explicitly trained

**Capabilities:**
- Few-shot learning
- Code generation
- Translation
- Question answering
- Creative writing

---

### GPT-4 (2023)

**Specifications:**
- **Parameters**: Not disclosed (estimated 1T+)
- **Architecture**: Details not fully disclosed
- **Context Length**: Up to 128K tokens (depending on variant)
- **Multimodal**: Can process both text and images (GPT-4V)

**Key Innovations:**
- **Improved reasoning**: Better at complex tasks
- **Longer context**: Can handle much longer documents
- **Multimodal capabilities**: Processes images and text
- **Better alignment**: More helpful, harmless, and honest

**Variants:**
- GPT-4: Standard version
- GPT-4 Turbo: Faster, cheaper
- GPT-4o: Optimized version

---

## ChatGPT and RLHF

### What Makes ChatGPT Different?

ChatGPT is GPT fine-tuned with **Reinforcement Learning from Human Feedback (RLHF)** to make it more helpful, harmless, and aligned with human preferences.

### Training Pipeline

#### Stage 1: Supervised Fine-Tuning (SFT)

1. **Human Labelers** create high-quality input-output pairs
2. **Examples**: Conversations, instructions, Q&A pairs
3. **Training**: Fine-tune GPT on these pairs
4. **Result**: Model learns to follow instructions and be helpful

**Example:**
```
Input: "Explain quantum computing"
Output: "Quantum computing is a type of computation that uses quantum mechanical phenomena..."
```

#### Stage 2: Reward Modeling

1. **Generate Multiple Outputs**: For each prompt, generate several responses
2. **Human Ranking**: Labelers rank outputs by quality
3. **Train Reward Model**: Learn to predict human preferences
4. **Result**: Reward model that scores response quality

**Example:**
```
Prompt: "Write a haiku about AI"
Response A: [Rank 1] "Silicon dreams rise, Neural pathways light the way, Future in code's eyes"
Response B: [Rank 2] "AI is smart. It learns things. It helps people."
Response C: [Rank 3] "I don't know what a haiku is."
```

#### Stage 3: Reinforcement Learning (PPO)

1. **Policy**: Current GPT model
2. **Reward Model**: Trained in Stage 2
3. **Optimization**: Use Proximal Policy Optimization (PPO) to maximize reward
4. **Result**: Model generates responses preferred by humans

**PPO Objective:**
```
L_PPO = E[min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)]
```

Where:
- `r(θ)`: Probability ratio between new and old policy
- `A`: Advantage estimate
- `ε`: Clipping parameter (typically 0.2)

### Benefits of RLHF

- **More helpful**: Better at following instructions
- **More aligned**: Responses match human preferences
- **Safer**: Less likely to generate harmful content
- **More conversational**: Better at dialogue

---

## Context Length

### What is Context Length?

**Context length** (or context window) refers to the **maximum number of tokens** a model can process in a single forward pass.

### Context Length by Model

| Model | Context Length | Notes |
|-------|---------------|-------|
| **GPT-1** | 512 tokens | Original limit |
| **GPT-2** | 1024 tokens | Doubled from GPT-1 |
| **GPT-3** | 2048 tokens | Standard version |
| **GPT-3.5** | 4096 tokens | Extended context |
| **GPT-4** | 8192 tokens | Standard version |
| **GPT-4 Turbo** | 128K tokens | Very long context |
| **BERT** | 512 tokens | Typical limit |

### Implications

- **Longer context**: Can process longer documents, maintain more conversation history
- **Computational cost**: Quadratic complexity with sequence length (O(n²))
- **Memory**: Requires more memory for longer contexts

### Handling Long Documents

If input exceeds context length:
1. **Truncation**: Cut off beginning or end
2. **Sliding Window**: Process in chunks
3. **Hierarchical**: Summarize chunks, then process summaries

---

## Key Differences from BERT

| Aspect | GPT | BERT |
|--------|-----|------|
| **Architecture** | Decoder-only | Encoder-only |
| **Direction** | Unidirectional (left-to-right) | Bidirectional |
| **Training** | Autoregressive language modeling | Masked language modeling + NSP |
| **Context** | Only sees previous tokens | Sees all tokens |
| **Best For** | Text generation, completion | Classification, NER, QA |
| **Generation** | Can generate text | Cannot generate (without modifications) |
| **Understanding** | Good for generation tasks | Better for understanding tasks |

### When to Use Which?

**Use GPT for:**
- Text generation
- Language completion
- Creative writing
- Code generation
- Dialogue systems

**Use BERT for:**
- Text classification
- Named Entity Recognition
- Question Answering
- Sentiment analysis
- Tasks requiring bidirectional context

---

## Applications

### 1. Text Generation

- Creative writing
- Story generation
- Poetry
- Article writing

### 2. Code Generation

- Code completion
- Function generation
- Bug fixing
- Code explanation

### 3. Conversational AI

- Chatbots
- Virtual assistants
- Customer service
- Interactive dialogue

### 4. Translation

- Language translation
- Style transfer
- Paraphrasing

### 5. Question Answering

- Open-domain QA
- Reading comprehension
- Information extraction

### 6. Summarization

- Document summarization
- Meeting notes
- Article summaries

### 7. Few-Shot Learning

- Learning from examples
- Task adaptation
- In-context learning

---

## Limitations

### 1. Hallucination

- **Problem**: Generates plausible but incorrect information
- **Cause**: Trained to be fluent, not necessarily factual
- **Solution**: Fact-checking, retrieval-augmented generation

### 2. Context Window Limits

- **Problem**: Limited context length (even 128K may not be enough)
- **Cause**: Quadratic complexity of attention
- **Solution**: Hierarchical processing, retrieval systems

### 3. Computational Cost

- **Problem**: Very expensive to train and run
- **Cause**: Large number of parameters
- **Solution**: Model compression, quantization, smaller models

### 4. Bias and Safety

- **Problem**: Can generate biased or harmful content
- **Cause**: Reflects biases in training data
- **Solution**: RLHF, content filtering, careful training

### 5. Lack of Real-Time Learning

- **Problem**: Knowledge cutoff date, can't learn new information
- **Cause**: Static model after training
- **Solution**: Retrieval-augmented generation, fine-tuning

### 6. Repetition

- **Problem**: Can get stuck in repetitive loops
- **Cause**: Autoregressive nature, sampling issues
- **Solution**: Better sampling strategies, repetition penalties

### 7. Inconsistency

- **Problem**: May give different answers to same question
- **Cause**: Stochastic generation
- **Solution**: Temperature control, deterministic settings

---

## Summary

GPT revolutionized NLP by demonstrating:

1. **Scalability**: Larger models → better performance
2. **Transfer Learning**: Pretrain once, use for many tasks
3. **In-Context Learning**: Few-shot learning without fine-tuning
4. **Text Generation**: High-quality autoregressive generation

**Key Components:**
- Transformer decoder stack
- Masked (causal) self-attention
- Autoregressive language modeling
- Byte-level BPE tokenization

**Evolution:**
- **GPT-1**: Proof of concept (117M)
- **GPT-2**: Zero-shot learning (1.5B)
- **GPT-3**: In-context learning (175B)
- **GPT-4**: Multimodal, improved reasoning (1T+)

**Best For:**
- Text generation
- Code generation
- Conversational AI
- Creative tasks

**Not Ideal For:**
- Tasks requiring bidirectional context (use BERT)
- Classification tasks (BERT often better)
- Real-time applications (use smaller models)

For related topics:
- [Transformer Architecture](./8.%20Transformer.md) - Foundation of GPT
- [BERT](./9.%20BERT.md) - Encoder-only alternative
- [Tokenization](./2.%20Tokenization.md) - BPE tokenization used by GPT
