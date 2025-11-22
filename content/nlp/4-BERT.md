+++
title = "BERT: Bidirectional Encoder Representations from Transformers"
date = 2025-11-22T12:00:00+05:30
draft = false
weight = 4
description = "Comprehensive guide to BERT architecture covering bidirectional context understanding, masked language modeling, next sentence prediction, fine-tuning strategies, and BERT variants including DistilBERT, RoBERTa, and ALBERT."
+++

## Introduction

**BERT (Bidirectional Encoder Representations from Transformers)** is a revolutionary language model introduced by Google AI in 2018. Unlike previous models that processed text sequentially (left-to-right or right-to-left), BERT reads the entire sequence of words at once, enabling it to understand context from both directions.

**Key Achievement:** BERT achieved state-of-the-art results on 11 NLP tasks at the time of its release, demonstrating the power of bidirectional context understanding.

---

## BERT Architecture

### Overview

BERT uses **only the Transformer Encoder stack** (no decoder). This makes it ideal for understanding tasks like classification, NER, and question answering.

### Architecture Diagram

```
Input Sentence
    ↓
[Tokenization (WordPiece)]
    ↓
[Embedding Layer]
    ├─ Token Embeddings
    ├─ Position Embeddings
    └─ Segment Embeddings
    ↓
┌─────────────────────────────────────┐
│  Transformer Encoder Stack          │
│  ┌───────────────────────────────┐  │
│  │  Encoder Block × N            │  │
│  │  (N = 12 for BERT-base)       │  │
│  │  (N = 24 for BERT-large)      │  │
│  │                                │  │
│  │  Each block contains:          │  │
│  │  - Multi-Head Self-Attention   │  │
│  │  - Feed Forward Network        │  │
│  │  - Layer Normalization         │  │
│  │  - Residual Connections        │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
[Contextualized Embeddings]
    ↓
[Task-Specific Head]
    ↓
Output
```

### Model Variants

| Model | Layers | Hidden Size | Attention Heads | Parameters |
|-------|--------|-------------|-----------------|------------|
| **BERT-base** | 12 | 768 | 12 | 110M |
| **BERT-large** | 24 | 1024 | 16 | 340M |

---

## Input Representation

BERT uses a sophisticated input representation that combines three types of embeddings.

### Input Format

```
[CLS] token₁ token₂ ... tokenₙ [SEP] token₁ token₂ ... tokenₘ [SEP]
```

### Embedding Components

#### 1. Token Embeddings

- Each token is mapped to a dense vector
- Uses **WordPiece tokenization** (30,000 token vocabulary)
- Example: "playing" → ["play", "##ing"]

#### 2. Position Embeddings

- Learnable positional encodings (not sinusoidal like original Transformer)
- Indicates position of token in sequence
- Maximum sequence length: 512 tokens

#### 3. Segment Embeddings

- Used to distinguish between two sentences
- Sentence A: `E_A = 0` for all tokens
- Sentence B: `E_B = 1` for all tokens
- Single sentence tasks: all tokens use `E_A`

### Final Input Embedding

```
E = TokenEmbedding + PositionEmbedding + SegmentEmbedding
```

### Special Tokens

- **`[CLS]`**: Classification token (used for sentence-level tasks)
- **`[SEP]`**: Separator token (separates sentences)
- **`[MASK]`**: Masked token (used during pretraining)
- **`[UNK]`**: Unknown token (rare due to subword tokenization)
- **`[PAD]`**: Padding token (for batching)

### Example

```
Input: "I love NLP" and "It is fascinating"
Tokenized: ["[CLS]", "I", "love", "NL", "##P", "[SEP]", "It", "is", "fascinating", "[SEP]"]
Segment:   [  0  ,  0 ,   0  ,  0  ,   0  ,    0  ,  1  ,  1 ,      1      ,    1  ]
```

---

## Pretraining Objectives

BERT is pretrained on two tasks simultaneously, which enables it to learn rich bidirectional representations.

### 1. Masked Language Modeling (MLM)

**Goal:** Predict masked tokens using bidirectional context.

#### Process

1. **Masking Strategy:**
   - Randomly mask ~15% of input tokens
   - Of the 15%:
     - 80% → Replace with `[MASK]`
     - 10% → Replace with random token
     - 10% → Keep original token

2. **Training:**
   - Model predicts the original token at masked positions
   - Uses both left and right context (bidirectional)

#### Example

```
Input:  "The capital of France is [MASK]."
Target: "Paris"

Model sees: "The capital of France is [MASK]."
Model predicts: "Paris" (using context from both sides)
```

#### Loss Function

**Cross-entropy loss** over the vocabulary:

```
L_MLM = -Σ log P(y_masked | context)
```

Where:
- `y_masked`: Original token at masked position
- `context`: All other tokens in the sequence

#### Why This Works

- **Bidirectional context**: Unlike GPT (left-to-right), BERT sees full context
- **Random masking**: Prevents overfitting to specific masking patterns
- **Partial masking**: 10% random replacement teaches robustness

---

### 2. Next Sentence Prediction (NSP)

**Goal:** Predict whether sentence B actually follows sentence A.

#### Process

1. **Input Format:**
   ```
   [CLS] Sentence A [SEP] Sentence B [SEP]
   ```

2. **Training Data:**
   - **Positive (50%)**: Sentence B actually follows A
   - **Negative (50%)**: Sentence B is random from corpus

3. **Prediction:**
   - Use `[CLS]` token representation
   - Binary classification: "IsNext" or "NotNext"

#### Example

**Positive:**
```
Sentence A: "I love machine learning."
Sentence B: "It is a fascinating field."
Label: IsNext
```

**Negative:**
```
Sentence A: "I love machine learning."
Sentence B: "The weather is nice today."
Label: NotNext
```

#### Loss Function

**Binary cross-entropy loss:**

```
L_NSP = -[y × log(p) + (1-y) × log(1-p)]
```

Where:
- `y`: True label (1 for IsNext, 0 for NotNext)
- `p`: Predicted probability

#### Why This Matters

- Helps model understand **sentence relationships**
- Useful for tasks like:
  - Question Answering
  - Natural Language Inference
  - Summarization

---

### Combined Pretraining Loss

```
L_BERT = L_MLM + L_NSP
```

Both losses are computed and backpropagated simultaneously during pretraining.

---

## Fine-Tuning for Downstream Tasks

After pretraining, BERT can be fine-tuned for various NLP tasks by adding task-specific heads.

### Fine-Tuning Process

1. **Initialize** with pretrained BERT weights
2. **Add** task-specific output layer
3. **Train** on task-specific labeled data
4. **Update** all parameters (not just the head)

### Task-Specific Architectures

#### 1. Text Classification

```
[CLS] token₁ token₂ ... tokenₙ
    ↓
BERT Encoder
    ↓
[CLS] representation (768-dim for base)
    ↓
Dense Layer → Softmax
    ↓
Class Probabilities
```

**Example Tasks:**
- Sentiment analysis
- Spam detection
- Topic classification

#### 2. Named Entity Recognition (NER)

```
token₁ token₂ ... tokenₙ
    ↓
BERT Encoder
    ↓
Token representations (768-dim each)
    ↓
Token-wise Classification Layer
    ↓
Entity Labels (B-PER, I-PER, O, etc.)
```

**Example:**
```
Input: "John lives in New York"
Output: [B-PER, O, O, B-LOC, I-LOC]
```

#### 3. Question Answering

```
[CLS] question [SEP] context [SEP]
    ↓
BERT Encoder
    ↓
Token representations
    ↓
Start/End Position Predictors
    ↓
Answer Span (start_idx, end_idx)
```

**Example:**
```
Question: "What is the capital of France?"
Context: "Paris is the capital and largest city of France."
Answer: "Paris" (start=1, end=1)
```

#### 4. Sentence Pair Classification

```
[CLS] sentence₁ [SEP] sentence₂ [SEP]
    ↓
BERT Encoder
    ↓
[CLS] representation
    ↓
Classification Layer
    ↓
Label (entailment, contradiction, neutral)
```

**Example Tasks:**
- Natural Language Inference (NLI)
- Semantic Similarity
- Paraphrase Detection

---

## BERT Variants

Several variants of BERT have been developed to improve performance, efficiency, or address limitations.

### 1. DistilBERT

**Goal:** Reduce size and latency while keeping most of BERT's accuracy.

#### Key Features

- **40% fewer parameters** (6 transformer layers instead of 12)
- **60% faster inference**
- **Removes token-type embeddings**
- Uses **knowledge distillation**

#### Training

**Triple Loss:**
1. **Distillation Loss**: Student mimics teacher's logits
   ```
   L_distill = KL(P_teacher || P_student)
   ```

2. **Cosine Embedding Loss**: Aligns hidden states
   ```
   L_cosine = 1 - cosine(h_teacher, h_student)
   ```

3. **MLM Loss**: Standard masked language modeling
   ```
   L_MLM = -Σ log P(y_masked | context)
   ```

**Total Loss:**
```
L = α × L_distill + β × L_cosine + γ × L_MLM
```

#### Trade-offs

- ✅ Much faster inference
- ✅ Smaller model size
- ✅ Good for mobile/edge devices
- ❌ Slight drop in accuracy (~3%)

---

### 2. RoBERTa

**Goal:** Show that BERT's performance can be improved with better training practices (no architecture changes).

#### Key Changes

1. **Removes NSP objective** (found to be unnecessary)
2. **Trains on 10× more data** (160GB vs 16GB)
3. **Longer training** (more epochs)
4. **Larger batch sizes** (8K vs 256)
5. **Dynamic masking** (masking patterns change every epoch)
6. **Larger BPE vocabulary** (50K vs 30K)

#### Results

- Often **outperforms BERT** on NLP benchmarks
- Better understanding without NSP
- Demonstrates importance of training data and procedure

---

### 3. ALBERT

**Goal:** Reduce memory usage and improve scalability.

#### Key Innovations

1. **Factorized Embedding Parameterization**
   - Word embeddings: `V × E` → `V × H` → `H × E`
   - Reduces parameters from `V × H` to `V × E + E × H`
   - Where `E << H` (e.g., E=128, H=768)

2. **Cross-Layer Parameter Sharing**
   - All transformer layers share the same parameters
   - Dramatically reduces parameter count
   - Allows deeper models with same memory

3. **Sentence Order Prediction (SOP)**
   - Replaces NSP
   - Predicts if two sentences are in correct order
   - More challenging and useful than NSP

#### Results

- **Much fewer parameters** (18M vs 110M for base)
- Can train **deeper models** (24 layers vs 12)
- Similar or better performance than BERT

---

## Key Innovations

### 1. Bidirectional Context

- **Previous models**: Unidirectional (left-to-right or right-to-left)
- **BERT**: Bidirectional (sees full context)
- **Impact**: Better understanding of word meaning in context

### 2. Transfer Learning

- **Pretrain once** on large unlabeled corpus
- **Fine-tune** on specific tasks with small labeled datasets
- **Impact**: State-of-the-art results with less labeled data

### 3. Contextual Embeddings

- **Previous embeddings**: Same word always has same embedding
- **BERT**: Embedding depends on context
- **Example**: "bank" (financial) vs "bank" (river) have different embeddings

### 4. Task-Agnostic Architecture

- Same architecture works for multiple tasks
- Only output layer changes
- **Impact**: One model for many applications

---

## Limitations

### 1. Maximum Sequence Length

- Limited to **512 tokens** (can be extended but expensive)
- Cannot handle very long documents in single pass
- **Solutions**: Sliding window, hierarchical models

### 2. Autoregressive Generation

- Not designed for text generation
- Cannot generate text like GPT
- **Solution**: Use GPT or encoder-decoder models (T5, BART)

### 3. Computational Cost

- Large models require significant compute
- Fine-tuning can be expensive
- **Solutions**: DistilBERT, quantization, pruning

### 4. Pretraining Data Bias

- Reflects biases in training data
- Can produce biased outputs
- **Solution**: Careful data curation, debiasing techniques

### 5. Static Embeddings

- Embeddings are fixed after training
- Cannot update knowledge without retraining
- **Solution**: Continual learning, retrieval-augmented models

---

## Summary

BERT revolutionized NLP by:

1. **Bidirectional understanding** - sees full context
2. **Transfer learning** - pretrain once, fine-tune for many tasks
3. **Contextual embeddings** - word meaning depends on context
4. **Task-agnostic architecture** - same model for multiple tasks

**Key Components:**
- Transformer encoder stack
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- WordPiece tokenization

**Variants:**
- **DistilBERT**: Faster, smaller
- **RoBERTa**: Better training, stronger performance
- **ALBERT**: Parameter-efficient, deeper models

**Best For:**
- Text classification
- Named Entity Recognition
- Question Answering
- Natural Language Inference

**Not Ideal For:**
- Text generation (use GPT instead)
- Very long documents (use hierarchical models)
- Real-time applications (use DistilBERT)

For related topics:
- [Transformer Architecture](./8.%20Transformer.md) - Foundation of BERT
- [Tokenization](./2.%20Tokenization.md) - WordPiece tokenization used by BERT
- [GPT](./10.%20GPT.md) - Decoder-only alternative for generation
