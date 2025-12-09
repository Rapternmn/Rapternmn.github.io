+++
title = "Tokenization in NLP"
date = 2025-11-22T12:00:00+05:30
draft = false
weight = 2
description = "Comprehensive guide to tokenization in NLP covering word-level, subword, and byte-level tokenization techniques. Includes detailed explanations of BPE, WordPiece, SentencePiece algorithms and their applications."
+++

## What is Tokenization?

Tokenization is the process of **splitting text into smaller meaningful units (tokens)**, which can be words, subwords, characters, or even sentences depending on the application.

**Example:**  
`"I love NLP!"` → tokens: `["I", "love", "NLP", "!"]`

---

## Why is Tokenization Important?

- **Machines don't understand raw text** - they need text broken into structured units
- **Tokens help convert text into numeric representations** (embeddings, one-hot, etc.)
- **Good tokenization helps models generalize better** and handle unseen words
- **Vocabulary size management** - controls the size of the model's vocabulary
- **Handling rare words** - subword tokenization can handle words not seen during training

---

## Tokenization Techniques

### 1. Word-level Tokenization

Splits text into words separated by spaces or punctuation.

**Example:**  
`"Natural Language Processing"` → `["Natural", "Language", "Processing"]`

**Pros:**
- ✅ Simple and intuitive
- ✅ Preserves semantic meaning
- ✅ Easy to implement

**Cons:**
- ❌ **Out-of-Vocabulary (OOV) problem** - can't handle unseen words
- ❌ **Large vocabulary size** - requires storing all unique words
- ❌ **Language-specific** - different rules for different languages
- ❌ **Handles rare words poorly**

**Usage:** Rare in modern LLMs (mostly early NLP pipelines, NLTK, spaCy). This limitation led to the development of subword tokenization methods.

---

### 2. Subword Tokenization (Most Widely Used Today)

Splits words into smaller units (subwords), balancing word-level and character-level approaches. This is the **standard approach** for modern NLP models.

**Example:**  
`"unhappiness"` → `["un", "happiness"]` or `["un", "happy", "ness"]`

**Pros:**
- ✅ **Efficient vocabulary** - compact vocabulary size
- ✅ **Handles rare words** - can break down unknown words
- ✅ **Good balance** - between word and character level
- ✅ **Works across languages** - especially with multilingual models

**Cons:**
- ❌ **Tokens sometimes don't align with linguistic meaning**
- ❌ **More complex** - requires training tokenization algorithms

**Popular subword methods:**
- **Byte-Pair Encoding (BPE)** → Used in GPT-2, RoBERTa, LLaMA, Mistral
- **WordPiece** → Used in BERT, DistilBERT, ALBERT
- **SentencePiece** → Used in T5, XLNet, multilingual models

---

### 3. Byte-level Tokenization

Breaks text into raw bytes rather than characters. Often combined with BPE (byte-level BPE).

**Example:**  
`"ChatGPT"` → byte tokens like `[67, 104, 97, 116, 71, 80, 84]`

**Pros:**
- ✅ **No OOV issue** - any text can be represented
- ✅ **Language-agnostic** - works for any language/script
- ✅ **Unified representation** - handles emojis, special characters naturally

**Usage:** GPT-3, GPT-4 (byte-level BPE), Falcon

---

## Subword Tokenization Algorithms

### Byte-Pair Encoding (BPE)

**Algorithm:**
1. Start with character-level vocabulary
2. Count all pairs of consecutive symbols
3. Replace most frequent pair with a new symbol
4. Repeat until desired vocabulary size

**Example:**
```
Initial: "low" = ['l', 'o', 'w'], "lower" = ['l', 'o', 'w', 'e', 'r']
Step 1: Most frequent pair is 'l' + 'o' → merge to 'lo'
Step 2: Most frequent pair is 'lo' + 'w' → merge to 'low'
Result: "low" = ['low'], "lower" = ['low', 'e', 'r']
```

**Characteristics:**
- Greedy, frequency-based algorithm
- Deterministic (same corpus → same vocabulary)
- Used by: GPT-2, RoBERTa, XLM-R, LLaMA, Mistral

---

### WordPiece

**Algorithm:**
1. Start with word-level vocabulary
2. Train a language model to score subword units
3. Iteratively merge pairs that maximize likelihood
4. Uses `##` prefix for continuation tokens

**Example:**
```
"playing" → ["play", "##ing"]
"unhappiness" → ["un", "##happy", "##ness"]
```

**Key Differences from BPE:**
- Uses **likelihood-based merging** (not frequency-based)
- Continuation tokens marked with `##`
- Used by: BERT, DistilBERT, ALBERT, mBERT

---

### SentencePiece

**Framework** that implements both BPE and Unigram Language Model algorithms.

**Features:**
- Treats input as raw Unicode
- No preprocessing required (handles spaces, punctuation)
- Language-agnostic
- Can use either BPE or Unigram algorithm

**Unigram Language Model (used in SentencePiece):**
- Starts with large vocabulary (all possible subwords)
- Trains a unigram language model
- Iteratively removes subwords that least affect likelihood
- Probabilistic approach, can handle multiple segmentations

**Example:**
```
Input: "Hello world!"
Output: ["▁Hello", "▁world", "!"]
(▁ indicates word beginning)
```

**Usage:** T5, XLNet, ALBERT (variant), multilingual models

---

## Comparison and Usage

### Summary: Tokenization Techniques → Models

| Tokenization Technique | Major Models Using It | Characteristics |
|------------------------|----------------------|-----------------|
| **Word-level** | Rare in modern LLMs (early NLP pipelines, NLTK, spaCy) | Simple but OOV problem |
| **WordPiece** | BERT, DistilBERT, ALBERT, mBERT | Likelihood-based, `##` prefix |
| **Byte-Pair Encoding (BPE)** | GPT-2, RoBERTa, XLM-R, LLaMA, Mistral | Frequency-based, greedy |
| **SentencePiece** | T5, XLNet, ALBERT (variant), multilingual models | Language-agnostic, BPE or Unigram |
| **Byte-level BPE** | GPT-3, GPT-4, Falcon, recent OpenAI models | Handles any text, no OOV |

---

### Choosing a Tokenization Method

**Considerations:**

1. **Model Architecture**
   - **BERT family**: WordPiece
   - **GPT family**: BPE or byte-level BPE
   - **T5**: SentencePiece (Unigram)

2. **Task Type**
   - Generation tasks: Often use BPE or SentencePiece
   - Classification tasks: WordPiece or BPE both work well

3. **Language**
   - English: Most methods work well
   - Morphologically rich languages (Turkish, Finnish): Subword methods preferred
   - Multilingual: SentencePiece or byte-level methods

4. **Vocabulary Size**
   - Balance between coverage and model size
   - Subword methods typically use 30K-50K tokens

---

### Tokenization Best Practices

1. **Consistency**: Use the same tokenizer for training and inference
2. **Special Tokens**: Handle special tokens (`[CLS]`, `[SEP]`, `[MASK]`) appropriately
3. **Padding/Truncation**: Set appropriate max length for your model
4. **Unknown Tokens**: Understand how your tokenizer handles OOV words
5. **Vocabulary Size**: Balance between coverage and model size

---

## Key Takeaways

1. **Subword tokenization** is the standard for modern NLP models
2. **BPE and WordPiece** are the most popular algorithms
3. **Byte-level methods** eliminate OOV issues completely
4. **Choice of tokenization** depends on model architecture and task
5. **Consistent tokenization** is crucial for model performance

For more details on how tokenization is used in specific models:
- [BERT](./9.%20BERT.md) - Uses WordPiece
- [GPT](./10.%20GPT.md) - Uses BPE or byte-level BPE
- [Transformer](./8.%20Transformer.md) - Architecture that processes tokens
