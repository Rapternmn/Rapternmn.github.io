+++
title = "NLP Basics"
date = 2025-11-22T12:00:00+05:30
draft = false
description = "Comprehensive guide to NLP fundamentals covering text preprocessing, text representation methods, word embeddings, language models, sequence models, transformers, NLP tasks, and evaluation metrics."
+++

## 1. Text Preprocessing

### What are the steps in text preprocessing?

Text preprocessing is crucial for preparing raw text data for NLP models. The main steps include:

1. **Tokenization**
   - Splitting text into smaller units (words, subwords, or characters)
   - Example: `"I love NLP!"` → `["I", "love", "NLP", "!"]`

2. **Lowercasing**
   - Converting all text to lowercase for consistency
   - Note: May lose information (e.g., "US" vs "us")

3. **Removing Stopwords**
   - Removing common words that don't carry much meaning (e.g., "the", "is", "a")
   - Language-specific stopword lists available

4. **Stemming vs. Lemmatization**
   - **Stemming**: Reduces words to root form (e.g., "running" → "run")
   - **Lemmatization**: Uses vocabulary and morphological analysis (e.g., "better" → "good")
   - Lemmatization is more accurate but slower

5. **Removing Punctuation/Special Characters**
   - Removes or normalizes punctuation marks
   - May preserve some punctuation for certain tasks (e.g., question marks for sentiment)

6. **Handling Emojis, Numbers, Spelling Correction**
   - Emojis: Convert to text or remove
   - Numbers: Normalize or replace with placeholders
   - Spelling: Use spell checkers or autocorrect libraries

---

## 2. Text Representation Methods

### What is the difference between Bag-of-Words, TF-IDF, and Word Embeddings?

| Feature | Bag-of-Words (BoW) | TF-IDF | Word Embeddings |
|---------|-------------------|--------|-----------------|
| **Captures** | Frequency | Frequency + Importance | Semantic meaning |
| **Output** | Sparse matrix | Sparse matrix | Dense vector |
| **Dimensionality** | Vocabulary size | Vocabulary size | Fixed (e.g., 100-300) |
| **Context Awareness** | No | No | Yes (contextual embeddings) |
| **Examples** | CountVectorizer | TfidfVectorizer | Word2Vec, GloVe, FastText, BERT |

### Bag-of-Words (BoW)

- Represents text as a vector of word counts
- Ignores word order and context
- Creates a vocabulary from all unique words
- Each document is represented as a count vector

**Example:**
```
Document 1: "I love NLP"
Document 2: "I love Python"

Vocabulary: ["I", "love", "NLP", "Python"]
Doc1: [1, 1, 1, 0]
Doc2: [1, 1, 0, 1]
```

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF stands for **Term Frequency – Inverse Document Frequency**. It's a statistical measure that evaluates how important a word is to a document in a collection (corpus).

#### Components:

1. **TF (Term Frequency)**
   - How often a word appears in a document
   - Formula: `TF = (Number of times word appears in document) / (Total number of words in document)`

2. **IDF (Inverse Document Frequency)**
   - How rare the word is across all documents
   - Formula: `IDF = log(Total number of documents / Number of documents containing the word)`

3. **TF-IDF Score**
   - `TF-IDF = TF × IDF`
   - High TF-IDF: Word is frequent in this document but rare in the corpus (important)
   - Low TF-IDF: Word is common across all documents (less informative)

**Example:**
```
Document 1: "The cat sat on the mat"
Document 2: "The dog sat on the log"

For word "cat" in Doc1:
TF = 1/6 = 0.167
IDF = log(2/1) = 0.693
TF-IDF = 0.167 × 0.693 = 0.116
```

---

## 3. Word Embeddings

### What are Word Embeddings?

Word embeddings are **dense vector representations of words** that capture semantic similarity. Words with similar meanings have similar vectors.

**Key Properties:**
- Dense vectors (typically 50-300 dimensions)
- Capture semantic relationships
- Enable arithmetic operations (e.g., king - man + woman ≈ queen)

### Word2Vec

Word2Vec learns word embeddings by predicting words in context. It has two architectures:

1. **CBOW (Continuous Bag of Words)**
   - Predicts the target word from surrounding context words
   - Faster training
   - Better for frequent words

2. **Skip-gram**
   - Predicts context words from a target word
   - Better for rare words
   - More accurate overall

**Architecture:**
```
Input Layer → Hidden Layer (Embeddings) → Output Layer (Softmax)
```

### GloVe (Global Vectors for Word Representation)

- **Matrix factorization-based** approach
- Combines global statistics (co-occurrence matrix) with local context
- Trained on aggregated global word-word co-occurrence statistics

### Word2Vec vs. GloVe

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| **Approach** | Prediction-based | Matrix factorization-based |
| **Training** | Local context windows | Global co-occurrence statistics |
| **Efficiency** | Faster for large corpora | Slower but captures global patterns |
| **Performance** | Good for semantic tasks | Often better for analogy tasks |

### FastText

- Extension of Word2Vec
- Uses character n-grams instead of words
- Handles out-of-vocabulary words better
- Good for morphologically rich languages

---

## 4. Language Models

### What is a Language Model?

A language model assigns **probabilities to sequences of words**. It learns the probability distribution of words in a language.

**Mathematical Formulation:**
```
P(w1, w2, ..., wn) = P(w1) × P(w2|w1) × P(w3|w1,w2) × ... × P(wn|w1...wn-1)
```

### N-gram Models

- Predict next word based on previous N-1 words
- Example: Bigram (N=2) uses previous 1 word
- Requires smoothing techniques (e.g., Laplace, Kneser-Ney)
- Limited context window

### Neural Language Models

- Use neural networks (RNNs, LSTMs, Transformers)
- Can capture longer dependencies
- More context-aware
- Examples: GPT, BERT, T5

### Perplexity

**Perplexity** is a measure of uncertainty in predicting the next word. Lower perplexity indicates better model performance.

**Formula:**
```
Perplexity = 2^H, where H is the cross-entropy
```

**Interpretation:**
- Lower = Better model
- Perplexity of 10 means the model is as confused as if it had to choose uniformly among 10 possibilities

---

## 5. Sequence Models

### RNNs, LSTMs, and GRUs

Recurrent Neural Networks (RNNs) are designed to capture sequential dependencies in data.

#### Basic RNN

- Processes sequences step by step
- Maintains hidden state across time steps
- **Problem**: Vanishing/exploding gradients for long sequences

**Architecture:**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
y_t = W_hy * h_t + b_y
```

#### LSTM (Long Short-Term Memory)

- Solves vanishing gradient problem
- Uses gates (forget, input, output) to control information flow
- Better at capturing long-term dependencies

**Key Components:**
- **Forget Gate**: Decides what to forget
- **Input Gate**: Decides what new information to store
- **Output Gate**: Decides what parts of cell state to output

#### GRU (Gated Recurrent Unit)

- Simpler than LSTM (fewer parameters)
- Combines forget and input gates into update gate
- Often performs similarly to LSTM with less computation

### Why LSTM over RNN?

1. **Handles long-term dependencies** better
2. **Avoids vanishing gradients** through gating mechanisms
3. **More stable training** for long sequences

### Unidirectional vs. Bidirectional LSTM

| Type | Direction | Context |
|------|-----------|---------|
| **Unidirectional** | Past only | Can only see previous tokens |
| **Bidirectional** | Past + Future | Can see both past and future context |

**Use Cases:**
- Unidirectional: Language generation, real-time applications
- Bidirectional: Text classification, NER, sentiment analysis

---

## 6. Transformers and BERT

### What is the Transformer Architecture?

The Transformer is an **attention-based model** that eliminates recurrence and convolutions for sequence modeling. Introduced in "Attention is All You Need" (Vaswani et al., 2017).

**Key Innovation:** Self-attention mechanism allows parallel processing of entire sequences.

### What is Self-Attention?

Self-attention is a mechanism that **relates each word to all other words** in a sentence, computing attention scores to determine how much each word should attend to others.

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q** (Query): What am I looking for?
- **K** (Key): What do I contain?
- **V** (Value): What information do I provide?

### BERT (Bidirectional Encoder Representations from Transformers)

BERT is a **bidirectional encoder** that uses only the Transformer encoder stack.

**Key Features:**
- Pretrained on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- Bidirectional context understanding
- Excellent for classification tasks

**Architecture:**
```
Input → Token Embeddings + Position Embeddings + Segment Embeddings
  ↓
Transformer Encoder Stack (12 or 24 layers)
  ↓
Contextualized Embeddings
```

### BERT vs. GPT

| Feature | BERT | GPT |
|---------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Direction** | Bidirectional | Unidirectional (left-to-right) |
| **Training** | MLM + NSP | Autoregressive language modeling |
| **Best For** | Classification, NER, QA | Text generation, completion |
| **Context** | Sees full sequence | Only sees previous tokens |

---

## 7. NLP Tasks

### Text Classification

**Process:**
1. Preprocess text (tokenization, cleaning)
2. Vectorize (BoW, TF-IDF, embeddings)
3. Train model (Logistic Regression, SVM, LSTM, BERT)
4. Evaluate (Accuracy, Precision, Recall, F1)

**Applications:**
- Sentiment analysis
- Spam detection
- Topic classification

### Named Entity Recognition (NER)

Extracts entities like **person, location, organization** from text.

**Example:**
```
Input: "Apple is located in Cupertino, California."
Output: [ORG: Apple] [LOC: Cupertino] [LOC: California]
```

**Approaches:**
- Rule-based (regex, patterns)
- Machine learning (CRF, BiLSTM)
- Deep learning (BERT, RoBERTa)

### Question Answering (QA)

**How does Question Answering work in BERT?**

1. Input format: `[CLS] question [SEP] context [SEP]`
2. Model predicts:
   - Start token position in context
   - End token position in context
3. Answer = tokens between start and end positions

**Example:**
```
Question: "What is the capital of France?"
Context: "Paris is the capital and largest city of France."
Answer: "Paris" (start=1, end=1)
```

### Summarization

**Extractive Summarization:**
- Selects key sentences from the original text
- Preserves original wording
- Easier to implement
- Example: TextRank, BERT-based extractors

**Abstractive Summarization:**
- Generates new summary text
- More like human writing
- More challenging
- Example: T5, BART, GPT-based models

---

## 8. Evaluation Metrics

### Classification Metrics

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Language Model Metrics

- **Perplexity**: Measure of uncertainty (lower is better)
- **Cross-entropy**: Average negative log-likelihood

### NER / POS Tagging Metrics

- **Token-level F1**: F1-score calculated per token
- **Entity-level F1**: F1-score calculated per entity (exact match)
- **Partial Match F1**: Partial credit for overlapping entities

### Question Answering Metrics

- **Exact Match (EM)**: Percentage of predictions that exactly match ground truth
- **F1-Score**: Token-level overlap between prediction and ground truth

### Summarization Metrics

- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
  - ROUGE-N: N-gram overlap
  - ROUGE-L: Longest common subsequence
  - ROUGE-W: Weighted longest common subsequence

- **BLEU (Bilingual Evaluation Understudy)**
  - Precision-based metric
  - Measures n-gram precision with brevity penalty

---

## 9. Advanced Topics

### Contextual Embeddings

**Traditional Embeddings:**
- Same word always has the same embedding
- Example: "bank" (financial) and "bank" (river) have same vector

**Contextual Embeddings:**
- Embeddings vary based on surrounding words
- Example: BERT, ELMo, GPT
- "bank" in "I went to the bank" vs "I sat by the bank" have different embeddings

### Attention vs. Self-Attention

| Type | Query Source | Key/Value Source | Use Case |
|------|--------------|------------------|----------|
| **Attention** | One sequence | Another sequence | Encoder-Decoder (e.g., translation) |
| **Self-Attention** | Same sequence | Same sequence | Understanding relationships within one sequence |

**Self-Attention Example:**
```
"I love NLP"
Each word attends to all words including itself:
- "love" attends to "I" (subject), "NLP" (object), and itself
```

### Transfer Learning in NLP

**Process:**
1. **Pretraining**: Train on large unlabeled corpus (e.g., Wikipedia, books)
2. **Fine-tuning**: Adapt to specific downstream task with labeled data

**Benefits:**
- Requires less labeled data
- Better performance
- Faster training

**Examples:**
- BERT → Fine-tune for sentiment analysis
- GPT → Fine-tune for text generation
- T5 → Fine-tune for summarization

### Tokenization in BERT

BERT uses **WordPiece tokenization**:
- Splits words into subword units
- Handles out-of-vocabulary words
- Example: "playing" → ["play", "##ing"]
- `##` indicates continuation of previous token

**Special Tokens:**
- `[CLS]`: Classification token (used for sentence-level tasks)
- `[SEP]`: Separator token (separates sentences)
- `[MASK]`: Masked token (used during pretraining)
- `[UNK]`: Unknown token (rare, due to subword tokenization)

---

## Summary

This guide covers fundamental NLP concepts from preprocessing to advanced transformer architectures. Key takeaways:

1. **Text preprocessing** is essential for model performance
2. **Representation methods** evolve from sparse (BoW, TF-IDF) to dense (embeddings)
3. **Sequence models** progress from RNNs to LSTMs to Transformers
4. **Transfer learning** with pretrained models is the current standard
5. **Contextual embeddings** capture word meaning based on context

For deeper dives into specific topics, refer to:
- [Tokenization](./2.%20Tokenization.md)
- [Transformer Architecture](./8.%20Transformer.md)
- [BERT](./9.%20BERT.md)
- [GPT](./10.%20GPT.md)
