+++
title = "Retrievers: Comprehensive Guide"
date = 2025-11-22T13:00:00+05:30
draft = false
weight = 6
description = "Comprehensive guide to retrieval systems in RAG and GenAI applications covering dense retrieval, sparse retrieval, hybrid retrieval, reranking, query processing, evaluation metrics, and optimization techniques."
+++

## 1. What is a Retriever?

### Definition

A **Retriever** is a component that finds and returns relevant documents/passages from a large corpus given a user query. It's a critical component in RAG (Retrieval-Augmented Generation) systems.

### Role in RAG Pipeline

```
User Query
    ↓
[Retriever] → Finds relevant documents
    ↓
Top-K Documents
    ↓
[LLM] → Generates answer using retrieved context
    ↓
Final Answer
```

### Key Responsibilities

1. **Query Understanding**: Process and understand user queries
2. **Document Matching**: Find relevant documents from corpus
3. **Ranking**: Order documents by relevance
4. **Filtering**: Apply metadata filters if needed
5. **Return Top-K**: Return most relevant documents

### Why Retrieval Matters

- **Accuracy**: Better retrieval → better final answers
- **Efficiency**: Reduces context window for LLM
- **Grounding**: Provides factual basis for LLM responses
- **Cost**: Fewer tokens to LLM = lower cost

---

## 2. Retrieval Methods

### 2.1 Dense Retrieval

**How it works**:
- Uses embeddings (dense vectors)
- Semantic similarity search
- Captures meaning, not just keywords

**Pros**:
- ✅ Handles synonyms and paraphrasing
- ✅ Semantic understanding
- ✅ Good for conceptual queries

**Cons**:
- ❌ Requires embedding model
- ❌ May miss exact keyword matches
- ❌ Computationally more expensive

### 2.2 Sparse Retrieval

**How it works**:
- Uses keyword matching
- TF-IDF, BM25 algorithms
- Lexical similarity

**Pros**:
- ✅ Fast and efficient
- ✅ Good for exact matches
- ✅ No model needed
- ✅ Interpretable

**Cons**:
- ❌ Misses synonyms
- ❌ No semantic understanding
- ❌ Struggles with paraphrasing

### 2.3 Hybrid Retrieval

**How it works**:
- Combines dense + sparse
- Best of both worlds
- Fusion of results

**Pros**:
- ✅ Handles both semantic and keyword matches
- ✅ Better coverage
- ✅ Higher recall

**Cons**:
- ❌ More complex
- ❌ Higher computational cost
- ❌ Need to tune fusion weights

### 2.4 Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Sparse (BM25)** | Very Fast | Medium | Keyword-heavy queries |
| **Dense (Embeddings)** | Fast | High | Semantic queries |
| **Hybrid** | Medium | Very High | Production RAG systems |

---

## 3. Dense Retrieval

### 3.1 How It Works

```
Query: "What is machine learning?"
    ↓
[Embedding Model]
    ↓
Query Vector: [0.23, -0.45, 0.67, ...]
    ↓
[Vector Similarity Search]
    ↓
Top-K Similar Documents
```

### 3.2 Embedding Models for Retrieval

**Recommended Models**:
- **BGE-base-en-v1.5**: Strong retrieval performance
- **E5-base-v2**: Instruction-aware, good for retrieval
- **multilingual-e5-base**: Multilingual retrieval
- **instructor-base**: Domain-specific fine-tuning

**Model Selection**:
- **General**: BGE or E5
- **Multilingual**: multilingual-e5
- **Domain-specific**: Fine-tune on your data

### 3.3 Implementation

**Basic Dense Retrieval**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Encode documents
documents = ["doc1 text", "doc2 text", "doc3 text"]
doc_embeddings = model.encode(documents)

# Encode query
query = "What is machine learning?"
query_embedding = model.encode([query])

# Compute similarity
similarities = np.dot(query_embedding, doc_embeddings.T)[0]

# Get top-K
top_k_indices = np.argsort(similarities)[::-1][:10]
top_k_docs = [documents[i] for i in top_k_indices]
```

**With Vector Store**:
```python
# Using Qdrant
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Query
results = client.search(
    collection_name="documents",
    query_vector=query_embedding[0],
    limit=10
)
```

### 3.4 Best Practices

1. **Normalize Embeddings**: L2 normalize for cosine similarity
2. **Batch Processing**: Encode multiple queries together
3. **Cache Embeddings**: Store document embeddings
4. **Model Selection**: Choose model based on your domain

---

## 4. Sparse Retrieval

### 4.1 BM25 Algorithm

**BM25 (Best Matching 25)** is the most popular sparse retrieval algorithm.

**Formula**:
```
BM25(q, d) = Σ IDF(q_i) × (f(q_i, d) × (k1 + 1)) / (f(q_i, d) + k1 × (1 - b + b × |d|/avgdl))

Where:
- q: query
- d: document
- f(q_i, d): frequency of term q_i in document d
- IDF(q_i): inverse document frequency of term q_i
- k1, b: tuning parameters (typically k1=1.5, b=0.75)
- |d|: document length
- avgdl: average document length
```

**Key Features**:
- Term frequency (TF): More occurrences = higher score
- Inverse document frequency (IDF): Rare terms = higher weight
- Length normalization: Prevents bias toward long documents

### 4.2 Implementation

**Using rank-bm25**:
```python
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Tokenize documents
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Create BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Query
query = "machine learning"
tokenized_query = word_tokenize(query.lower())

# Get scores
scores = bm25.get_scores(tokenized_query)

# Get top-K
top_k_indices = np.argsort(scores)[::-1][:10]
```

**Using Elasticsearch**:
```python
from elasticsearch import Elasticsearch

es = Elasticsearch()

# Index documents
for i, doc in enumerate(documents):
    es.index(index="documents", id=i, body={"text": doc})

# Search
results = es.search(
    index="documents",
    body={
        "query": {
            "match": {
                "text": "machine learning"
            }
        }
    },
    size=10
)
```

### 4.3 SPLADE (Sparse Lexical and Expansion)

**SPLADE** is a learned sparse retrieval method.

**Advantages over BM25**:
- Learned from data
- Better handling of synonyms
- Query expansion

**Use Cases**:
- When you have training data
- Need better than BM25
- Can't use dense retrieval

---

## 5. Hybrid Retrieval

### 5.1 Why Hybrid?

**Combines**:
- **Dense**: Semantic understanding
- **Sparse**: Exact keyword matching

**Benefits**:
- Higher recall
- Better coverage
- Handles both semantic and keyword queries

### 5.2 Fusion Methods

#### Reciprocal Rank Fusion (RRF)

**Formula**:
```
RRF_score(d) = Σ (1 / (k + rank_i(d)))

Where:
- rank_i(d): rank of document d in result set i
- k: constant (typically 60)
```

**Implementation**:
```python
def reciprocal_rank_fusion(results_list, k=60):
    doc_scores = {}
    
    for results in results_list:
        for rank, doc_id in enumerate(results, 1):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += 1 / (k + rank)
    
    # Sort by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]
```

#### Weighted Combination

**Formula**:
```
final_score = α × dense_score + β × sparse_score

Where:
- α + β = 1 (typically α = 0.7, β = 0.3)
```

**Implementation**:
```python
def weighted_fusion(dense_scores, sparse_scores, alpha=0.7):
    # Normalize scores to [0, 1]
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
    sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
    
    # Combine
    final_scores = alpha * dense_norm + (1 - alpha) * sparse_norm
    return final_scores
```

### 5.3 Implementation

**Complete Hybrid Retrieval**:
```python
def hybrid_retrieval(query, documents, alpha=0.7, top_k=10):
    # Dense retrieval
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(documents)
    dense_scores = np.dot(query_embedding, doc_embeddings.T)[0]
    
    # Sparse retrieval (BM25)
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    sparse_scores = bm25.get_scores(tokenized_query)
    
    # Fusion
    final_scores = weighted_fusion(dense_scores, sparse_scores, alpha)
    
    # Top-K
    top_k_indices = np.argsort(final_scores)[::-1][:top_k]
    return [documents[i] for i in top_k_indices]
```

### 5.4 Tuning Fusion Weights

**Approach**:
1. Create validation set with queries and relevant documents
2. Try different α values (0.0 to 1.0)
3. Measure recall@K for each α
4. Choose α with best performance

**Typical Values**:
- **Semantic-heavy**: α = 0.7-0.8 (more weight on dense)
- **Keyword-heavy**: α = 0.3-0.4 (more weight on sparse)
- **Balanced**: α = 0.5-0.6

---

## 6. Reranking

### 6.1 Why Rerank?

**Problem**: Initial retrieval may miss some relevant documents or rank them poorly.

**Solution**: Rerank top-K candidates (e.g., top 100) with a more powerful model.

**Benefits**:
- Higher precision in top results
- Better ranking quality
- Can use more expensive models (smaller candidate set)

### 6.2 Cross-Encoder Reranking

**How it works**:
- Takes query + document as input together
- More accurate than bi-encoder (separate encoding)
- Slower (can't pre-compute document embeddings)

**Model**:
```python
from sentence_transformers import CrossEncoder

# Load cross-encoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Rerank
pairs = [[query, doc] for doc in top_k_docs]
scores = reranker.predict(pairs)

# Sort by score
reranked_indices = np.argsort(scores)[::-1]
reranked_docs = [top_k_docs[i] for i in reranked_indices]
```

### 6.3 LLM-based Reranking

**Approach**:
- Use LLM to score relevance
- More expensive but potentially better

**Prompt**:
```
Rate the relevance of this document to the query (0-10):

Query: {query}
Document: {document}

Relevance score:
```

### 6.4 When to Rerank

**Use reranking when**:
- ✅ Need high precision in top results
- ✅ Can afford additional latency
- ✅ Have labeled data for training

**Skip reranking when**:
- ❌ Latency critical
- ❌ Initial retrieval is already good
- ❌ Limited computational resources

---

## 7. Retrieval Strategies

### 7.1 Single-Stage Retrieval

**Approach**: One retrieval step, return top-K

**Use when**:
- Fast response needed
- Good enough accuracy
- Simple implementation

### 7.2 Two-Stage Retrieval

**Approach**: 
1. Broad retrieval (get top 100-1000)
2. Rerank to get top-K

**Use when**:
- Need high precision
- Can afford extra latency
- Have reranking model

### 7.3 Hierarchical Retrieval

**Approach**:
1. Coarse retrieval (document level)
2. Fine retrieval (chunk level within documents)

**Use when**:
- Large documents
- Need specific passages
- Documents have multiple relevant sections

### 7.4 Query Expansion

**Approach**: Expand query with related terms

**Methods**:
- **Synonym expansion**: Add synonyms
- **LLM expansion**: Use LLM to generate related terms
- **Pseudo-relevance**: Use top results to expand query

**Example**:
```python
# Original query
query = "machine learning"

# Expanded query
expanded_query = "machine learning artificial intelligence neural networks deep learning"
```

### 7.5 Multi-Query Retrieval

**Approach**: Generate multiple query variations, retrieve for each, combine results

**Benefits**:
- Better coverage
- Handles query ambiguity
- Higher recall

**Implementation**:
```python
# Generate query variations using LLM
variations = generate_query_variations(query)  # ["ML", "AI learning", "neural networks"]

# Retrieve for each
all_results = []
for var in variations:
    results = retrieve(var, top_k=20)
    all_results.extend(results)

# Deduplicate and rerank
final_results = deduplicate_and_rerank(all_results, top_k=10)
```

---

## 8. Query Processing

### 8.1 Query Preprocessing

**Steps**:
1. **Normalization**: Lowercase, remove extra spaces
2. **Tokenization**: Split into tokens
3. **Stop word removal**: Remove common words (optional)
4. **Stemming/Lemmatization**: Reduce to root form (optional)

**Example**:
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_query(query):
    # Lowercase
    query = query.lower()
    
    # Tokenize
    tokens = word_tokenize(query)
    
    # Remove stop words (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Stemming (optional)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    
    return ' '.join(tokens)
```

### 8.2 Query Understanding

**Intent Detection**:
- Classify query type (factual, how-to, comparison)
- Route to appropriate retrieval strategy

**Entity Extraction**:
- Extract named entities
- Use for metadata filtering

**Query Classification**:
- Simple vs. complex queries
- Use different strategies

### 8.3 Query Reformulation

**Techniques**:
- **Spell correction**: Fix typos
- **Query expansion**: Add related terms
- **Query decomposition**: Break complex queries
- **LLM-based reformulation**: Use LLM to improve query

---

## 9. Evaluation Metrics

### 9.1 Retrieval Metrics

#### Precision@K

**Definition**: Fraction of retrieved documents that are relevant

```
Precision@K = (Relevant docs in top-K) / K
```

**Example**:
- Top 10 results, 7 are relevant
- Precision@10 = 7/10 = 0.7

#### Recall@K

**Definition**: Fraction of relevant documents retrieved

```
Recall@K = (Relevant docs in top-K) / (Total relevant docs)
```

**Example**:
- 20 relevant docs total, 7 in top 10
- Recall@10 = 7/20 = 0.35

#### NDCG@K (Normalized Discounted Cumulative Gain)

**Definition**: Measures ranking quality considering position

```
DCG@K = Σ (relevance_i / log2(i + 1))
NDCG@K = DCG@K / IDCG@K

Where:
- relevance_i: relevance score of result at position i
- IDCG: Ideal DCG (perfect ranking)
```

**Range**: [0, 1], higher is better

#### MRR (Mean Reciprocal Rank)

**Definition**: Average of reciprocal ranks of first relevant result

```
MRR = (1/n) × Σ (1 / rank_i)

Where:
- rank_i: position of first relevant result for query i
```

**Example**:
- Query 1: First relevant at position 2 → 1/2 = 0.5
- Query 2: First relevant at position 1 → 1/1 = 1.0
- MRR = (0.5 + 1.0) / 2 = 0.75

### 9.2 Evaluation Dataset

**Components**:
- **Queries**: Test queries
- **Relevant Documents**: Ground truth (human-labeled)
- **Relevance Scores**: Binary (relevant/not) or graded (0-4)

**Creating Evaluation Set**:
1. Collect queries
2. Human labelers mark relevant documents
3. Create query-document pairs with labels
4. Split into train/val/test

### 9.3 Evaluation Process

```python
def evaluate_retriever(retriever, test_queries, ground_truth, k=10):
    metrics = {
        'precision@k': [],
        'recall@k': [],
        'ndcg@k': [],
        'mrr': []
    }
    
    for query, relevant_docs in zip(test_queries, ground_truth):
        # Retrieve
        retrieved = retriever.retrieve(query, top_k=k)
        retrieved_ids = [doc.id for doc in retrieved]
        
        # Calculate metrics
        precision = len(set(retrieved_ids) & set(relevant_docs)) / k
        recall = len(set(retrieved_ids) & set(relevant_docs)) / len(relevant_docs)
        ndcg = calculate_ndcg(retrieved_ids, relevant_docs, k)
        mrr = calculate_mrr(retrieved_ids, relevant_docs)
        
        metrics['precision@k'].append(precision)
        metrics['recall@k'].append(recall)
        metrics['ndcg@k'].append(ndcg)
        metrics['mrr'].append(mrr)
    
    # Average
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## 10. Optimization Techniques

### 10.1 Chunking Strategy

**Impact on Retrieval**:
- **Too small**: May lose context
- **Too large**: May include irrelevant content
- **Optimal**: 300-500 tokens with overlap

**Strategies**:
- **Fixed size**: Simple, fast
- **Semantic chunking**: Split at sentence boundaries
- **Sliding window**: Overlap for context preservation

### 10.2 Embedding Optimization

**Techniques**:
1. **Fine-tuning**: Fine-tune on your domain
2. **Instruction tuning**: Use instruction-aware models (E5)
3. **Dimension reduction**: Reduce dimensions if needed
4. **Normalization**: L2 normalize for cosine similarity

### 10.3 Caching

**What to Cache**:
- **Query embeddings**: Cache computed query embeddings
- **Document embeddings**: Pre-compute and store
- **Retrieval results**: Cache frequent queries

**Implementation**:
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_query_embedding(query):
    return model.encode([query])[0]

def retrieve_with_cache(query, top_k=10):
    # Check cache
    cache_key = hashlib.md5(query.encode()).hexdigest()
    if cache_key in result_cache:
        return result_cache[cache_key]
    
    # Retrieve
    results = retriever.retrieve(query, top_k)
    
    # Cache
    result_cache[cache_key] = results
    return results
```

### 10.4 Batch Processing

**Batch Retrieval**:
```python
# Instead of one-by-one
for query in queries:
    results = retrieve(query)  # Slow

# Batch process
all_results = retrieve_batch(queries)  # Faster
```

### 10.5 Index Optimization

**Tune Index Parameters**:
- **HNSW**: Adjust `m` (connections), `ef_construction` (index quality)
- **IVF**: Adjust `nlist` (number of clusters)
- **Test with your data**: Different parameters for different datasets

---

## 11. Best Practices

### 11.1 Retrieval Pipeline Design

1. **Start Simple**: Begin with basic dense retrieval
2. **Add Sparse**: Add BM25 for hybrid if needed
3. **Add Reranking**: If precision is critical
4. **Optimize**: Tune based on evaluation

### 11.2 Query Handling

1. **Preprocess**: Normalize, tokenize queries
2. **Handle Edge Cases**: Empty queries, very long queries
3. **Error Handling**: Graceful degradation
4. **Logging**: Log queries and results for analysis

### 11.3 Document Management

1. **Chunking**: Optimal chunk size (300-500 tokens)
2. **Metadata**: Store useful metadata (source, date, category)
3. **Updates**: Handle document updates efficiently
4. **Versioning**: Track document versions

### 11.4 Performance

1. **Latency**: Target < 100ms for retrieval
2. **Throughput**: Handle concurrent queries
3. **Caching**: Cache frequent queries
4. **Monitoring**: Track latency, accuracy, errors

### 11.5 Evaluation

1. **Regular Evaluation**: Evaluate on test set regularly
2. **A/B Testing**: Test new retrieval strategies
3. **User Feedback**: Collect and use user feedback
4. **Continuous Improvement**: Iterate based on results

---

## 12. Common Interview Questions

### Q1: What is a retriever and why is it important in RAG?

**Answer**: A retriever finds relevant documents from a corpus given a query. It's critical in RAG because:
- Provides factual grounding for LLM
- Reduces hallucination
- Enables access to up-to-date information
- Improves answer quality

### Q2: What's the difference between dense and sparse retrieval?

**Answer**:
- **Dense**: Uses embeddings, semantic similarity, handles synonyms
- **Sparse**: Uses keywords (BM25), exact matches, faster
- **Hybrid**: Combines both for best results

### Q3: How do you choose between dense and sparse retrieval?

**Answer**: Consider:
- **Query type**: Semantic → dense, keyword-heavy → sparse
- **Domain**: Technical terms → sparse, general → dense
- **Resources**: Have embeddings → dense, need speed → sparse
- **Best practice**: Use hybrid for production

### Q4: What is reranking and when do you use it?

**Answer**: Reranking improves ranking of top-K candidates using a more powerful model. Use when:
- Need high precision in top results
- Can afford extra latency (~100-200ms)
- Have labeled data for training reranker

### Q5: How do you evaluate retrieval quality?

**Answer**: Use metrics:
- **Precision@K**: % of retrieved that are relevant
- **Recall@K**: % of relevant that are retrieved
- **NDCG@K**: Ranking quality
- **MRR**: Position of first relevant result

### Q6: What is hybrid retrieval and how does it work?

**Answer**: Hybrid combines dense + sparse retrieval:
1. Retrieve with both methods
2. Combine results (RRF or weighted fusion)
3. Return top-K

**Benefits**: Better coverage, handles both semantic and keyword queries

### Q7: How do you handle long queries or complex queries?

**Answer**:
1. **Query decomposition**: Break into sub-queries
2. **Query expansion**: Add related terms
3. **Multi-query retrieval**: Retrieve for each part, combine
4. **LLM reformulation**: Use LLM to improve query

### Q8: What are the trade-offs of different retrieval methods?

**Answer**:
- **Sparse**: Fast, good for keywords, misses synonyms
- **Dense**: Semantic, handles synonyms, requires model
- **Hybrid**: Best quality, more complex, higher cost
- **Reranking**: Highest precision, slower, more expensive

### Q9: How do you optimize retrieval latency?

**Answer**:
1. **Caching**: Cache query embeddings and results
2. **Batch processing**: Process multiple queries together
3. **Index optimization**: Tune index parameters
4. **Model selection**: Use faster models when appropriate
5. **Parallel processing**: Retrieve in parallel

### Q10: How do you handle updates to the document corpus?

**Answer**:
1. **Incremental updates**: Add new documents without full re-indexing
2. **Versioning**: Track document versions
3. **Periodic re-indexing**: Full re-index if needed
4. **Soft deletes**: Mark as deleted, remove later
5. **Update strategy**: Depends on vector store capabilities

---

## Summary

### Key Takeaways

- **Retrievers** are critical for RAG systems, finding relevant documents
- **Dense retrieval** uses embeddings for semantic search
- **Sparse retrieval** uses keywords (BM25) for exact matches
- **Hybrid retrieval** combines both for best results
- **Reranking** improves precision in top results
- **Evaluation** using Precision@K, Recall@K, NDCG, MRR
- **Optimization**: Caching, batching, index tuning

### Recommended Approach

1. **Start**: Dense retrieval with BGE/E5
2. **Improve**: Add sparse (BM25) for hybrid
3. **Optimize**: Add reranking if needed
4. **Evaluate**: Regular evaluation and iteration

---

*Last Updated: 2024*

