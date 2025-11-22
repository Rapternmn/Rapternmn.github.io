# Recommender Systems: Interview Q&A Guide

Comprehensive guide to recommender systems, collaborative filtering, and recommendation algorithms.


## 1. Recommender Systems Overview

### What is a Recommender System?

**Definition**: System that predicts user preferences and suggests items (products, movies, content) that users might like.

**Applications**:
- E-commerce (Amazon, eBay)
- Streaming (Netflix, Spotify)
- Social media (Facebook, Twitter)
- News (Google News)
- Job platforms (LinkedIn)

### Types of Recommender Systems

1. **Collaborative Filtering**: Based on user-item interactions
2. **Content-Based**: Based on item features
3. **Hybrid**: Combines multiple approaches
4. **Knowledge-Based**: Uses domain knowledge
5. **Demographic-Based**: Uses user demographics

---

## 2. Collaborative Filtering

### Definition

**Collaborative Filtering**: Recommends items based on similarity between users or items.

**Core Idea**: Users who liked similar items in the past will like similar items in the future.

### User-Based Collaborative Filtering

**Approach**: Find users similar to target user, recommend items they liked.

**Steps**:
1. Find similar users (neighbors)
2. Aggregate their preferences
3. Recommend items with high aggregated scores

**Similarity Metrics**:
- **Cosine Similarity**: $sim(u,v) = \frac{u \cdot v}{||u|| ||v||}$
- **Pearson Correlation**: Measures linear correlation
- **Jaccard Similarity**: For binary data

**Limitations**:
- Sparsity: User-item matrix is sparse
- Scalability: Expensive for large user bases
- Cold start: New users have no history

### Item-Based Collaborative Filtering

**Approach**: Find items similar to items user liked, recommend similar items.

**Steps**:
1. Compute item-item similarity matrix
2. For each user, find items similar to their liked items
3. Recommend items with high similarity scores

**Advantages over User-Based**:
- More stable (items change less than users)
- Better scalability
- Interpretable (item similarity)

**Example**: "Users who bought X also bought Y"

---

## 3. Content-Based Filtering

### Definition

**Content-Based Filtering**: Recommends items similar to items user liked, based on item features.

**Core Idea**: If user liked item with features {A, B, C}, recommend items with similar features.

### Process

1. **Extract features**: Item attributes (genre, director, keywords)
2. **Build user profile**: Aggregate features of liked items
3. **Compute similarity**: Compare item features to user profile
4. **Recommend**: Items with high similarity

### Feature Representation

**Text Items** (articles, descriptions):
- TF-IDF vectors
- Word embeddings
- Topic modeling

**Structured Items** (movies, products):
- Categorical features (genre, category)
- Numerical features (price, rating)
- Tags/keywords

### Advantages

- No cold start for new items
- Interpretable recommendations
- Works for niche users
- No sparsity issues

### Limitations

- Limited diversity (similar items only)
- Requires feature engineering
- Cold start for new users
- Overspecialization

---

## 4. Hybrid Approaches

### Why Hybrid?

**Combines strengths** of multiple approaches to overcome individual limitations.

### Common Hybrid Methods

#### 1. Weighted Hybrid

**Formula**:
```
score(u,i) = α × CF_score(u,i) + (1-α) × CB_score(u,i)
```

Where $\alpha$ is mixing weight.

#### 2. Switching Hybrid

- Use CF when user has history
- Use CB for new users/items
- Use knowledge-based as fallback

#### 3. Cascade Hybrid

- First filter: Content-based
- Second filter: Collaborative filtering
- Refine recommendations

#### 4. Feature Combination

- Combine user-item interactions with content features
- Train single model on combined features

#### 5. Meta-Level Hybrid

- Train separate models
- Use ensemble (voting, stacking)

---

## 5. Matrix Factorization

### Problem

**User-Item Matrix**: $R \in \mathbb{R}^{m \times n}$
- m users, n items
- $R_{ij}$ = rating of user i for item j
- Sparse (most entries unknown)

**Goal**: Predict missing ratings

### Matrix Factorization

**Decomposition**:
```
R ≈ U × Vᵀ
```

Where:
- $U \in \mathbb{R}^{m \times k}$: User latent factors
- $V \in \mathbb{R}^{n \times k}$: Item latent factors
- k: Number of latent dimensions

**Prediction**:
```
R̂_ij = U_i · V_jᵀ
```

### Optimization

**Objective**:
```
min_(U,V) Σ((R_ij - U_i V_jᵀ)²) + λ(||U||² + ||V||²)
```

**Regularization**: Prevents overfitting

### Advantages

- Handles sparsity well
- Captures latent factors
- Scalable
- Good performance

### Variants

#### Non-Negative Matrix Factorization (NMF)
- Constraint: $U, V \geq 0$
- Use: Interpretable factors

#### Probabilistic Matrix Factorization (PMF)
- Bayesian approach
- Uncertainty estimates

---

## 6. Deep Learning for Recommendations

### Neural Collaborative Filtering

**Architecture**:
- Embedding layers for users and items
- Multi-layer neural network
- Output: Rating/score

**Advantages**:
- Non-linear interactions
- Feature learning
- Handles complex patterns

### Wide & Deep Learning

**Components**:
- **Wide**: Memorization (linear model)
- **Deep**: Generalization (neural network)

**Use**: Combines interpretability with power

### Autoencoders

**Architecture**:
- Encoder: User-item vector → latent representation
- Decoder: Latent representation → reconstructed vector

**Use**: Denoising, handling sparsity

### Graph Neural Networks

**Approach**: Model user-item interactions as graph

**Advantages**:
- Captures higher-order interactions
- Handles heterogeneous data
- Good for social recommendations

---

## 7. Cold Start Problem

### Types

#### 1. New User Cold Start

**Problem**: No user history

**Solutions**:
- Demographic-based recommendations
- Popular items
- Content-based (if user provides preferences)
- Onboarding questions

#### 2. New Item Cold Start

**Problem**: No interaction history

**Solutions**:
- Content-based features
- Similar items
- Promotional placement
- Time-based decay

#### 3. New System Cold Start

**Problem**: No data at all

**Solutions**:
- Knowledge-based rules
- External data sources
- Hybrid with content features

---

## 8. Evaluation Metrics

### Offline Metrics

#### Precision@K

**Definition**: Fraction of recommended items that are relevant

```
Precision@K = |Relevant ∩ Recommended| / K
```

#### Recall@K

**Definition**: Fraction of relevant items that are recommended

```
Recall@K = |Relevant ∩ Recommended| / |Relevant|
```

#### F1@K

**Definition**: Harmonic mean of Precision and Recall

```
F1@K = (2 × Precision@K × Recall@K) / (Precision@K + Recall@K)
```

#### NDCG@K (Normalized Discounted Cumulative Gain)

**Definition**: Position-weighted ranking quality

```
NDCG@K = DCG@K / IDCG@K
```

Where:
- $DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$
- $IDCG@K$: Ideal DCG

**Use**: When position matters (top results more important)

#### MAP (Mean Average Precision)

**Definition**: Average precision across all users

```
MAP = (1/|U|) × Σ AP_u
```

Where $AP_u$ is average precision for user u.

### Online Metrics

- **Click-Through Rate (CTR)**: Clicks / Impressions
- **Conversion Rate**: Purchases / Recommendations
- **Revenue**: Total revenue from recommendations
- **Dwell Time**: Time spent on recommended items
- **Diversity**: Variety of recommended items

---

## 9. Challenges and Solutions

### Data Sparsity

**Problem**: User-item matrix is very sparse (most entries unknown)

**Solutions**:
- Matrix factorization
- Dimensionality reduction
- Use content features
- Imputation techniques

### Scalability

**Problem**: Millions of users and items

**Solutions**:
- Item-based CF (more stable)
- Matrix factorization (efficient)
- Approximate nearest neighbors
- Distributed computing

### Diversity

**Problem**: Recommendations too similar (filter bubble)

**Solutions**:
- Diversity metrics in optimization
- Re-ranking for diversity
- Serendipity (unexpected but relevant)
- Multi-objective optimization

### Fairness

**Problem**: Biased recommendations

**Solutions**:
- Fairness constraints
- Diverse representation
- Bias detection and mitigation
- Regular audits

### Privacy

**Problem**: User data privacy concerns

**Solutions**:
- Federated learning
- Differential privacy
- Local recommendations
- Privacy-preserving techniques

### Explainability

**Problem**: Users want to know why items are recommended

**Solutions**:
- Feature importance
- Similar users/items explanation
- Rule-based explanations
- Visual explanations

---

## Quick Reference

### Algorithm Selection

| Scenario | Recommended Approach |
|----------|---------------------|
| New users | Content-based, Popular items |
| New items | Content-based, Similar items |
| Sparse data | Matrix factorization |
| Rich features | Deep learning, Hybrid |
| Interpretability | Content-based, Rule-based |
| Scalability | Item-based CF, Matrix factorization |

### Key Concepts

- **Collaborative Filtering**: User/item similarity
- **Content-Based**: Item feature similarity
- **Matrix Factorization**: Latent factor models
- **Cold Start**: New users/items problem
- **Sparsity**: Missing data in user-item matrix

---

*Last Updated: 2024*

