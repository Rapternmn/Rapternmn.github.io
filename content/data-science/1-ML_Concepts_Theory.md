+++
title = "Machine Learning Concepts & Theory"
date = 2025-11-22T10:00:00+05:30
draft = false
description = "A comprehensive guide covering fundamental ML concepts, algorithms, and mathematical foundations for interviews. Topics include bias-variance tradeoff, regularization, ensemble methods, optimization algorithms, and more."
+++

---

## 1. Bias-Variance Tradeoff

### Definition

**Bias** is the error due to overly simplistic assumptions in the learning algorithm. It measures how far off the model's predictions are from the true values on average.

**Variance** is the error due to too much sensitivity to small fluctuations in the training set. It measures how much the model's predictions vary across different training sets.

### Mathematical Formulation

The total error can be decomposed as:

```
Total Error = Bias² + Variance + Irreducible Error
```

```
E[(y - f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²
```

Where:
- **Bias**: $E[\hat{f}(x)] - f(x)$ (difference between expected prediction and true value)
- **Variance**: $E[(\hat{f}(x) - E[\hat{f}(x)])^2]$ (variability of predictions)
- **Irreducible Error**: $\sigma^2$ (noise in data, cannot be reduced)

### Key Relationships

| Model Type | Bias | Variance | Training Error | Test Error |
|------------|------|----------|----------------|------------|
| **High Bias (Underfitting)** | High | Low | High | High |
| **High Variance (Overfitting)** | Low | High | Low | High |
| **Optimal** | Balanced | Balanced | Moderate | Low |

### Symptoms of High Bias

- High training error
- High test error
- Model is not flexible enough
- Cannot capture underlying patterns
- **Solutions**: Increase model complexity, add features, reduce regularization

### Symptoms of High Variance

- Low training error
- High test error
- Model is too complex
- Fits noise in training data
- **Solutions**: Reduce model complexity, add regularization, get more data, use ensemble methods

### Tradeoff Principle

You want to find a balance—enough complexity to capture patterns, but not so much that it fits noise. The goal is to minimize total error by finding the optimal bias-variance tradeoff.

---

## 2. Classification vs Regression

### Classification

**Definition**: Predicts a discrete label or category.

**Output**: Categorical (e.g., spam/not spam, fraud/legitimate, class labels)

**Examples**:
- Fraud detection
- Churn prediction
- Email spam classification
- Image recognition
- Sentiment analysis

**Common Algorithms**:
- Logistic Regression
- Decision Trees
- Random Forest
- SVM
- Naive Bayes
- Neural Networks

### Regression

**Definition**: Predicts a continuous numerical value.

**Output**: Continuous (e.g., house price = ₹52.7L, temperature = 25.3°C)

**Examples**:
- Stock price prediction
- Weather forecasting
- Sales forecasting
- House price estimation
- Demand prediction

**Common Algorithms**:
- Linear Regression
- Polynomial Regression
- Ridge/Lasso Regression
- Decision Trees (for regression)
- Random Forest (for regression)
- Neural Networks

### Key Differences

| Aspect | Classification | Regression |
|--------|----------------|------------|
| **Output Type** | Discrete/Categorical | Continuous |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC | MSE, RMSE, MAE, R² |
| **Loss Functions** | Cross-entropy, Hinge loss | MSE, MAE, Huber loss |
| **Decision Boundary** | Separates classes | Fits a curve/surface |

---

## 3. Algorithm Selection

### Logistic Regression vs Decision Trees

#### Logistic Regression

**When to Use**:
- Simple, interpretable models needed
- Linear decision boundaries are sufficient
- Fast training and prediction
- Need probabilistic outputs
- Feature relationships are mostly linear
- Good baseline model

**Characteristics**:
- Linear model with sigmoid activation
- Assumes linear relationship between features and log-odds
- Can be regularized (L1/L2)
- Fast and memory efficient

**Mathematical Formulation**:

```
P(y=1|x) = 1 / (1 + e^(-(wᵀx + b))) = σ(wᵀx + b)
```

Where $\sigma$ is the sigmoid function.

#### Decision Trees

**When to Use**:
- Non-linear relationships expected
- Feature interactions are important
- Need interpretable rules
- Mixed data types (categorical + numerical)
- Want to capture complex patterns

**Characteristics**:
- Non-parametric model
- Handles non-linearities naturally
- Can model interactions between features
- Prone to overfitting (needs regularization)
- Can be combined into Random Forest or Gradient Boosting

**Splitting Criteria**:
- **Gini Impurity**: $Gini = 1 - \sum_{i=1}^{c} p_i^2$
- **Entropy**: $Entropy = -\sum_{i=1}^{c} p_i \log_2(p_i)$
- **Information Gain**: $IG = Entropy(parent) - \sum \frac{|child|}{|parent|} \times Entropy(child)$

### Algorithm Comparison Table

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Logistic Regression** | Binary classification, interpretability | Fast, interpretable, probabilistic | Linear boundaries only |
| **Decision Trees** | Non-linear patterns, interactions | Non-linear, interpretable | Prone to overfitting |
| **Random Forest** | Robust predictions, feature importance | Reduces overfitting, handles missing values | Less interpretable, slower |
| **XGBoost** | High performance, competitions | Very accurate, handles missing values | Complex, requires tuning |
| **SVM** | High-dimensional data, clear margins | Effective in high dimensions | Slow on large datasets |
| **Naive Bayes** | Text classification, small datasets | Fast, works well with small data | Strong independence assumption |

---

## 4. Regularization

### Overview

Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function that discourages overly complex models.

### L1 Regularization (Lasso)

**Penalty Term**: $L1 = \lambda \sum_{i=1}^{n} |w_i|$

**Loss Function**:
```
L = MSE + λ × Σ|w_i|
```

**Characteristics**:
- Leads to sparse models (some weights = 0)
- Performs feature selection automatically
- Can eliminate irrelevant features
- Useful for high-dimensional data with many irrelevant features

**Geometric Interpretation**: Constrains weights to lie within a diamond-shaped region (L1 ball)

### L2 Regularization (Ridge)

**Penalty Term**: $L2 = \lambda \sum_{i=1}^{n} w_i^2$

**Loss Function**:
```
L = MSE + λ × Σw_i²
```

**Characteristics**:
- Shrinks weights but doesn't zero them out
- All features are retained (just reduced in magnitude)
- Prevents large weights
- More stable than L1

**Geometric Interpretation**: Constrains weights to lie within a circular region (L2 ball)

### Elastic Net

**Penalty Term**: Combination of L1 and L2

**Loss Function**:
```
L = MSE + λ₁ × Σ|w_i| + λ₂ × Σw_i²
```

**Characteristics**:
- Best of both worlds
- Combines feature selection (L1) with weight shrinkage (L2)
- Useful for high-dimensional datasets with correlated features

### Comparison Table

| Regularization | Penalty | Effect | Use Case |
|----------------|---------|-------|----------|
| **L1 (Lasso)** | $\lambda \sum \|w_i\|$ | Sparse models, feature selection | High-dimensional data, feature selection needed |
| **L2 (Ridge)** | $\lambda \sum w_i^2$ | Shrinks all weights | General shrinkage, correlated features |
| **Elastic Net** | $\lambda_1 \sum \|w_i\| + \lambda_2 \sum w_i^2$ | Feature selection + shrinkage | High-dimensional datasets with correlations |

### Choosing $\lambda$ (Regularization Parameter)

- **Small $\lambda$**: Less regularization, model can be more complex
- **Large $\lambda$**: More regularization, simpler model
- **Optimal $\lambda$**: Found via cross-validation

---

## 5. Cross-Validation

### Definition

Cross-validation splits data into multiple folds to ensure the model generalizes well and prevents overfitting to a specific train/test split.

### K-Fold Cross-Validation

**Process**:
1. Split data into k equal parts (folds)
2. For each fold i:
   - Train on k-1 folds
   - Test on fold i
3. Repeat k times
4. Average the results

**Mathematical Formulation**:

```
CV(k) = (1/k) × Σ Error_i
```

Where $\text{Error}_i$ is the error on fold i.

### Types of Cross-Validation

#### 1. K-Fold CV
- Most common (typically k=5 or k=10)
- Good balance between bias and variance
- Computationally efficient

#### 2. Leave-One-Out CV (LOOCV)
- k = n (number of samples)
- Low bias, high variance
- Computationally expensive
- Useful for small datasets

#### 3. Stratified K-Fold CV
- Maintains class distribution in each fold
- Important for imbalanced datasets
- Ensures each fold has representative samples

#### 4. Time Series CV
- Respects temporal order
- Walk-forward validation
- Prevents data leakage

### Why Use Cross-Validation?

1. **Prevents Overfitting**: More reliable estimate of model performance
2. **Hyperparameter Tuning**: Find optimal hyperparameters reliably
3. **Model Selection**: Compare different models fairly
4. **Better Use of Data**: All data used for both training and validation

### Code Example (Conceptual)

```python
# K-Fold Cross-Validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores.append(score)

mean_score = np.mean(scores)
```

---

## 6. Ensemble Methods

### Overview

Ensemble methods combine multiple models to improve performance. Two main approaches: **Bagging** and **Boosting**.

### Bagging (Bootstrap Aggregating)

**Approach**: Train models in parallel on different bootstrap samples

**Goal**: Reduce variance

**Process**:
1. Create multiple bootstrap samples (sampling with replacement)
2. Train a model on each sample
3. Aggregate predictions (voting for classification, averaging for regression)

**Examples**: Random Forest

#### Random Forest

**How it Works**:

1. **Bootstrapping**: Randomly sample training data with replacement to create multiple datasets
2. **Random Feature Selection**: At each split, only consider a random subset of features
3. **Train Trees**: Train a decision tree on each bootstrap sample
4. **Ensemble Prediction**:
   - Classification: Majority vote
   - Regression: Average of predictions

**Key Features**:
- **Out-of-Bag (OOB) Error**: Built-in cross-validation using data not in bootstrap sample
- **Feature Importance**: Measures how much each feature reduces impurity across trees
- **Handles Missing Values**: Can impute missing values during training

**Advantages**:
- Reduces overfitting compared to single decision tree
- Handles large datasets efficiently
- Provides feature importance
- Works well with default parameters

### Boosting

**Approach**: Train models sequentially, each correcting errors of previous ones

**Goal**: Reduce both bias and variance

**Process**:
1. Start with initial prediction
2. Compute residuals (errors)
3. Fit new model to predict residuals
4. Add to ensemble with learning rate
5. Repeat

**Examples**: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

#### XGBoost (Extreme Gradient Boosting)

**Core Idea**: Sequential ensemble of weak learners (trees) where each tree corrects previous errors

**How XGBoost Works**:

1. Start with initial prediction (e.g., average for regression)
2. Compute residuals (errors) from current model
3. Fit a new decision tree to predict these residuals
4. Add tree to ensemble with learning rate (shrinkage)
5. Repeat for many iterations
6. Final prediction = sum of all trees

**Mathematical Formulation**:

```
ŷ_i = Σ f_k(x_i)
```

Where $f_k$ are the trees and the objective is:

```
Obj = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
```

Where:
- $l$ is the loss function
- $\Omega$ is the regularization term

**Key Features**:
- **Gradient-based**: Uses second-order gradients for better optimization
- **Regularization**: Built-in L1 and L2 regularization
- **Handles Missing Values**: Automatically learns best imputation
- **Parallel Processing**: Efficient implementation
- **Early Stopping**: Prevents overfitting

**XGBoost Advantages**:
- Very high accuracy
- Handles missing values automatically
- Regularization prevents overfitting
- Fast and scalable
- Feature importance available

#### XGBoost Ranker (Learning to Rank)

**How it Works**:

1. **Create Features**: Extract features for every document-query pair
2. **Create Groups**: Group documents by query
3. **Pairwise Training**: For each query group:
   - Generate all pairs (i, j) where label_i > label_j
   - Model learns to assign higher score to i than j
4. **Pairwise Loss**: Optimizes logistic loss over score differences

**Loss Function**:
```
L = Σ log(1 + e^(-(s_i - s_j)))
```

Where $s_i, s_j$ are predicted scores for documents i and j.

### Comparison: Bagging vs Boosting

| Feature | Bagging | Boosting |
|---------|---------|----------|
| **Approach** | Train models in parallel | Train models sequentially |
| **Goal** | Reduce variance | Reduce bias and variance |
| **Data Sampling** | Bootstrap (with replacement) | Weighted sampling (focus on errors) |
| **Model Independence** | Independent models | Dependent models |
| **Examples** | Random Forest | XGBoost, AdaBoost, LightGBM |
| **Overfitting** | Less prone | More prone (needs regularization) |
| **Training Speed** | Faster (parallel) | Slower (sequential) |

---

## 7. Evaluation Metrics

> **Note**: For a comprehensive guide to evaluation metrics, see [5-Evaluation_Metrics.md](./5-Evaluation_Metrics.md)

### Classification Metrics

#### Confusion Matrix

| | Predicted Positive | Predicted Negative |
|-------------------|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

#### Precision

**Definition**: Of all predicted positives, how many are actually positive?

**Formula**:
```
Precision = TP / (TP + FP)
```

**Use Case**: Important when false positives are costly
- Spam detection (don't want to mark legitimate emails as spam)
- Fraud detection (don't want to block legitimate transactions)

#### Recall (Sensitivity)

**Definition**: Of all actual positives, how many did we catch?

**Formula**:
```
Recall = TP / (TP + FN)
```

**Use Case**: Important when false negatives are costly
- Cancer detection (don't want to miss cancer cases)
- Security systems (don't want to miss threats)

#### F1 Score

**Definition**: Harmonic mean of precision and recall

**Formula**:
```
F1 = (2 × Precision × Recall) / (Precision + Recall) = 2TP / (2TP + FP + FN)
```

**Use Case**: When both precision and recall are important
- Balanced metric for imbalanced datasets
- Single metric to optimize

#### Accuracy

**Definition**: Overall correctness of predictions

**Formula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Limitation**: Can be misleading with imbalanced datasets

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**Definition**: Area under the curve of True Positive Rate (TPR) vs False Positive Rate (FPR) across different thresholds

**Formulas**:
```
TPR = TP / (TP + FN) = Recall
FPR = FP / (FP + TN)
AUC = ∫₀¹ TPR(FPR⁻¹(x)) dx
```

**Interpretation**:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier
- **AUC > 0.7**: Good classifier
- **AUC > 0.8**: Very good classifier

**Use Case**: Compare model quality independent of threshold
- Model comparison
- Threshold-independent evaluation

#### Precision-Recall AUC

**Definition**: Area under Precision-Recall curve

**Use Case**: Better than ROC-AUC for imbalanced datasets
- Focuses on positive class performance
- More informative when positive class is rare

### Regression Metrics

#### Mean Squared Error (MSE)

```
MSE = (1/n) × Σ(y_i - ŷ_i)²
```

- Penalizes large errors more
- Units are squared of target variable

#### Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √((1/n) × Σ(y_i - ŷ_i)²)
```

- Same units as target variable
- More interpretable than MSE

#### Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|y_i - ŷ_i|
```

- Less sensitive to outliers than MSE
- Linear penalty for errors

#### R² (Coefficient of Determination)

```
R² = 1 - (Σ(y_i - ŷ_i)²) / (Σ(y_i - ȳ)²)
```

- Proportion of variance explained
- Range: $-\infty$ to 1 (1 = perfect, 0 = no better than mean)

### Ranking Metrics

#### NDCG (Normalized Discounted Cumulative Gain)

**Definition**: Measures ranking quality, especially for search and recommendation systems

**Formulas**:

```
DCG@K = Σ(rel_i / log₂(i + 1))
NDCG@K = DCG@K / IDCG@K
```

Where:
- $rel_i$ = relevance score of item at position i
- $IDCG@K$ = Ideal DCG (DCG of perfect ranking)
- K = number of items to consider

**Use Case**: Search engines, recommendation systems, learning-to-rank

### Metric Selection Guide

| Scenario | Preferred Metric |
|----------|------------------|
| **Balanced binary classification** | Accuracy, F1, ROC-AUC |
| **Imbalanced binary classification** | Precision, Recall, F1, PR-AUC |
| **Spam/Fraud detection** | Precision (minimize false positives) |
| **Medical diagnosis** | Recall (minimize false negatives) |
| **Regression** | RMSE, MAE, R² |
| **Ranking problems** | NDCG, MAP (Mean Average Precision) |

---

## 8. Optimization Algorithms

### Gradient Descent

**Definition**: Optimization algorithm to minimize the loss function by updating weights in the opposite direction of the gradient.

**Update Rule**:
```
θ_(t+1) = θ_t - α × ∇_θ J(θ_t)
```

Where:
- $\theta$ = parameters (weights)
- $\alpha$ = learning rate
- $\nabla_\theta J$ = gradient of loss function

### Variants of Gradient Descent

#### 1. Batch Gradient Descent

- Uses full dataset for each update
- **Pros**: Stable convergence, accurate gradients
- **Cons**: Slow, memory intensive, can't update online
- **Use Case**: Small datasets, convex optimization

#### 2. Stochastic Gradient Descent (SGD)

- Uses one example per update
- **Pros**: Fast, can update online, escapes local minima
- **Cons**: Noisy updates, may not converge
- **Update Rule**: $\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t; x_i, y_i)$

#### 3. Mini-Batch Gradient Descent

- Uses small batch (typically 32, 64, 128) per update
- **Pros**: Balanced - faster than batch, more stable than SGD
- **Cons**: Need to tune batch size
- **Most Common**: Default choice for deep learning

### Advanced Optimizers

#### Momentum

**Idea**: Adds velocity to avoid local minima and speed up convergence

**Update Rules**:
```
v_t = β × v_(t-1) + (1-β) × ∇_θ J(θ_t)
θ_(t+1) = θ_t - α × v_t
```

Where $\beta$ is momentum coefficient (typically 0.9)

**Benefits**: Smoother updates, faster convergence, escapes shallow local minima

#### RMSprop

**Idea**: Adapts learning rate per parameter based on recent gradient magnitudes

**Update Rules**:
```
s_t = β × s_(t-1) + (1-β) × (∇_θ J(θ_t))²
θ_(t+1) = θ_t - (α / √(s_t + ε)) × ∇_θ J(θ_t)
```

**Benefits**: Handles non-stationary objectives, adapts to different parameter scales

#### Adam (Adaptive Moment Estimation)

**Idea**: Combines momentum and RMSprop - most popular optimizer

**Update Rules**:
```
m_t = β₁ × m_(t-1) + (1-β₁) × ∇_θ J(θ_t)  (momentum)
v_t = β₂ × v_(t-1) + (1-β₂) × (∇_θ J(θ_t))²  (RMSprop)
m̂_t = m_t / (1-β₁ᵗ),  v̂_t = v_t / (1-β₂ᵗ)  (bias correction)
θ_(t+1) = θ_t - (α / (√v̂_t + ε)) × m̂_t
```

**Default Parameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**Benefits**: 
- Combines benefits of momentum and adaptive learning rates
- Works well with default parameters
- Fast convergence
- Handles sparse gradients well

### Comparison Table

| Optimizer | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **SGD** | Simple, theoretical guarantees | Slow, needs tuning | Baseline, convex problems |
| **Momentum** | Faster convergence, escapes local minima | Extra hyperparameter | When SGD is slow |
| **RMSprop** | Adapts learning rate | May not converge | Non-stationary objectives |
| **Adam** | Fast, adaptive, works well default | Memory overhead | Deep learning (default choice) |

---

## 9. Handling Imbalanced Datasets

### Problem

When classes are imbalanced (e.g., 99% class A, 1% class B), models tend to:
- Predict majority class always
- High accuracy but poor performance on minority class
- Accuracy metric becomes misleading

### Solutions

#### 1. Resampling

**Oversampling Minority Class**:
- Duplicate minority samples
- Use SMOTE (Synthetic Minority Oversampling Technique)
- Use ADASYN (Adaptive Synthetic Sampling)

**Undersampling Majority Class**:
- Random undersampling
- Cluster-based undersampling
- Risk: Loss of information

**SMOTE Algorithm**:
1. For each minority sample, find k nearest neighbors
2. Randomly select one neighbor
3. Create synthetic sample along line connecting them
4. Repeat until desired balance

#### 2. Change Evaluation Metric

Instead of accuracy, use:
- **F1 Score**: Harmonic mean of precision and recall
- **Precision-Recall AUC**: Better for imbalanced data
- **ROC-AUC**: Threshold-independent
- **Confusion Matrix**: Detailed view of performance

#### 3. Class Weights

Penalize misclassifying minority class more:

```
Loss = Σ(w_(y_i) × L(y_i, ŷ_i))
```

Where $w_{y_i}$ is the weight for class $y_i$, typically:
```
w_class = n_total / (n_classes × n_class)
```

#### 4. Synthetic Data Generation

- **SMOTE**: Creates synthetic minority samples
- **ADASYN**: Adaptive version that focuses on harder-to-learn samples
- **GANs**: Generate synthetic samples (advanced)

#### 5. Anomaly Detection Approach

When minority class is extreme outlier:
- Treat as anomaly detection problem
- Use Isolation Forest, One-Class SVM
- Focus on detecting rare events

#### 6. Ensemble Methods

- Use ensemble with balanced sampling
- Combine multiple models trained on balanced subsets
- XGBoost with scale_pos_weight parameter

### Best Practices

1. **Start with metrics**: Use F1, PR-AUC instead of accuracy
2. **Try class weights first**: Often sufficient, no data loss
3. **Use SMOTE**: If class weights don't work
4. **Combine techniques**: Class weights + SMOTE + ensemble
5. **Collect more data**: If possible, get more minority class samples

---

## 10. Dimensionality Reduction

### PCA (Principal Component Analysis)

#### Intuition

PCA projects high-dimensional data onto a lower-dimensional subspace by finding orthogonal directions (principal components) of maximum variance.

#### Mathematical Formulation

**Goal**: Find directions of maximum variance

1. **Standardize data**: $X_{std} = \frac{X - \mu}{\sigma}$

2. **Compute covariance matrix**: 

```
C = (1/(n-1)) × X_stdᵀ × X_std
```

3. **Eigenvalue decomposition**:

```
C = P × Λ × Pᵀ
```

Where:
- $P$ = eigenvectors (principal components)
- $\Lambda$ = eigenvalues (variance explained)

4. **Project data**:

```
Y = X_std × P_k
```

Where $P_k$ contains top k eigenvectors.

#### Key Properties

- **Variance Preservation**: First PC captures most variance
- **Orthogonality**: PCs are orthogonal (uncorrelated)
- **Linear Transformation**: Linear projection
- **Dimensionality**: Reduces from d to k dimensions

#### Applications

- **Visualization**: Reduce to 2D/3D for plotting
- **Noise Reduction**: Remove low-variance components
- **Feature Extraction**: Use PCs as features
- **Speed**: Faster training with fewer dimensions
- **Multicollinearity**: Remove correlated features

#### Limitations

- **Linear Only**: Cannot capture non-linear relationships
- **Interpretability**: PCs are linear combinations, harder to interpret
- **Scale Dependent**: Sensitive to feature scaling

### Other Dimensionality Reduction Techniques

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

- Non-linear dimensionality reduction
- Great for visualization (2D/3D)
- Preserves local structure
- **Use Case**: Data exploration, visualization

#### UMAP (Uniform Manifold Approximation and Projection)

- Non-linear, preserves both local and global structure
- Faster than t-SNE
- **Use Case**: Visualization, feature reduction

#### Autoencoders

- Neural network-based
- Learns non-linear representations
- **Use Case**: Deep learning pipelines, feature learning

---

## 11. Core Algorithms

### Support Vector Machine (SVM)

#### Overview

SVM finds the optimal hyperplane that separates classes with maximum margin.

#### Mathematical Formulation

**Objective**: Maximize margin between classes

```
min_(w,b) (1/2) × ||w||² + C × Σ ξ_i
```

Subject to:
```
y_i(wᵀx_i + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
```

Where:
- $w$ = weight vector
- $b$ = bias
- $C$ = regularization parameter
- $\xi_i$ = slack variables (for soft margin)

#### Kernel Trick

For non-linear boundaries, use kernel functions:

- **Linear**: $K(x_i, x_j) = x_i^T x_j$
- **Polynomial**: $K(x_i, x_j) = (x_i^T x_j + 1)^d$
- **RBF**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$

#### When to Use

- High-dimensional data
- Clear margin of separation
- Non-linear boundaries (with kernels)
- Small to medium datasets

### K-Means Clustering

#### Overview

Unsupervised learning algorithm that partitions data into k clusters.

#### Algorithm

1. Initialize k cluster centroids randomly
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

#### Objective Function

```
J = Σ Σ ||x - μ_i||²
```

Where $\mu_i$ is the centroid of cluster $C_i$.

#### Limitations

- Need to specify k
- Sensitive to initialization
- Assumes spherical clusters
- Sensitive to outliers

#### Solutions

- **K-means++**: Better initialization
- **Elbow method**: Find optimal k
- **Silhouette score**: Evaluate clustering quality

### DBSCAN (Density-Based Spatial Clustering)

#### Overview

Density-based clustering algorithm that groups points in high-density regions.

#### Algorithm

1. **Core Point**: Point with at least min_samples neighbors within eps distance
2. **Border Point**: Point in cluster but not core point
3. **Noise Point**: Point not in any cluster

#### Parameters

- **eps (ε)**: Maximum distance between points in same cluster
- **min_samples**: Minimum points to form cluster

#### Advantages

- No need to specify number of clusters
- Handles non-spherical clusters
- Identifies outliers (noise points)
- Robust to outliers

#### Limitations

- Sensitive to eps and min_samples
- Struggles with varying densities
- Can be slow for large datasets

#### Use Cases

- Anomaly detection
- Image segmentation
- Geographic data clustering

### Hierarchical Clustering

#### Overview

Builds hierarchy of clusters using tree structure (dendrogram).

#### Types

**Agglomerative (Bottom-Up)**:
1. Start with each point as cluster
2. Merge closest clusters iteratively
3. Continue until desired number of clusters

**Divisive (Top-Down)**:
1. Start with all points in one cluster
2. Split clusters iteratively
3. Continue until desired number of clusters

#### Linkage Criteria

- **Single Linkage**: Minimum distance between clusters
- **Complete Linkage**: Maximum distance between clusters
- **Average Linkage**: Average distance between clusters
- **Ward Linkage**: Minimizes within-cluster variance

#### Advantages

- No need to specify k (can cut dendrogram)
- Visual representation (dendrogram)
- Handles non-spherical clusters

#### Limitations

- Computationally expensive: O(n³) or O(n² log n)
- Sensitive to noise and outliers
- Difficult for large datasets

### Gaussian Mixture Models (GMM)

#### Overview

Probabilistic clustering that assumes data comes from mixture of Gaussian distributions.

#### Formula

```
P(x) = Σ(π_k × N(x|μ_k, Σ_k))
```

Where:
- $\pi_k$: Mixing coefficient (weight of component k)
- $\mu_k$: Mean of component k
- $\Sigma_k$: Covariance of component k

#### Advantages

- Soft clustering (probabilistic assignments)
- Handles elliptical clusters
- Can model overlapping clusters

#### Limitations

- Assumes Gaussian distribution
- Sensitive to initialization
- Can be slow

### Anomaly Detection

#### Overview

Identifying rare items, events, or observations that differ significantly from majority.

#### Methods

**Statistical Methods**:
- Z-score: |z| > 3
- IQR: Outside Q1-1.5×IQR or Q3+1.5×IQR

**Distance-Based**:
- K-NN: Points with few neighbors
- Local Outlier Factor (LOF)

**Isolation Forest**:
- Random forests that isolate points
- Outliers easier to isolate

**Autoencoders**:
- Reconstruct input
- High reconstruction error → outlier

#### Use Cases

- Fraud detection
- Network intrusion
- Manufacturing defects
- Medical diagnosis

### Naive Bayes

#### Overview

Probabilistic classifier based on Bayes' theorem with strong independence assumption.

#### Bayes' Theorem

```
P(y|x) = (P(x|y) × P(y)) / P(x)
```

#### Naive Assumption

Features are conditionally independent given class:

```
P(x|y) = ∏ P(x_i|y)
```

#### Classification Rule

```
ŷ = argmax_y P(y) × ∏ P(x_i|y)
```

#### Variants

- **Gaussian NB**: For continuous features
- **Multinomial NB**: For count data (text classification)
- **Bernoulli NB**: For binary features

#### When to Use

- Text classification
- Small datasets
- Fast training and prediction
- Baseline model

### K-Nearest Neighbors (KNN)

#### Overview

Instance-based learning: classify based on k nearest neighbors.

#### Algorithm

1. Choose k
2. For new point, find k nearest training points
3. Classify by majority vote (classification) or average (regression)

#### Distance Metrics

- **Euclidean**: $d(x,y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
- **Manhattan**: $d(x,y) = \sum_{i=1}^{n} |x_i - y_i|$
- **Cosine**: $d(x,y) = 1 - \frac{x \cdot y}{||x|| ||y||}$

#### When to Use

- Simple, interpretable
- Non-linear boundaries
- Small datasets
- **Cons**: Slow prediction, sensitive to irrelevant features

---

## 12. Loss Functions

### Classification Loss Functions

#### Cross-Entropy Loss (Log Loss)

**Binary Classification**:
```
L = -(1/n) × Σ[y_i × log(ŷ_i) + (1-y_i) × log(1-ŷ_i)]
```

**Multi-class Classification**:
```
L = -(1/n) × Σ Σ(y_(i,c) × log(ŷ_(i,c)))
```

**Properties**: 
- Penalizes confident wrong predictions heavily
- Used with sigmoid/softmax outputs
- Differentiable, good for gradient descent

#### Hinge Loss (SVM)

```
L = (1/n) × Σ max(0, 1 - y_i(wᵀx_i + b))
```

**Properties**: 
- Used for SVM
- Less sensitive to outliers than cross-entropy
- Creates margin between classes

### Regression Loss Functions

#### Mean Squared Error (MSE)

```
L = (1/n) × Σ(y_i - ŷ_i)²
```

**Properties**: 
- Penalizes large errors more
- Differentiable everywhere
- Sensitive to outliers

#### Mean Absolute Error (MAE)

```
L = (1/n) × Σ|y_i - ŷ_i|
```

**Properties**: 
- Less sensitive to outliers
- Not differentiable at 0
- Linear penalty

#### Huber Loss

Combines MSE and MAE:

```
L = {
    (1/2) × (y_i - ŷ_i)²  if |y_i - ŷ_i| ≤ δ
    δ × |y_i - ŷ_i| - (1/2) × δ²  otherwise
}
```

**Properties**: 
- Robust to outliers
- Smooth transition between MSE and MAE
- Best of both worlds

---

## 13. Feature Engineering

### Overview

Feature engineering is the process of creating, transforming, and selecting features to improve model performance.

### Techniques

#### 1. Encoding Categorical Variables

- **One-Hot Encoding**: Create binary columns for each category
- **Label Encoding**: Assign integer to each category (for ordinal)
- **Target Encoding**: Encode by target variable statistics
- **Embeddings**: Learn dense representations (for high cardinality)

#### 2. Handling Missing Values

- **Deletion**: Remove rows/columns with missing values
- **Imputation**: Mean, median, mode, or learned values
- **Indicator Variables**: Create binary flag for missingness
- **Advanced**: KNN imputation, model-based imputation

#### 3. Feature Scaling

- **Standardization**: $z = \frac{x - \mu}{\sigma}$ (mean=0, std=1)
- **Normalization**: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$ (range [0,1])
- **Robust Scaling**: Use median and IQR (robust to outliers)

#### 4. Feature Transformation

- **Log Transform**: $x' = \log(x + 1)$ (for skewed data)
- **Polynomial Features**: $x^2, x^3, x_1 \times x_2$
- **Binning**: Convert continuous to categorical
- **Power Transforms**: Box-Cox, Yeo-Johnson

#### 5. Feature Creation

- **Interaction Terms**: $x_1 \times x_2$
- **Aggregations**: Mean, sum, count by groups
- **Time-based**: Day of week, hour, time since event
- **Domain-specific**: Ratios, differences, combinations

#### 6. Feature Selection

- **Filter Methods**: Correlation, mutual information, chi-square
- **Wrapper Methods**: Forward/backward selection, recursive feature elimination
- **Embedded Methods**: L1 regularization, tree-based importance

---

## 14. Model Selection & Hyperparameter Tuning

### Model Selection

Process of choosing the best algorithm for your problem.

#### Considerations

1. **Problem Type**: Classification vs Regression
2. **Data Size**: Small vs Large datasets
3. **Interpretability**: Need for explainability
4. **Training Time**: Computational constraints
5. **Prediction Time**: Real-time requirements
6. **Data Characteristics**: Linearity, interactions, missing values

### Hyperparameter Tuning

Process of finding optimal hyperparameters for chosen algorithm.

#### Methods

#### 1. Grid Search

- Exhaustive search over specified parameter grid
- **Pros**: Guaranteed to find best in grid
- **Cons**: Computationally expensive

#### 2. Random Search

- Randomly sample from parameter space
- **Pros**: Faster, often finds good solutions
- **Cons**: May miss optimal values

#### 3. Bayesian Optimization

- Uses probabilistic model to guide search
- **Pros**: Efficient, learns from previous evaluations
- **Cons**: More complex implementation

#### 4. Automated ML (AutoML)

- Tools like Optuna, Hyperopt
- Automatically search hyperparameter space
- Uses advanced optimization techniques

### Common Hyperparameters

| Algorithm | Key Hyperparameters |
|-----------|---------------------|
| **Logistic Regression** | C (regularization), penalty (L1/L2) |
| **Random Forest** | n_estimators, max_depth, min_samples_split |
| **XGBoost** | learning_rate, n_estimators, max_depth, subsample |
| **SVM** | C, kernel, gamma |
| **Neural Networks** | learning_rate, batch_size, layers, neurons |

---

## 15. Learning Curves & Overfitting

### Learning Curves

Plots showing model performance (training and validation) vs training set size.

#### What to Look For

**Underfitting (High Bias)**:
- Training error high and decreasing
- Validation error high and decreasing
- Gap between curves is small
- **Solution**: Increase model complexity

**Overfitting (High Variance)**:
- Training error low
- Validation error high
- Large gap between curves
- **Solution**: Reduce complexity, add regularization, get more data

**Good Fit**:
- Both errors decreasing
- Small gap between curves
- Errors converge to similar values

### Detecting Overfitting

1. **Learning Curves**: Plot train/val error vs iterations
2. **Validation Set**: Monitor validation performance
3. **Early Stopping**: Stop when validation error increases
4. **Cross-Validation**: Use CV to detect overfitting

### Preventing Overfitting

1. **Regularization**: L1, L2, dropout
2. **More Data**: Collect more training samples
3. **Feature Selection**: Remove irrelevant features
4. **Simpler Models**: Reduce model complexity
5. **Ensemble Methods**: Bagging reduces variance
6. **Cross-Validation**: Proper validation strategy

---

## Summary

This guide covers fundamental ML concepts essential for interviews:

- **Core Concepts**: Bias-variance tradeoff, overfitting, regularization
- **Algorithms**: Logistic regression, decision trees, ensemble methods, SVM, clustering
- **Evaluation**: Metrics for classification, regression, and ranking
- **Optimization**: Gradient descent and advanced optimizers
- **Practical Skills**: Handling imbalanced data, feature engineering, model selection

### Key Takeaways

1. **Bias-Variance Tradeoff**: Balance model complexity to minimize total error
2. **Regularization**: Essential for preventing overfitting
3. **Cross-Validation**: Reliable way to evaluate models
4. **Ensemble Methods**: Bagging reduces variance, boosting reduces bias
5. **Evaluation Metrics**: Choose metrics appropriate for your problem
6. **Feature Engineering**: Often more important than algorithm choice
7. **Hyperparameter Tuning**: Systematic search improves performance

---

## Additional Resources

- **Books**: "Introduction to Statistical Learning", "Elements of Statistical Learning"
- **Courses**: Andrew Ng's ML Course, Fast.ai
- **Practice**: Kaggle competitions, LeetCode ML problems
- **Papers**: Algorithm-specific papers for deep understanding

---

*Last Updated: 2024*

