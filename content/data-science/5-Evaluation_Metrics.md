# Evaluation Metrics: Comprehensive Guide

A comprehensive guide to evaluation metrics for classification, regression, ranking, and specialized ML tasks.


## 1. Classification Metrics

### Confusion Matrix

The foundation for all classification metrics. A 2x2 table showing predicted vs actual labels.

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

**Key Terms:**
- **TP**: Correctly predicted positive cases
- **TN**: Correctly predicted negative cases
- **FP**: Incorrectly predicted as positive (Type I error)
- **FN**: Incorrectly predicted as negative (Type II error)

---

### Accuracy

**Definition**: Overall correctness of predictions

**Formula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Range**: 0 to 1 (higher is better)

**Use Case**: 
- Balanced datasets
- When all classes are equally important
- Quick sanity check

**Limitations**:
- Misleading with imbalanced datasets
- Example: 99% accuracy with 99% negative class means model predicts all negatives

---

### Precision

**Definition**: Of all predicted positives, how many are actually positive?

**Formula**:
```
Precision = TP / (TP + FP)
```

**Range**: 0 to 1 (higher is better)

**Interpretation**: "When the model says positive, how often is it right?"

**Use Cases**:
- **Spam detection**: Don't want to mark legitimate emails as spam
- **Fraud detection**: Don't want to block legitimate transactions
- **Recommendation systems**: Don't want to show irrelevant items
- When **false positives are costly**

**Trade-off**: High precision often means lower recall

---

### Recall (Sensitivity, True Positive Rate)

**Definition**: Of all actual positives, how many did we catch?

**Formula**:
```
Recall = TP / (TP + FN) = Sensitivity = TPR
```

**Range**: 0 to 1 (higher is better)

**Interpretation**: "Of all actual positives, what fraction did we find?"

**Use Cases**:
- **Medical diagnosis**: Don't want to miss cancer cases
- **Security systems**: Don't want to miss threats
- **Search engines**: Don't want to miss relevant results
- When **false negatives are costly**

**Trade-off**: High recall often means lower precision

---

### Specificity (True Negative Rate)

**Definition**: Of all actual negatives, how many did we correctly identify?

**Formula**:
```
Specificity = TN / (TN + FP) = TNR
```

**Range**: 0 to 1 (higher is better)

**Use Case**: Important when correctly identifying negatives matters (e.g., screening tests)

---

### F1 Score

**Definition**: Harmonic mean of precision and recall

**Formula**:
```
F1 = (2 × Precision × Recall) / (Precision + Recall) = 2TP / (2TP + FP + FN)
```

**Range**: 0 to 1 (higher is better)

**Why Harmonic Mean?**: 
- Penalizes extreme values more than arithmetic mean
- Only high when both precision and recall are high

**Use Cases**:
- Balanced metric for imbalanced datasets
- Single metric to optimize when both precision and recall matter
- Default metric when unsure which to prioritize

**Variants**:
- **Fβ Score**: Weighted F1 score
  ```
  F_β = (1 + β²) × (Precision × Recall) / ((β² × Precision) + Recall)
  ```
  - β > 1: Favors recall
  - β < 1: Favors precision
  - β = 1: Standard F1

---

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**Definition**: Area under the curve of True Positive Rate (TPR) vs False Positive Rate (FPR) across different classification thresholds

**Formulas**:
```
TPR = TP / (TP + FN) = Recall
FPR = FP / (FP + TN)
AUC = ∫₀¹ TPR(FPR⁻¹(x)) dx
```

**Range**: 0 to 1
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier (no better than chance)
- **AUC < 0.5**: Worse than random (flip predictions)
- **AUC > 0.7**: Good classifier
- **AUC > 0.8**: Very good classifier
- **AUC > 0.9**: Excellent classifier

**Interpretation**: Probability that a randomly chosen positive instance will have a higher predicted probability than a randomly chosen negative instance

**Advantages**:
- Threshold-independent evaluation
- Works well with balanced datasets
- Good for model comparison

**Limitations**:
- Can be misleading with highly imbalanced datasets
- Doesn't show actual precision/recall at specific thresholds

**Use Cases**:
- Model comparison
- Threshold selection
- Balanced binary classification

---

### Precision-Recall AUC (PR-AUC)

**Definition**: Area under the Precision-Recall curve

**Range**: 0 to 1 (higher is better)

**Advantages over ROC-AUC**:
- Better for imbalanced datasets
- Focuses on positive class performance
- More informative when positive class is rare
- Less affected by true negatives

**Use Cases**:
- Imbalanced binary classification
- When positive class is rare
- When false positives and false negatives both matter

**Rule of Thumb**: Use PR-AUC when positive class < 10% of data

---

### Matthews Correlation Coefficient (MCC)

**Definition**: Balanced metric that considers all four confusion matrix values

**Formula**:
```
MCC = (TP × TN - FP × FN) / √((TP + FP)(TP + FN)(TN + FP)(TN + FN))
```

**Range**: -1 to +1
- **+1**: Perfect positive correlation
- **0**: Random predictions
- **-1**: Perfect negative correlation (inverse predictions)

**Advantages**:
- Works well with imbalanced datasets
- Considers all confusion matrix values
- Only high when all classes are predicted well

**Use Case**: Balanced evaluation metric for binary classification

---

## 2. Regression Metrics

### Mean Squared Error (MSE)

**Definition**: Average of squared differences between predicted and actual values

**Formula**:
```
MSE = (1/n) × Σ(y_i - ŷ_i)²
```

**Range**: 0 to ∞ (lower is better)

**Properties**:
- Penalizes large errors more (due to squaring)
- Units are squared of target variable
- Sensitive to outliers

**Use Cases**:
- When large errors are particularly bad
- When errors should be penalized quadratically

---

### Root Mean Squared Error (RMSE)

**Definition**: Square root of MSE

**Formula**:
```
RMSE = √MSE = √((1/n) × Σ(y_i - ŷ_i)²)
```

**Range**: 0 to ∞ (lower is better)

**Advantages over MSE**:
- Same units as target variable (more interpretable)
- Still penalizes large errors
- Easier to understand in context

**Use Cases**:
- Most common regression metric
- When interpretability matters
- When you want to penalize large errors

---

### Mean Absolute Error (MAE)

**Definition**: Average of absolute differences between predicted and actual values

**Formula**:
```
MAE = (1/n) × Σ|y_i - ŷ_i|
```

**Range**: 0 to ∞ (lower is better)

**Properties**:
- Less sensitive to outliers than MSE/RMSE
- Linear penalty for errors
- Same units as target variable
- More robust to outliers

**Use Cases**:
- When outliers should not dominate the metric
- When all errors should be treated equally
- Robust regression evaluation

---

### Mean Absolute Percentage Error (MAPE)

**Definition**: Average of absolute percentage errors

**Formula**:
```
MAPE = (100/n) × Σ|(y_i - ŷ_i) / y_i|
```

**Range**: 0 to ∞ (lower is better, expressed as percentage)

**Advantages**:
- Scale-independent (percentage)
- Easy to interpret
- Good for comparing across different scales

**Limitations**:
- Undefined when actual value is 0
- Can be biased when actual values are small
- Asymmetric (penalizes over-prediction differently than under-prediction)

**Use Cases**:
- Forecasting
- When relative errors matter more than absolute
- Comparing models across different scales

---

### R² (Coefficient of Determination)

**Definition**: Proportion of variance in target variable explained by the model

**Formula**:
```
R² = 1 - (Σ(y_i - ŷ_i)²) / (Σ(y_i - ȳ)²) = 1 - SS_res / SS_tot
```

Where:
- $SS_{res}$ = Sum of squares of residuals
- $SS_{tot}$ = Total sum of squares

**Range**: $-\infty$ to 1
- **R² = 1**: Perfect predictions (model explains 100% of variance)
- **R² = 0**: Model performs as well as predicting the mean
- **R² < 0**: Model performs worse than predicting the mean

**Interpretation**: 
- 0.7 means model explains 70% of variance
- Higher is better, but context matters

**Use Cases**:
- Model comparison
- Understanding model explanatory power
- Standard regression evaluation

**Adjusted R²**: Accounts for number of features
```
R²_adj = 1 - ((1 - R²)(n - 1)) / (n - p - 1)
```
Where p = number of features

---

### Mean Squared Logarithmic Error (MSLE)

**Definition**: MSE of logarithmic values

**Formula**:
```
MSLE = (1/n) × Σ(log(1 + y_i) - log(1 + ŷ_i))²
```

**Properties**:
- Penalizes underestimation more than overestimation
- Works well with targets that have wide ranges
- Less sensitive to outliers

**Use Cases**:
- When target has exponential growth
- When relative errors matter
- Forecasting with wide ranges

---

## 3. Multi-class Classification Metrics

### Macro-Averaged Metrics

**Definition**: Calculate metric for each class separately, then average

**Example - Macro F1**:
```
MacroF1 = (1/C) × Σ F1_i
```

Where C = number of classes

**Properties**:
- Treats all classes equally
- Good when all classes are equally important
- Can be dominated by rare classes

---

### Micro-Averaged Metrics

**Definition**: Aggregate all TP, FP, TN, FN across classes, then calculate metric

**Example - Micro F1**:
- Aggregate all TP, FP, FN across classes
- Calculate F1 from aggregated values

**Properties**:
- Weighted by class frequency
- Good when class imbalance exists
- More influenced by frequent classes

---

### Weighted-Averaged Metrics

**Definition**: Calculate metric for each class, then average weighted by class frequency

**Example - Weighted F1**:
```
WeightedF1 = Σ(w_i × F1_i)
```

Where $w_i$ = proportion of class i in dataset

**Properties**:
- Accounts for class imbalance
- More representative of overall performance
- Good default for imbalanced multi-class

---

## 4. Ranking Metrics

### Mean Average Precision (MAP)

**Definition**: Average of Average Precision (AP) across all queries

**Formula**:
```
AP@K = (1/R) × Σ(P@k × rel_k)
MAP = (1/Q) × Σ AP_q
```

Where:
- $P@k$ = Precision at position k
- $rel_k$ = 1 if item at position k is relevant, else 0
- $R$ = number of relevant items
- $Q$ = number of queries

**Use Cases**: Information retrieval, search engines, recommendation systems

---

### Normalized Discounted Cumulative Gain (NDCG)

**Definition**: Measures ranking quality with position-based discounting

**Formulas**:
```
DCG@K = Σ(rel_i / log₂(i + 1))
NDCG@K = DCG@K / IDCG@K
```

Where:
- $rel_i$ = relevance score of item at position i
- $IDCG@K$ = Ideal DCG (DCG of perfect ranking)
- K = number of items to consider

**Range**: 0 to 1 (higher is better)

**Properties**:
- Accounts for position (top results matter more)
- Handles graded relevance
- Normalized for comparison

**Use Cases**:
- Search engines
- Recommendation systems
- Learning-to-rank problems

---

### Mean Reciprocal Rank (MRR)

**Definition**: Average of reciprocal ranks of first relevant item

**Formula**:
```
MRR = (1/Q) × Σ(1 / rank_q)
```

Where $rank_q$ = position of first relevant item for query q

**Use Cases**: When only the first relevant result matters (e.g., question answering)

---

## 5. Clustering Metrics

### Silhouette Score

**Definition**: Measures how similar an object is to its own cluster vs other clusters

**Formula**:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- $a(i)$ = average distance to points in same cluster
- $b(i)$ = average distance to points in nearest other cluster

**Range**: -1 to 1
- **+1**: Well-clustered
- **0**: On cluster boundary
- **-1**: Mis-clustered

**Use Cases**: Unsupervised clustering evaluation, determining optimal number of clusters

---

### Adjusted Rand Index (ARI)

**Definition**: Measures similarity between two clusterings, adjusted for chance

**Range**: -1 to 1 (higher is better)
- **1**: Perfect agreement
- **0**: Random labeling
- **Negative**: Worse than random

**Use Cases**: Comparing clusterings, evaluating clustering quality

---

## 6. Metric Selection Guide

| Scenario | Preferred Metrics | Why |
|----------|------------------|-----|
| **Balanced binary classification** | Accuracy, F1, ROC-AUC | All classes equally important |
| **Imbalanced binary classification** | Precision, Recall, F1, PR-AUC | Focus on positive class |
| **Spam/Fraud detection** | Precision | Minimize false positives |
| **Medical diagnosis** | Recall, Sensitivity | Minimize false negatives |
| **Regression** | RMSE, MAE, R² | Standard regression metrics |
| **Ranking problems** | NDCG, MAP, MRR | Position matters |
| **Multi-class balanced** | Macro-averaged metrics | All classes equal |
| **Multi-class imbalanced** | Weighted-averaged metrics | Account for imbalance |
| **Clustering** | Silhouette Score, ARI | Unsupervised evaluation |

---

## 7. Threshold Selection

### Optimal Threshold Selection

**Methods**:
1. **Youden's J Statistic**: Maximize (Sensitivity + Specificity - 1)
2. **F1 Score**: Maximize F1 score
3. **Cost-based**: Minimize total cost given FP and FN costs
4. **ROC Curve**: Choose point closest to (0,1)
5. **PR Curve**: Choose point with best precision-recall trade-off

**Considerations**:
- Business requirements (cost of FP vs FN)
- Class imbalance
- Desired precision/recall balance

---

## Quick Reference Summary

### Classification
- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we find?
- **F1**: Harmonic mean of precision and recall
- **ROC-AUC**: Threshold-independent model quality
- **PR-AUC**: Better for imbalanced data

### Regression
- **MSE**: Squared errors (penalizes large errors)
- **RMSE**: Square root of MSE (interpretable units)
- **MAE**: Absolute errors (robust to outliers)
- **R²**: Variance explained

### Ranking
- **NDCG**: Position-weighted ranking quality
- **MAP**: Average precision across queries
- **MRR**: Reciprocal rank of first relevant result

---

*Last Updated: 2024*

