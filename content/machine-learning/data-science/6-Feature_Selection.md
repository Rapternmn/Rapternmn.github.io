+++
title = "Feature Selection Techniques"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Comprehensive guide to feature selection techniques. Covers filter methods, wrapper methods, embedded methods, dimensionality reduction, feature importance, and best practices for selecting optimal features."
+++

---

## Introduction

**Feature Selection** is the process of selecting a subset of relevant features for model construction. It helps reduce overfitting, improve model performance, reduce training time, and enhance model interpretability.

**Key Principle**: Not all features are created equal. Selecting the right features is often more important than selecting the right algorithm.

---

## 1. Why Feature Selection?

### Benefits

1. **Reduces Overfitting**: Fewer features → simpler model → less overfitting
2. **Improves Performance**: Removes noise and irrelevant features
3. **Faster Training**: Fewer features → faster computation
4. **Better Interpretability**: Easier to understand models with fewer features
5. **Reduces Curse of Dimensionality**: High-dimensional data is sparse and hard to model

### When to Use Feature Selection

- **High-dimensional data**: Many features relative to samples
- **Irrelevant features**: Features that don't contribute to prediction
- **Redundant features**: Highly correlated features
- **Computational constraints**: Need faster training/prediction
- **Interpretability**: Need to understand which features matter

---

## 2. Feature Selection Methods

### Categories

1. **Filter Methods**: Select features based on statistical measures
2. **Wrapper Methods**: Use a model to evaluate feature subsets
3. **Embedded Methods**: Feature selection built into model training
4. **Dimensionality Reduction**: Transform features to lower dimensions

---

## 3. Filter Methods

Filter methods evaluate features independently of any machine learning algorithm. They use statistical measures to score features.

### Univariate Statistical Tests

#### Chi-Square Test (Categorical Features)

```python
from sklearn.feature_selection import chi2, SelectKBest

# For categorical features
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected features
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features}")

# Get scores
scores = selector.scores_
```

#### F-Test (ANOVA) (Numerical Features)

```python
from sklearn.feature_selection import f_classif, SelectKBest

# For classification
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# For regression
from sklearn.feature_selection import f_regression
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)
```

#### Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# For classification
mi_scores = mutual_info_classif(X, y)
feature_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Select top k
top_k = 10
selected_features = feature_scores.head(top_k).index

# For regression
mi_scores = mutual_info_regression(X, y)
```

### Correlation-Based Methods

#### Remove Highly Correlated Features

```python
# Calculate correlation matrix
correlation_matrix = X.corr().abs()

# Find highly correlated pairs
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Find features with correlation > threshold
threshold = 0.8
high_corr_features = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]

# Remove one from each highly correlated pair
to_remove = set()
for col in high_corr_features:
    for row in upper_triangle.index:
        if upper_triangle.loc[row, col] > threshold:
            to_remove.add(col)
            break

X_filtered = X.drop(columns=to_remove)
```

#### Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with low variance (likely constant)
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Get selected features
selected_features = X.columns[selector.get_support()]
```

### Information Gain / Entropy

```python
from sklearn.feature_selection import mutual_info_classif

# Calculate information gain
ig_scores = mutual_info_classif(X, y, random_state=42)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': ig_scores
}).sort_values('importance', ascending=False)
```

---

## 4. Wrapper Methods

Wrapper methods use a machine learning model to evaluate feature subsets. They search for the best subset of features.

### Forward Selection

```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# Forward selection
sfs = SequentialFeatureSelector(
    RandomForestClassifier(n_estimators=100, random_state=42),
    k_features=10,
    forward=True,
    scoring='accuracy',
    cv=5
)

sfs.fit(X, y)
selected_features = X.columns[list(sfs.k_feature_idx_)]
print(f"Selected features: {selected_features}")
```

### Backward Elimination

```python
# Backward elimination
sfs = SequentialFeatureSelector(
    RandomForestClassifier(n_estimators=100, random_state=42),
    k_features=10,
    forward=False,  # Backward elimination
    scoring='accuracy',
    cv=5
)

sfs.fit(X, y)
selected_features = X.columns[list(sfs.k_feature_idx_)]
```

### Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# RFE
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X, y)

# Get selected features
selected_features = X.columns[selector.support_]
print(f"Selected features: {selected_features}")

# Get feature rankings
feature_rankings = pd.DataFrame({
    'feature': X.columns,
    'rank': selector.ranking_
}).sort_values('rank')
```

### Recursive Feature Elimination with Cross-Validation (RFECV)

```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# RFECV automatically finds optimal number of features
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(5)

selector = RFECV(estimator, step=1, cv=cv, scoring='accuracy')
selector = selector.fit(X, y)

# Optimal number of features
print(f"Optimal number of features: {selector.n_features_}")

# Selected features
selected_features = X.columns[selector.support_]

# Plot number of features vs CV score
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()
```

---

## 5. Embedded Methods

Embedded methods perform feature selection during model training. They're more efficient than wrapper methods.

### L1 Regularization (Lasso)

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with cross-validation to find optimal alpha
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)

# Get selected features (non-zero coefficients)
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_scaled, y)

selected_features = X.columns[lasso.coef_ != 0]
print(f"Selected features: {selected_features}")
print(f"Number of features: {len(selected_features)}")
```

### L2 Regularization (Ridge)

```python
from sklearn.linear_model import Ridge, RidgeCV

# Ridge doesn't eliminate features, but shrinks coefficients
ridge_cv = RidgeCV(cv=5)
ridge_cv.fit(X_scaled, y)

# Features with larger coefficients are more important
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': ridge_cv.coef_
}).sort_values('coefficient', key=abs, ascending=False)
```

### Elastic Net

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Elastic Net combines L1 and L2
elastic_net_cv = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
elastic_net_cv.fit(X_scaled, y)

elastic_net = ElasticNet(alpha=elastic_net_cv.alpha_, 
                         l1_ratio=elastic_net_cv.l1_ratio_)
elastic_net.fit(X_scaled, y)

selected_features = X.columns[elastic_net.coef_ != 0]
```

### Tree-Based Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top k features
top_k = 10
selected_features = feature_importance.head(top_k)['feature'].tolist()

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X, y)

feature_importance_gb = pd.DataFrame({
    'feature': X.columns,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate permutation importance
perm_importance = permutation_importance(
    model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)
```

---

## 6. Dimensionality Reduction

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"Number of components: {pca.n_components_}")
print(f"Variance explained: {cumulative_variance[-1]:.2%}")

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Explained Variance')
plt.show()
```

### Linear Discriminant Analysis (LDA)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA for classification
lda = LinearDiscriminantAnalysis(n_components=None)
X_lda = lda.fit_transform(X, y)

# Explained variance ratio
explained_variance = lda.explained_variance_ratio_
```

### Independent Component Analysis (ICA)

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=10, random_state=42)
X_ica = ica.fit_transform(X_scaled)
```

---

## 7. Feature Selection Pipeline

### Combining Methods

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline: Scale → Select → Model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=20)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

### Custom Feature Selector

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method='correlation', threshold=0.8):
        self.method = method
        self.threshold = threshold
        self.selected_features = None
    
    def fit(self, X, y=None):
        if self.method == 'correlation':
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_remove = [col for col in upper_triangle.columns 
                        if any(upper_triangle[col] > self.threshold)]
            self.selected_features = [col for col in X.columns 
                                    if col not in to_remove]
        return self
    
    def transform(self, X):
        return X[self.selected_features]
```

---

## 8. Best Practices

### 1. Start with Domain Knowledge

- Remove obviously irrelevant features
- Consider feature engineering before selection
- Understand which features should matter

### 2. Use Multiple Methods

- Don't rely on single method
- Compare results from different methods
- Look for consensus across methods

### 3. Consider Model Type

- Tree-based models: Less sensitive to irrelevant features
- Linear models: Benefit more from feature selection
- Neural networks: Can handle many features but benefit from selection

### 4. Cross-Validation

- Always use cross-validation for wrapper methods
- Avoid data leakage in feature selection
- Fit selectors on training data only

```python
# CORRECT: Fit selector on training data
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# WRONG: Fitting on entire dataset
selector.fit(X_all, y_all)  # Data leakage!
```

### 5. Balance Performance and Interpretability

- More features → potentially better performance
- Fewer features → better interpretability
- Find the right balance for your use case

### 6. Monitor Feature Stability

- Check if selected features are stable across folds
- Unstable features might indicate overfitting
- Use ensemble of feature selectors

### 7. Document Feature Selection Process

- Record which features were selected
- Note the method and parameters used
- Track performance with/without selection

---

## 9. Evaluation Metrics

### Model Performance

```python
from sklearn.metrics import accuracy_score, roc_auc_score

# Compare performance with/without feature selection
model_all = RandomForestClassifier(random_state=42)
model_all.fit(X_train, y_train)
score_all = model_all.score(X_test, y_test)

model_selected = RandomForestClassifier(random_state=42)
model_selected.fit(X_train_selected, y_train)
score_selected = model_selected.score(X_test_selected, y_test)

print(f"All features: {score_all:.4f}")
print(f"Selected features: {score_selected:.4f}")
```

### Feature Stability

```python
# Check feature selection stability across CV folds
from sklearn.model_selection import cross_val_score

stability_scores = []
for fold in range(5):
    selector = SelectKBest(f_classif, k=10)
    X_fold_selected = selector.fit_transform(X_fold, y_fold)
    stability_scores.append(set(X.columns[selector.get_support()]))

# Calculate overlap
from itertools import combinations
overlaps = [len(a & b) for a, b in combinations(stability_scores, 2)]
average_overlap = np.mean(overlaps) / 10  # Normalize by k
```

---

## Key Takeaways

1. **Feature Selection Matters**: Can improve performance, reduce overfitting, and enhance interpretability
2. **Choose Method Wisely**: Filter (fast), Wrapper (accurate), Embedded (balanced)
3. **Avoid Data Leakage**: Always fit selectors on training data only
4. **Use Multiple Methods**: Compare results, look for consensus
5. **Consider Model Type**: Different models benefit differently from feature selection
6. **Domain Knowledge First**: Remove obviously irrelevant features before automated selection
7. **Balance Trade-offs**: Performance vs interpretability vs computational cost
8. **Document Process**: Track which features were selected and why

---

## Practice Problems

- **Kaggle**: House Prices, Credit Card Fraud Detection
- **Real-world**: Perform feature selection on a high-dimensional dataset
- **Challenge**: Compare multiple feature selection methods and evaluate their impact

---

## References

- Scikit-learn Feature Selection: https://scikit-learn.org/stable/modules/feature_selection.html
- "An Introduction to Statistical Learning" by James et al.
- "Feature Engineering and Selection" by Kuhn & Johnson
- MLxtend Documentation: https://rasbt.github.io/mlxtend/

