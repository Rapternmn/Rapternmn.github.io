+++
title = "Model Selection & Hyperparameter Tuning"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Comprehensive guide to model selection and hyperparameter tuning. Covers cross-validation, grid search, random search, Bayesian optimization, model evaluation, ensemble methods, and best practices."
+++

---

## Introduction

**Model Selection** is choosing the best algorithm for your problem, while **Hyperparameter Tuning** is finding the optimal hyperparameters for a chosen algorithm. Both are crucial for building high-performing machine learning models.

**Key Principle**: A well-tuned simple model often outperforms a poorly-tuned complex model.

---

## 1. Model Selection Overview

### Why Model Selection?

- **Different algorithms** work better for different problems
- **No free lunch theorem**: No single algorithm is best for all problems
- **Trade-offs**: Accuracy vs interpretability vs speed vs complexity

### Model Selection Process

1. **Define Problem**: Classification, regression, clustering, etc.
2. **Select Candidates**: Choose algorithms to try
3. **Train & Evaluate**: Use cross-validation
4. **Compare Performance**: Use appropriate metrics
5. **Select Best Model**: Consider multiple factors

---

## 2. Cross-Validation

### Why Cross-Validation?

- **Prevents Overfitting**: Evaluate on unseen data
- **Better Estimate**: More reliable performance estimate
- **Model Selection**: Compare models fairly

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# K-fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-fold (maintains class distribution)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

### Leave-One-Out Cross-Validation (LOOCV)

```python
from sklearn.model_selection import LeaveOneOut

# LOOCV (one sample per fold)
loocv = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loocv, scoring='accuracy')
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series CV (respects temporal order)
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

### Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Outer CV for model evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner CV for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=inner_cv, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model on outer test set
    score = grid_search.score(X_test, y_test)
    outer_scores.append(score)

print(f"Mean Nested CV Score: {np.mean(outer_scores):.4f}")
```

---

## 3. Hyperparameter Tuning

### What are Hyperparameters?

**Hyperparameters** are parameters set before training (not learned during training):
- Learning rate, number of trees, regularization strength, etc.
- Different from **parameters** (weights, biases learned during training)

### Manual Tuning

```python
# Try different values manually
for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10, 20]:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        print(f"n_estimators={n_estimators}, max_depth={max_depth}: {score:.4f}")
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Random search
random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

### Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Define search space
space = [
    Integer(50, 300, name='n_estimators'),
    Integer(5, 30, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Integer(1, 10, name='min_samples_leaf')
]

# Objective function
@use_named_args(space)
def objective(**params):
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return -score  # Minimize negative score

# Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=50,
    random_state=42
)

# Best parameters
best_params = {
    'n_estimators': result.x[0],
    'max_depth': result.x[1],
    'min_samples_split': result.x[2],
    'min_samples_leaf': result.x[3]
}
print(f"Best parameters: {best_params}")
print(f"Best score: {-result.fun:.4f}")
```

### Optuna (Advanced Bayesian Optimization)

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Evaluate
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters
print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")

# Visualization
import optuna.visualization as vis
vis.plot_optimization_history(study)
vis.plot_param_importances(study)
```

---

## 4. Model Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# Metrics
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Custom Scoring Functions

```python
from sklearn.metrics import make_scorer

# Custom scoring function
def custom_score(y_true, y_pred):
    # Your custom metric
    return your_calculation(y_true, y_pred)

custom_scorer = make_scorer(custom_score, greater_is_better=True)

# Use in grid search
grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring=custom_scorer
)
```

---

## 5. Model Comparison

### Compare Multiple Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

# Compare models
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Visualize comparison
results_df = pd.DataFrame(results).T
results_df.plot(kind='barh', y='mean', xerr='std', figsize=(10, 6))
plt.xlabel('Accuracy')
plt.title('Model Comparison')
plt.show()
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()

plot_learning_curve(model, X_train, y_train)
```

---

## 6. Ensemble Methods

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

# Combine multiple models
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ],
    voting='soft'  # or 'hard'
)

voting_clf.fit(X_train, y_train)
score = voting_clf.score(X_test, y_test)
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier

# Stacking with meta-learner
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

stacking_clf.fit(X_train, y_train)
score = stacking_clf.score(X_test, y_test)
```

### Bagging

```python
from sklearn.ensemble import BaggingClassifier

# Bagging
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

bagging_clf.fit(X_train, y_train)
score = bagging_clf.score(X_test, y_test)
```

---

## 7. Best Practices

### 1. Train-Validation-Test Split

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Use: Train for training, Val for tuning, Test for final evaluation
```

### 2. Avoid Data Leakage

- **Fit preprocessors on training data only**
- **Don't use test set for hyperparameter tuning**
- **Use nested CV for unbiased evaluation**

### 3. Use Appropriate Metrics

- **Classification**: Accuracy, Precision, Recall, F1, ROC AUC
- **Imbalanced Data**: F1, ROC AUC, Precision-Recall AUC
- **Regression**: MSE, RMSE, MAE, R²

### 4. Consider Computational Cost

- **Grid Search**: Exhaustive but slow
- **Random Search**: Faster, often as good
- **Bayesian Optimization**: Best balance

### 5. Early Stopping

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import validation_curve

# Use early stopping
model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
model.fit(X_train, y_train)
```

### 6. Feature Importance Analysis

```python
# Analyze feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

### 7. Save Best Model

```python
import joblib

# Save best model
joblib.dump(best_model, 'best_model.pkl')

# Save grid search results
joblib.dump(grid_search, 'grid_search.pkl')

# Load later
best_model = joblib.load('best_model.pkl')
```

---

## Key Takeaways

1. **Cross-Validation is Essential**: Use CV for reliable performance estimates
2. **Hyperparameter Tuning Matters**: Can significantly improve model performance
3. **Choose Tuning Method Wisely**: Grid (exhaustive), Random (fast), Bayesian (optimal)
4. **Avoid Data Leakage**: Never use test set for tuning
5. **Use Appropriate Metrics**: Choose metrics that match your problem
6. **Compare Multiple Models**: Don't assume one algorithm is best
7. **Consider Ensemble Methods**: Often improve performance
8. **Document Everything**: Track experiments, parameters, and results

---

## Practice Problems

- **Kaggle**: House Prices, Titanic, Credit Card Fraud Detection
- **Real-world**: Perform complete model selection and tuning on a dataset
- **Challenge**: Build an automated model selection and tuning pipeline

---

## References

- Scikit-learn Model Selection: https://scikit-learn.org/stable/model_selection.html
- "An Introduction to Statistical Learning" by James et al.
- "The Elements of Statistical Learning" by Hastie et al.
- Optuna Documentation: https://optuna.org/
- Scikit-optimize Documentation: https://scikit-optimize.github.io/

