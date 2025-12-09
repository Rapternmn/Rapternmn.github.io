+++
title = "Gradient Boosting"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 9
description = "Complete implementation of Gradient Boosting from scratch using Python and NumPy. Covers sequential tree building, gradient computation, shrinkage, regularization, and XGBoost-style optimizations."
+++

---

## Introduction

Gradient Boosting is a powerful ensemble learning technique that builds models sequentially, where each new model corrects the errors of the previous ones. Unlike Random Forest (which uses bagging), Gradient Boosting uses boosting to create a strong learner from weak learners.

In this guide, we'll implement Gradient Boosting from scratch using Python and NumPy, covering sequential tree building, gradient computation, and various regularization techniques.

---

## Mathematical Foundation

### Boosting Concept

Gradient Boosting builds models sequentially:

```
F₀(x) = argmin_γ Σ L(yᵢ, γ)  # Initial model (usually mean/median)
```

For m = 1 to M:
```
rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]  # Negative gradient (pseudo-residuals)
hₘ(x) = fit_tree(rᵢₘ)  # Fit tree to residuals
γₘ = argmin_γ Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ * hₘ(xᵢ))  # Optimal step size
Fₘ(x) = Fₘ₋₁(x) + ν * γₘ * hₘ(x)  # Update model with shrinkage
```

Where:
- `L` is the loss function
- `ν` (nu) is the learning rate (shrinkage)
- `hₘ` is the m-th weak learner (decision tree)

### Loss Functions

**For Regression (MSE):**
```
L(y, ŷ) = (y - ŷ)²
∂L/∂ŷ = -2(y - ŷ)  # Negative gradient = 2(y - ŷ)
```

**For Classification (Log Loss):**
```
L(y, ŷ) = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
∂L/∂ŷ = -(y - ŷ) / [ŷ(1-ŷ)]  # For probability output
```

### Regularization

**Shrinkage (Learning Rate):**
```
Fₘ(x) = Fₘ₋₁(x) + ν * hₘ(x)  # ν ∈ (0, 1]
```

**Subsampling:**
Train each tree on a random subset of data (stochastic gradient boosting).

**Tree Constraints:**
- Max depth
- Min samples per leaf
- Min samples to split

---

## Implementation 1: Gradient Boosting Regressor

```python
import numpy as np
from collections import Counter

# Simplified Decision Tree Regressor
class DecisionTreeRegressor:
    """Decision Tree for Gradient Boosting."""
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _mse(self, y):
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _best_split(self, X, y):
        """Find best split."""
        best_mse_reduction = -1
        best_feature = None
        best_threshold = None
        
        parent_mse = self._mse(y)
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                
                weighted_mse = (np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse) / n_samples
                mse_reduction = parent_mse - weighted_mse
                
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Build tree recursively."""
        n_samples = len(y)
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return np.mean(y)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return np.mean(y)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Train tree."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict single sample."""
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Make predictions."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 random_state=None):
        """
        Initialize Gradient Boosting Regressor.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages
        learning_rate : float
            Shrinkage parameter (ν)
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples to split a node
        min_samples_leaf : int
            Minimum samples in a leaf
        subsample : float
            Fraction of samples to use for each tree (stochastic boosting)
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        
        self.trees = []
        self.initial_prediction = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _mse_loss_gradient(self, y_true, y_pred):
        """Compute negative gradient for MSE loss."""
        return y_true - y_pred
    
    def fit(self, X, y):
        """
        Train Gradient Boosting model.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        n_samples = len(y)
        
        # Initialize with mean (for regression)
        self.initial_prediction = np.mean(y)
        y_pred = np.full(n_samples, self.initial_prediction)
        
        # Train each tree sequentially
        for i in range(self.n_estimators):
            # Compute negative gradients (residuals)
            residuals = self._mse_loss_gradient(y, y_pred)
            
            # Subsample for stochastic gradient boosting
            if self.subsample < 1.0:
                n_subset = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, size=n_subset, replace=False)
                X_subset = X[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X
                residuals_subset = residuals
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, residuals_subset)
            
            # Predict residuals
            residuals_pred = tree.predict(X)
            
            # Update predictions
            y_pred += self.learning_rate * residuals_pred
            
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        predictions : numpy array
            Predicted values
        """
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def score(self, X, y):
        """Calculate R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
```

### Usage Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gb.fit(X_train, y_train)

# Make predictions
y_pred = gb.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = gb.score(X_test, y_test)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

---

## Implementation 2: Gradient Boosting Classifier

```python
class DecisionTreeClassifier:
    """Decision Tree for Gradient Boosting Classification."""
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _entropy(self, y):
        """Calculate entropy."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y.astype(int))
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _best_split(self, X, y):
        """Find best split."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_entropy = self._entropy(y)
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_entropy = self._entropy(y[left_mask])
                right_entropy = self._entropy(y[right_mask])
                
                weighted_entropy = (np.sum(left_mask) * left_entropy + np.sum(right_mask) * right_entropy) / n_samples
                gain = parent_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Build tree recursively."""
        n_samples = len(y)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return Counter(y.astype(int)).most_common(1)[0][0]
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return Counter(y.astype(int)).most_common(1)[0][0]
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Train tree."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict single sample."""
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Make predictions."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 random_state=None):
        """
        Initialize Gradient Boosting Classifier.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages
        learning_rate : float
            Shrinkage parameter
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples to split
        min_samples_leaf : int
            Minimum samples in leaf
        subsample : float
            Fraction of samples for each tree
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        
        self.trees = []
        self.initial_prediction = None
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _log_loss_gradient(self, y_true, y_pred_proba):
        """Compute negative gradient for log loss."""
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        return y_true - y_pred_proba
    
    def fit(self, X, y):
        """
        Train Gradient Boosting Classifier.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Binary class labels (0 or 1)
        """
        self.classes_ = np.unique(y)
        n_samples = len(y)
        
        # Initialize with log-odds
        positive_class_ratio = np.mean(y == 1)
        self.initial_prediction = np.log(positive_class_ratio / (1 - positive_class_ratio + 1e-15))
        
        # Convert to probability
        y_pred_proba = np.full(n_samples, self._sigmoid(self.initial_prediction))
        
        # Train trees
        for i in range(self.n_estimators):
            # Compute negative gradients
            residuals = self._log_loss_gradient(y, y_pred_proba)
            
            # Subsample
            if self.subsample < 1.0:
                n_subset = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, size=n_subset, replace=False)
                X_subset = X[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X
                residuals_subset = residuals
            
            # Fit tree to residuals
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, residuals_subset)
            
            # Predict residuals
            residuals_pred = tree.predict(X)
            
            # Update predictions (in log-odds space)
            log_odds = np.log(y_pred_proba / (1 - y_pred_proba + 1e-15) + 1e-15)
            log_odds += self.learning_rate * residuals_pred
            y_pred_proba = self._sigmoid(log_odds)
            
            self.trees.append(tree)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        probabilities : numpy array
            Class probabilities of shape (m, 2)
        """
        n_samples = X.shape[0]
        log_odds = np.full(n_samples, self.initial_prediction)
        
        for tree in self.trees:
            log_odds += self.learning_rate * tree.predict(X)
        
        prob_positive = self._sigmoid(log_odds)
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## Implementation 3: Early Stopping

```python
class GradientBoostingWithEarlyStopping(GradientBoostingRegressor):
    """Gradient Boosting with early stopping."""
    
    def fit(self, X, y, X_val=None, y_val=None, n_iter_no_change=10, tol=1e-4):
        """
        Train with early stopping.
        
        Parameters:
        -----------
        X : numpy array
            Training features
        y : numpy array
            Training labels
        X_val : numpy array
            Validation features
        y_val : numpy array
            Validation labels
        n_iter_no_change : int
            Number of iterations with no improvement before stopping
        tol : float
            Tolerance for improvement
        """
        n_samples = len(y)
        self.initial_prediction = np.mean(y)
        y_pred = np.full(n_samples, self.initial_prediction)
        
        best_val_score = float('inf')
        no_change_count = 0
        
        for i in range(self.n_estimators):
            residuals = self._mse_loss_gradient(y, y_pred)
            
            if self.subsample < 1.0:
                n_subset = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, size=n_subset, replace=False)
                X_subset = X[indices]
                residuals_subset = residuals[indices]
            else:
                X_subset = X
                residuals_subset = residuals
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, residuals_subset)
            residuals_pred = tree.predict(X)
            y_pred += self.learning_rate * residuals_pred
            
            self.trees.append(tree)
            
            # Early stopping check
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_score = np.mean((y_val - val_pred) ** 2)
                
                if val_score < best_val_score - tol:
                    best_val_score = val_score
                    no_change_count = 0
                else:
                    no_change_count += 1
                
                if no_change_count >= n_iter_no_change:
                    print(f"Early stopping at iteration {i+1}")
                    break
        
        return self
```

---

## Complete Example: Comparison with Random Forest

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_accuracy = gb.score(X_test, y_test)

# Random Forest (for comparison)
from content.ml_coding.random_forest import RandomForestClassifier as RF
rf = RF(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = rf.score(X_test, y_test)

print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
```

---

## Key Takeaways

1. **Sequential Learning**: Each tree corrects errors of previous trees
2. **Gradient Descent**: Uses gradients of loss function to guide learning
3. **Shrinkage**: Learning rate prevents overfitting
4. **Regularization**: Tree depth, subsampling, min samples control complexity
5. **Loss Functions**: Different losses for regression vs classification
6. **Early Stopping**: Can stop training when validation performance plateaus
7. **Stochastic Boosting**: Subsampling adds randomness and reduces overfitting

---

## When to Use Gradient Boosting

| Scenario | Suitable? | Notes |
|----------|-----------|-------|
| **High accuracy needed** | ✅ Yes | Often achieves best performance |
| **Non-linear relationships** | ✅ Yes | Captures complex patterns |
| **Feature interactions** | ✅ Yes | Trees can model interactions |
| **Large datasets** | ⚠️ Maybe | Can be slow, consider XGBoost/LightGBM |
| **Interpretability** | ❌ No | Less interpretable than single tree |
| **Real-time prediction** | ⚠️ Maybe | Slower than Random Forest |
| **Overfitting risk** | ⚠️ Careful | Needs careful tuning |

---

## Interview Tips

When implementing Gradient Boosting in interviews:

1. **Explain Boosting**: Sequential error correction
2. **Gradient Computation**: How negative gradients become targets
3. **Shrinkage**: Why learning rate prevents overfitting
4. **Loss Functions**: MSE for regression, log loss for classification
5. **Regularization**: Tree constraints, subsampling
6. **Comparison**: vs Random Forest (bagging vs boosting)
7. **XGBoost**: Mention optimizations (approximate algorithms, sparsity awareness)

---

## Time Complexity

- **Training**: O(n_estimators * m * n * log(m)) where m = samples, n = features
- **Prediction**: O(n_estimators * log(m)) per sample
- **Space Complexity**: O(n_estimators * m) for storing trees

---

## Advantages

1. **High Accuracy**: Often achieves state-of-the-art performance
2. **Flexible**: Works for regression and classification
3. **Feature Interactions**: Automatically captures interactions
4. **Handles Non-linearity**: Can model complex relationships
5. **No Feature Scaling**: Works with raw features

---

## Disadvantages

1. **Slow Training**: Sequential nature makes it slower
2. **Hyperparameter Sensitive**: Many hyperparameters to tune
3. **Overfitting Risk**: Can overfit without proper regularization
4. **Less Interpretable**: Harder to interpret than single tree
5. **Memory Intensive**: Stores many trees

---

## References

* Gradient boosting and boosting algorithms
* Loss functions and gradient computation
* Regularization in ensemble methods
* XGBoost and LightGBM optimizations
* Sequential model building

