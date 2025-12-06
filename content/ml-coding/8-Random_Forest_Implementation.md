+++
title = "Random Forest"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Complete implementation of Random Forest from scratch using Python and NumPy. Covers bagging, feature sampling, decision tree ensemble, voting mechanisms, and out-of-bag evaluation."
+++

---

## Introduction

Random Forest is an ensemble learning method that combines multiple decision trees through bagging and random feature selection. It reduces overfitting compared to a single decision tree and provides robust predictions by averaging the outputs of many trees.

In this guide, we'll implement Random Forest from scratch using Python and NumPy, covering bagging, feature sampling, tree ensemble, and various voting mechanisms.

---

## Mathematical Foundation

### Bagging (Bootstrap Aggregating)

Random Forest uses bagging:
1. **Bootstrap Sampling**: Create multiple datasets by sampling with replacement
2. **Training**: Train a decision tree on each bootstrap sample
3. **Aggregation**: Combine predictions from all trees

### Random Feature Selection

At each split, only a random subset of features is considered:

```
m_features = √n_features  (for classification)
m_features = n_features / 3  (for regression)
```

This decorrelates the trees and improves generalization.

### Voting Mechanisms

**Classification:**
- **Majority Voting**: Each tree votes, most common class wins
- **Weighted Voting**: Trees weighted by their performance

**Regression:**
- **Averaging**: Mean of all tree predictions
- **Weighted Averaging**: Weighted mean based on tree performance

### Out-of-Bag (OOB) Score

For each sample, trees that didn't use it in training can evaluate it:

```
OOB_score = accuracy of predictions from OOB trees
```

This provides an unbiased estimate of performance without a separate validation set.

---

## Implementation 1: Random Forest Classifier

```python
import numpy as np
from collections import Counter
import random

# We'll use a simplified Decision Tree class (from previous implementation)
class DecisionTree:
    """Simplified Decision Tree for Random Forest."""
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
    
    def _entropy(self, y):
        """Calculate entropy."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain."""
        parent_entropy = self._entropy(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n == 0:
            return 0
        
        child_entropy = (n_left / n) * self._entropy(y_left) + (n_right / n) * self._entropy(y_right)
        return parent_entropy - child_entropy
    
    def _best_split(self, X, y, feature_indices):
        """Find best split using only selected features."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Build decision tree recursively."""
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]  # Return majority class
        
        # Random feature selection
        if self.max_features is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = random.sample(list(range(n_features)), min(self.max_features, n_features))
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Train the decision tree."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict a single sample."""
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Make predictions."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True, random_state=None):
        """
        Initialize Random Forest Classifier.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of each tree
        min_samples_split : int
            Minimum samples required to split a node
        max_features : int, float, or str
            Number of features to consider for best split
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - int: exact number
            - float: fraction of features
        bootstrap : bool
            Whether to use bootstrap sampling
        random_state : int
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees = []
        self.feature_indices_per_tree = []  # Store which features each tree used
        self.oob_scores = []
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample."""
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def _calculate_max_features(self, n_features):
        """Calculate number of features to use."""
        if isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        else:
            return n_features
    
    def fit(self, X, y):
        """
        Train Random Forest.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        n_samples, n_features = X.shape
        max_features = self._calculate_max_features(n_features)
        
        # Track which samples are OOB for each tree
        oob_indices = [set(range(n_samples)) for _ in range(self.n_estimators)]
        
        # Train each tree
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_boot, y_boot, boot_indices = self._bootstrap_sample(X, y)
                # Update OOB indices (samples not in bootstrap)
                oob_indices[i] = oob_indices[i] - set(boot_indices)
            else:
                X_boot, y_boot = X, y
            
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            
            self.trees.append(tree)
            
            # Calculate OOB score for this tree
            if len(oob_indices[i]) > 0:
                oob_X = X[list(oob_indices[i])]
                oob_y = y[list(oob_indices[i])]
                oob_pred = tree.predict(oob_X)
                oob_score = np.mean(oob_pred == oob_y)
                self.oob_scores.append(oob_score)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using majority voting.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels
        """
        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        predictions = []
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            prediction = Counter(votes).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
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
            Class probabilities of shape (m, n_classes)
        """
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        classes = np.unique(np.concatenate([tree.predict(X) for tree in self.trees]))
        n_classes = len(classes)
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            votes = all_predictions[:, i]
            for j, cls in enumerate(classes):
                probabilities[i, j] = np.sum(votes == cls) / self.n_estimators
        
        return probabilities
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_oob_score(self):
        """Get average out-of-bag score."""
        if len(self.oob_scores) > 0:
            return np.mean(self.oob_scores)
        return None
```

### Usage Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=3,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

# Evaluate
accuracy = rf.score(X_test, y_test)
oob_score = rf.get_oob_score()

print(f"Test Accuracy: {accuracy:.4f}")
print(f"OOB Score: {oob_score:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## Implementation 2: Random Forest Regressor

```python
class DecisionTreeRegressor:
    """Simplified Decision Tree Regressor for Random Forest."""
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
    
    def _mse(self, y):
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _best_split(self, X, y, feature_indices):
        """Find best split for regression."""
        best_mse_reduction = -1
        best_feature = None
        best_threshold = None
        
        parent_mse = self._mse(y)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                
                weighted_mse = (np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse) / len(y)
                mse_reduction = parent_mse - weighted_mse
                
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Build regression tree recursively."""
        n_samples, n_features = X.shape
        
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)  # Return mean value
        
        if self.max_features is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = random.sample(list(range(n_features)), min(self.max_features, n_features))
        
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        
        if best_feature is None:
            return np.mean(y)
        
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
        """Train the tree."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict a single sample."""
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Make predictions."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 max_features=1.0, bootstrap=True, random_state=None):
        """
        Initialize Random Forest Regressor.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples to split
        max_features : int, float, or str
            Number of features to consider
        bootstrap : bool
            Use bootstrap sampling
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample."""
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _calculate_max_features(self, n_features):
        """Calculate number of features."""
        if isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        else:
            return n_features
    
    def fit(self, X, y):
        """Train Random Forest."""
        n_features = X.shape[1]
        max_features = self._calculate_max_features(n_features)
        
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_boot, y_boot = self._bootstrap_sample(X, y)
            else:
                X_boot, y_boot = X, y
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Make predictions by averaging tree outputs."""
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(all_predictions, axis=0)
    
    def score(self, X, y):
        """Calculate R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
```

---

## Implementation 3: Feature Importance

```python
class RandomForestWithImportance(RandomForestClassifier):
    """Random Forest with feature importance calculation."""
    
    def calculate_feature_importance(self, X, y):
        """
        Calculate feature importance based on how often features are used in splits.
        
        Parameters:
        -----------
        X : numpy array
            Training features
        y : numpy array
            Training labels
        
        Returns:
        --------
        importance : numpy array
            Feature importance scores
        """
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        for tree in self.trees:
            self._calculate_tree_importance(tree.tree, importance)
        
        # Normalize
        importance = importance / np.sum(importance)
        return importance
    
    def _calculate_tree_importance(self, node, importance, depth=0):
        """Recursively calculate importance from tree."""
        if not isinstance(node, dict):
            return
        
        # Increment importance for this feature
        importance[node['feature']] += 1 / (depth + 1)  # Weight by depth
        
        # Recursively process children
        self._calculate_tree_importance(node['left'], importance, depth + 1)
        self._calculate_tree_importance(node['right'], importance, depth + 1)
```

---

## Complete Example: Comparison with Single Decision Tree

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Single Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = rf.score(X_test, y_test)

print(f"Single Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Improvement: {(rf_accuracy - dt_accuracy) * 100:.2f}%")
```

---

## Key Takeaways

1. **Bagging**: Combines multiple models trained on bootstrap samples
2. **Random Features**: Each split considers only a random subset of features
3. **Voting**: Classification uses majority voting, regression uses averaging
4. **OOB Score**: Provides unbiased performance estimate without validation set
5. **Reduced Overfitting**: Ensemble reduces variance compared to single tree
6. **Feature Importance**: Can identify important features
7. **Parallelizable**: Trees can be trained in parallel

---

## When to Use Random Forest

| Scenario | Suitable? | Notes |
|----------|-----------|-------|
| **High-dimensional data** | ✅ Yes | Handles many features well |
| **Non-linear relationships** | ✅ Yes | Can capture complex patterns |
| **Feature importance** | ✅ Yes | Provides feature importance scores |
| **Missing values** | ⚠️ Maybe | Can handle some missing values |
| **Interpretability** | ⚠️ Partial | Less interpretable than single tree |
| **Large datasets** | ✅ Yes | Efficient and scalable |
| **Real-time prediction** | ⚠️ Maybe | Slower than single tree |

---

## Interview Tips

When implementing Random Forest in interviews:

1. **Explain Bagging**: Bootstrap sampling and aggregation
2. **Random Features**: Why random feature selection decorrelates trees
3. **Voting Mechanisms**: Majority voting vs weighted voting
4. **OOB Score**: How out-of-bag evaluation works
5. **Feature Importance**: How to calculate and interpret
6. **Hyperparameters**: n_estimators, max_depth, max_features
7. **Advantages**: Reduced overfitting, handles non-linearity, feature importance
8. **Comparison**: vs single tree, vs boosting methods

---

## Time Complexity

- **Training**: O(n_estimators * m * n * log(m)) where m = samples, n = features
- **Prediction**: O(n_estimators * log(m)) per sample
- **Space Complexity**: O(n_estimators * m) for storing trees

---

## Advantages

1. **Reduced Overfitting**: Ensemble reduces variance
2. **Handles Non-linearity**: Can model complex relationships
3. **Feature Importance**: Provides feature importance scores
4. **Robust**: Less sensitive to outliers than single tree
5. **Parallelizable**: Trees can be trained in parallel
6. **No Feature Scaling**: Works with raw features

---

## Disadvantages

1. **Less Interpretable**: Harder to interpret than single tree
2. **Memory Intensive**: Stores many trees
3. **Slower Prediction**: Must query all trees
4. **Black Box**: Less transparent than linear models

---

## References

* Bootstrap aggregating (bagging)
* Ensemble learning methods
* Decision tree algorithms
* Feature importance and selection
* Out-of-bag evaluation

