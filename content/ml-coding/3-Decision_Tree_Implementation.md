+++
title = "Decision Tree"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Complete implementation of decision trees from scratch using Python and NumPy. Covers tree building, splitting criteria (information gain, Gini impurity), handling categorical and numerical features, pruning, and practical examples."
+++

---

## Introduction

Decision Trees are versatile machine learning algorithms that can be used for both classification and regression. They work by recursively splitting the data based on feature values to create a tree-like model of decisions.

In this guide, we'll implement decision trees from scratch using Python, covering tree building, splitting criteria, and handling different data types.

---

## Mathematical Foundation

### Entropy

Entropy measures the impurity or uncertainty in a dataset:

```
H(S) = -Σ pᵢ * log₂(pᵢ)
```

Where:
- `S` is the dataset
- `pᵢ` is the proportion of class `i` in the dataset
- Entropy ranges from 0 (pure) to 1 (maximum impurity for binary classification)

### Information Gain

Information gain measures the reduction in entropy after splitting:

```
IG(S, A) = H(S) - Σ(|Sᵥ|/|S|) * H(Sᵥ)
```

Where:
- `S` is the parent dataset
- `A` is the attribute/feature to split on
- `Sᵥ` is the subset of data for value `v` of attribute `A`
- `|S|` and `|Sᵥ|` are the sizes of the datasets

### Gini Impurity

Gini impurity is an alternative to entropy:

```
Gini(S) = 1 - Σ pᵢ²
```

Where `pᵢ` is the proportion of class `i`.

**Gini Gain** (similar to information gain):
```
Gini_Gain(S, A) = Gini(S) - Σ(|Sᵥ|/|S|) * Gini(Sᵥ)
```

### Comparison: Entropy vs Gini

- **Entropy**: More sensitive to changes in class distribution
- **Gini**: Computationally faster (no log calculation)
- Both work well in practice; Gini is slightly faster

---

## Implementation 1: Basic Decision Tree Classifier

### Node Class

```python
import numpy as np
from collections import Counter

class TreeNode:
    """Node in a decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        """
        Parameters:
        -----------
        feature_idx : int
            Index of feature to split on
        threshold : float
            Threshold value for splitting
        left : TreeNode
            Left child node
        right : TreeNode
            Right child node
        value : any
            Class value (for leaf nodes)
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes: predicted class
    
    def is_leaf(self):
        """Check if node is a leaf."""
        return self.value is not None
```

### Decision Tree Classifier

```python
class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', max_features=None):
        """
        Initialize Decision Tree Classifier.
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree (default: 10)
        min_samples_split : int
            Minimum samples required to split a node (default: 2)
        min_samples_leaf : int
            Minimum samples required in a leaf node (default: 1)
        criterion : str
            Splitting criterion: 'gini' or 'entropy' (default: 'gini')
        max_features : int or None
            Maximum features to consider for split (None = all features)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.root = None
    
    def _entropy(self, y):
        """
        Calculate entropy of a target vector.
        
        Parameters:
        -----------
        y : numpy array
            Target vector
        
        Returns:
        --------
        entropy : float
            Entropy value
        """
        if len(y) == 0:
            return 0
        
        # Count class frequencies
        counts = np.bincount(y)
        # Calculate proportions
        proportions = counts / len(y)
        # Remove zeros to avoid log(0)
        proportions = proportions[proportions > 0]
        # Calculate entropy
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy
    
    def _gini(self, y):
        """
        Calculate Gini impurity of a target vector.
        
        Parameters:
        -----------
        y : numpy array
            Target vector
        
        Returns:
        --------
        gini : float
            Gini impurity value
        """
        if len(y) == 0:
            return 0
        
        # Count class frequencies
        counts = np.bincount(y)
        # Calculate proportions
        proportions = counts / len(y)
        # Calculate Gini impurity
        gini = 1 - np.sum(proportions ** 2)
        return gini
    
    def _information_gain(self, y_parent, y_left, y_right):
        """
        Calculate information gain from a split.
        
        Parameters:
        -----------
        y_parent : numpy array
            Parent node target vector
        y_left : numpy array
            Left child target vector
        y_right : numpy array
            Right child target vector
        
        Returns:
        --------
        gain : float
            Information gain
        """
        # Calculate parent impurity
        if self.criterion == 'gini':
            parent_impurity = self._gini(y_parent)
            left_impurity = self._gini(y_left)
            right_impurity = self._gini(y_right)
        else:  # entropy
            parent_impurity = self._entropy(y_parent)
            left_impurity = self._entropy(y_left)
            right_impurity = self._entropy(y_right)
        
        # Calculate weighted average of children
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # Information gain
        gain = parent_impurity - weighted_impurity
        return gain
    
    def _best_split(self, X, y):
        """
        Find the best split for the data.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        
        Returns:
        --------
        best_feature : int
            Index of best feature to split on
        best_threshold : float
            Best threshold value
        best_gain : float
            Information gain of best split
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        # Select features to consider
        if self.max_features is None:
            features_to_consider = range(n_features)
        else:
            features_to_consider = np.random.choice(
                n_features, 
                size=min(self.max_features, n_features), 
                replace=False
            )
        
        # Try each feature
        for feature_idx in features_to_consider:
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try each threshold (midpoint between consecutive values)
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        depth : int
            Current depth of the tree
        
        Returns:
        --------
        node : TreeNode
            Root node of the subtree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        # 1. All samples belong to same class
        if n_classes == 1:
            return TreeNode(value=y[0])
        
        # 2. Maximum depth reached
        if depth >= self.max_depth:
            # Return majority class
            most_common = np.bincount(y).argmax()
            return TreeNode(value=most_common)
        
        # 3. Not enough samples to split
        if n_samples < self.min_samples_split:
            most_common = np.bincount(y).argmax()
            return TreeNode(value=most_common)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # If no gain (shouldn't happen, but safety check)
        if best_gain <= 0:
            most_common = np.bincount(y).argmax()
            return TreeNode(value=most_common)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples in leaves
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            most_common = np.bincount(y).argmax()
            return TreeNode(value=most_common)
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Create node
        return TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X, y):
        """
        Train the decision tree.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        """
        Predict class for a single sample.
        
        Parameters:
        -----------
        x : numpy array
            Single feature vector
        node : TreeNode
            Current node in the tree
        
        Returns:
        --------
        prediction : int
            Predicted class
        """
        # If leaf node, return its value
        if node.is_leaf():
            return node.value
        
        # Otherwise, traverse tree
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Make predictions for multiple samples.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted classes of shape (m,)
        """
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            True target values
        
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
```

### Usage Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    criterion='gini'
)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualize decision boundary
def plot_decision_boundary(X, y, model):
    """Plot decision boundary of the tree."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(X_grid).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Tree Decision Boundary (Test Acc: {test_accuracy:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_test, y_test, model)
```

---

## Implementation 2: Decision Tree Regressor

For regression, we use variance reduction instead of information gain:

```python
class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize Decision Tree Regressor.
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required in a leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def _variance(self, y):
        """
        Calculate variance of target values.
        
        Parameters:
        -----------
        y : numpy array
            Target vector
        
        Returns:
        --------
        variance : float
            Variance value
        """
        if len(y) == 0:
            return 0
        return np.var(y)
    
    def _variance_reduction(self, y_parent, y_left, y_right):
        """
        Calculate variance reduction from a split.
        
        Parameters:
        -----------
        y_parent : numpy array
            Parent node target vector
        y_left : numpy array
            Left child target vector
        y_right : numpy array
            Right child target vector
        
        Returns:
        --------
        reduction : float
            Variance reduction
        """
        parent_variance = self._variance(y_parent)
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        weighted_variance = (n_left / n) * self._variance(y_left) + \
                           (n_right / n) * self._variance(y_right)
        
        reduction = parent_variance - weighted_variance
        return reduction
    
    def _best_split(self, X, y):
        """Find the best split for regression."""
        best_reduction = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])
            
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                reduction = self._variance_reduction(y, y_left, y_right)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_reduction
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree for regression."""
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=np.mean(y))
        
        # Find best split
        best_feature, best_threshold, best_reduction = self._best_split(X, y)
        
        if best_reduction <= 0:
            return TreeNode(value=np.mean(y))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return TreeNode(value=np.mean(y))
        
        # Recursively build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X, y):
        """Train the decision tree regressor."""
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        """Predict value for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Make predictions for multiple samples."""
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return r2
```

---

## Implementation 3: Handling Categorical Features

```python
class DecisionTreeWithCategorical(DecisionTreeClassifier):
    """
    Extended Decision Tree that handles both numerical and categorical features.
    """
    def __init__(self, categorical_features=None, **kwargs):
        """
        Parameters:
        -----------
        categorical_features : list or None
            List of indices of categorical features (None = all numerical)
        **kwargs : dict
            Arguments passed to parent DecisionTreeClassifier
        """
        super().__init__(**kwargs)
        self.categorical_features = categorical_features if categorical_features else []
    
    def _best_split_categorical(self, X, y, feature_idx):
        """
        Find best split for a categorical feature.
        
        For categorical features, we try all possible subsets.
        """
        unique_values = np.unique(X[:, feature_idx])
        best_gain = -1
        best_split = None
        
        # Try all possible binary splits
        # For each value, try splitting: value vs all others
        for value in unique_values:
            left_mask = X[:, feature_idx] == value
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            gain = self._information_gain(y, y_left, y_right)
            
            if gain > best_gain:
                best_gain = gain
                best_split = value
        
        return best_split, best_gain
    
    def _best_split(self, X, y):
        """Override to handle categorical features."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_is_categorical = False
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            if feature_idx in self.categorical_features:
                # Handle categorical feature
                split_value, gain = self._best_split_categorical(X, y, feature_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = split_value
                    best_is_categorical = True
            else:
                # Handle numerical feature (original logic)
                feature_values = np.unique(X[:, feature_idx])
                
                for i in range(len(feature_values) - 1):
                    threshold = (feature_values[i] + feature_values[i + 1]) / 2
                    
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    
                    gain = self._information_gain(y, y_left, y_right)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
                        best_is_categorical = False
        
        # Store whether this is a categorical split
        self._last_split_categorical = best_is_categorical
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Override to handle categorical splits."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if n_classes == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            most_common = np.bincount(y).argmax()
            return TreeNode(value=most_common)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain <= 0:
            most_common = np.bincount(y).argmax()
            return TreeNode(value=most_common)
        
        # Split data (handle categorical vs numerical)
        is_categorical = getattr(self, '_last_split_categorical', False)
        
        if is_categorical:
            left_mask = X[:, best_feature] == best_threshold
        else:
            left_mask = X[:, best_feature] <= best_threshold
        
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            most_common = np.bincount(y).argmax()
            return TreeNode(value=most_common)
        
        # Recursively build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _predict_sample(self, x, node):
        """Override to handle categorical splits."""
        if node.is_leaf():
            return node.value
        
        # Check if this was a categorical split
        # (We can't easily determine this, so we'll use a simple heuristic)
        # For now, assume numerical split
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
```

---

## Key Takeaways

1. **Splitting Criteria**: 
   - Classification: Information Gain (Entropy) or Gini Impurity
   - Regression: Variance Reduction

2. **Stopping Conditions**:
   - Maximum depth reached
   - Minimum samples to split
   - All samples in same class (classification)
   - No information gain

3. **Tree Building**: Recursive algorithm that splits data and builds subtrees

4. **Prediction**: Traverse tree from root to leaf based on feature values

5. **Overfitting**: Control with `max_depth`, `min_samples_split`, `min_samples_leaf`

---

## Interview Tips

1. **Explain Entropy/Gini**: Understand the mathematical formulation
2. **Information Gain**: Show how it measures split quality
3. **Recursive Building**: Explain the tree construction algorithm
4. **Stopping Conditions**: Know when to stop splitting
5. **Handling Features**: Numerical vs categorical splitting strategies
6. **Overfitting**: How to prevent and control it

---

## Time Complexity

- **Training**: O(m * n * log(m)) where m = samples, n = features
- **Prediction**: O(log(m)) per sample (tree depth)
- **Space**: O(m) for storing the tree

---

## Advantages and Disadvantages

**Advantages**:
- Easy to interpret and visualize
- Handles non-linear relationships
- No feature scaling needed
- Handles both numerical and categorical features

**Disadvantages**:
- Prone to overfitting
- Unstable (small data changes can change tree structure)
- Biased towards features with more levels
- Doesn't work well with XOR-like problems

---

## References

- Entropy and information gain
- Gini impurity calculation
- Tree building algorithms
- Pruning techniques
- Handling categorical features

