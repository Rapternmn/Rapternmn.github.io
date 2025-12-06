+++
title = "K-Nearest Neighbors (KNN) Implementation from Scratch"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Complete implementation of K-Nearest Neighbors algorithm from scratch using Python and NumPy. Covers distance metrics, k selection, weighted voting, classification and regression, and practical examples."
+++

---

## Introduction

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that makes predictions based on the k closest training examples. It's a non-parametric, lazy learning algorithm - it doesn't build an explicit model but stores all training data and makes predictions at query time.

In this guide, we'll implement KNN from scratch using Python and NumPy, covering classification, regression, various distance metrics, and optimization techniques.

---

## Mathematical Foundation

### Basic KNN Algorithm

For a query point `x`, KNN:
1. Finds the k nearest neighbors in the training set
2. For classification: Returns the majority class among k neighbors
3. For regression: Returns the average value among k neighbors

### Distance Metrics

#### Euclidean Distance

```
d(x, y) = √(Σ(xᵢ - yᵢ)²)
```

Most common for continuous features.

#### Manhattan Distance (L1)

```
d(x, y) = Σ|xᵢ - yᵢ|
```

Less sensitive to outliers than Euclidean.

#### Minkowski Distance

```
d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```

Generalization:
- p=1: Manhattan
- p=2: Euclidean
- p=∞: Chebyshev

#### Cosine Similarity

```
cos(θ) = (x · y) / (||x|| * ||y||)
```

Useful for high-dimensional sparse data.

#### Hamming Distance

```
d(x, y) = Σ(xᵢ ≠ yᵢ)
```

For categorical/binary features.

### Weighted Voting

Instead of simple majority voting, we can weight neighbors by distance:

```
weight = 1 / (distance + ε)
```

Closer neighbors have higher weight.

### Classification Rule

```
ŷ = argmax_c Σᵢ∈k_neighbors wᵢ * I(yᵢ = c)
```

Where `wᵢ` is the weight of neighbor i, and `I()` is the indicator function.

### Regression Rule

```
ŷ = (Σᵢ∈k_neighbors wᵢ * yᵢ) / (Σᵢ∈k_neighbors wᵢ)
```

Weighted average of neighbor values.

---

## Implementation 1: Basic KNN Classifier

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """
        Initialize KNN Classifier.
        
        Parameters:
        -----------
        k : int
            Number of neighbors to consider
        distance_metric : str
            Distance metric: 'euclidean', 'manhattan', 'minkowski', 'cosine'
        weights : str
            Weight function: 'uniform' (all neighbors equal) or 'distance' (weight by inverse distance)
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data (lazy learning - no model building).
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points.
        
        Parameters:
        -----------
        x1, x2 : numpy arrays
            Feature vectors
        
        Returns:
        --------
        distance : float
            Distance between x1 and x2
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        
        elif self.distance_metric == 'minkowski':
            p = 2  # Default p=2 (Euclidean)
            return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)
        
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm1 = np.linalg.norm(x1)
            norm2 = np.linalg.norm(x2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return 1 - (dot_product / (norm1 * norm2))  # Return distance (1 - similarity)
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _get_k_neighbors(self, x):
        """
        Find k nearest neighbors for a query point.
        
        Parameters:
        -----------
        x : numpy array
            Query point
        
        Returns:
        --------
        distances : numpy array
            Distances to k nearest neighbors
        indices : numpy array
            Indices of k nearest neighbors in training set
        """
        distances = []
        
        # Calculate distance to all training points
        for i in range(len(self.X_train)):
            dist = self._calculate_distance(x, self.X_train[i])
            distances.append((dist, i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        distances_array = np.array([d[0] for d in k_nearest])
        indices_array = np.array([d[1] for d in k_nearest])
        
        return distances_array, indices_array
    
    def predict(self, X):
        """
        Predict class labels for query points.
        
        Parameters:
        -----------
        X : numpy array
            Query points of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels of shape (m,)
        """
        predictions = []
        
        for x in X:
            # Get k nearest neighbors
            distances, indices = self._get_k_neighbors(x)
            
            # Get labels of k nearest neighbors
            k_neighbor_labels = self.y_train[indices]
            
            if self.weights == 'uniform':
                # Simple majority voting
                most_common = Counter(k_neighbor_labels).most_common(1)[0][0]
                predictions.append(most_common)
            
            elif self.weights == 'distance':
                # Weighted voting (inverse distance)
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                weights = 1 / (distances + epsilon)
                
                # Count weighted votes for each class
                class_votes = {}
                for label, weight in zip(k_neighbor_labels, weights):
                    if label not in class_votes:
                        class_votes[label] = 0
                    class_votes[label] += weight
                
                # Get class with highest weighted votes
                most_common = max(class_votes, key=class_votes.get)
                predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for query points.
        
        Parameters:
        -----------
        X : numpy array
            Query points of shape (m, n)
        
        Returns:
        --------
        probabilities : numpy array
            Class probabilities of shape (m, n_classes)
        """
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        probabilities = []
        
        for x in X:
            # Get k nearest neighbors
            distances, indices = self._get_k_neighbors(x)
            k_neighbor_labels = self.y_train[indices]
            
            if self.weights == 'uniform':
                # Count occurrences of each class
                class_counts = np.zeros(n_classes)
                for label in k_neighbor_labels:
                    class_idx = np.where(classes == label)[0][0]
                    class_counts[class_idx] += 1
                # Normalize to probabilities
                probabilities.append(class_counts / self.k)
            
            elif self.weights == 'distance':
                # Weighted probabilities
                epsilon = 1e-10
                weights = 1 / (distances + epsilon)
                
                class_weights = np.zeros(n_classes)
                for label, weight in zip(k_neighbor_labels, weights):
                    class_idx = np.where(classes == label)[0][0]
                    class_weights[class_idx] += weight
                
                # Normalize
                total_weight = np.sum(class_weights)
                if total_weight > 0:
                    probabilities.append(class_weights / total_weight)
                else:
                    probabilities.append(np.ones(n_classes) / n_classes)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            True class labels
        
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

### Usage Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model (KNN just stores the data)
knn_uniform = KNNClassifier(k=5, distance_metric='euclidean', weights='uniform')
knn_uniform.fit(X_train, y_train)

# Make predictions
y_pred = knn_uniform.predict(X_test)
y_proba = knn_uniform.predict_proba(X_test)

# Evaluate
accuracy = knn_uniform.score(X_test, y_test)
print(f"Accuracy (uniform weights): {accuracy:.4f}")
print(f"\nPredicted probabilities for first 5 samples:")
print(y_proba[:5])

# Test with distance-weighted KNN
knn_weighted = KNNClassifier(k=5, distance_metric='euclidean', weights='distance')
knn_weighted.fit(X_train, y_train)
accuracy_weighted = knn_weighted.score(X_test, y_test)
print(f"\nAccuracy (distance-weighted): {accuracy_weighted:.4f}")
```

---

## Implementation 2: KNN Regressor

```python
class KNNRegressor:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """
        Initialize KNN Regressor.
        
        Parameters:
        -----------
        k : int
            Number of neighbors to consider
        distance_metric : str
            Distance metric: 'euclidean', 'manhattan', 'minkowski', 'cosine'
        weights : str
            Weight function: 'uniform' or 'distance'
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def _calculate_distance(self, x1, x2):
        """Calculate distance between two points (same as classifier)."""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            p = 2
            return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm1 = np.linalg.norm(x1)
            norm2 = np.linalg.norm(x2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return 1 - (dot_product / (norm1 * norm2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _get_k_neighbors(self, x):
        """Find k nearest neighbors (same as classifier)."""
        distances = []
        for i in range(len(self.X_train)):
            dist = self._calculate_distance(x, self.X_train[i])
            distances.append((dist, i))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        distances_array = np.array([d[0] for d in k_nearest])
        indices_array = np.array([d[1] for d in k_nearest])
        
        return distances_array, indices_array
    
    def predict(self, X):
        """
        Predict target values for query points.
        
        Parameters:
        -----------
        X : numpy array
            Query points of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted values of shape (m,)
        """
        predictions = []
        
        for x in X:
            # Get k nearest neighbors
            distances, indices = self._get_k_neighbors(x)
            
            # Get target values of k nearest neighbors
            k_neighbor_values = self.y_train[indices]
            
            if self.weights == 'uniform':
                # Simple average
                prediction = np.mean(k_neighbor_values)
            
            elif self.weights == 'distance':
                # Weighted average (inverse distance)
                epsilon = 1e-10
                weights = 1 / (distances + epsilon)
                
                # Weighted average
                prediction = np.average(k_neighbor_values, weights=weights)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calculate R² score.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            True target values
        
        Returns:
        --------
        r2_score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return r2
```

### Usage Example (Regression)

```python
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X, y = make_regression(
    n_samples=500,
    n_features=3,
    n_informative=3,
    noise=10,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
knn_reg = KNNRegressor(k=5, distance_metric='euclidean', weights='distance')
knn_reg.fit(X_train, y_train)

# Make predictions
y_pred = knn_reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = knn_reg.score(X_test, y_test)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

---

## Implementation 3: Optimized KNN (Vectorized)

The basic implementation calculates distances one by one. Here's a vectorized version for better performance:

```python
class KNNClassifierOptimized:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """Initialize optimized KNN Classifier."""
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data."""
        self.X_train = X
        self.y_train = y
        return self
    
    def _calculate_distances_vectorized(self, X):
        """
        Calculate distances from all query points to all training points (vectorized).
        
        Parameters:
        -----------
        X : numpy array
            Query points of shape (m, n)
        
        Returns:
        --------
        distances : numpy array
            Distance matrix of shape (m, m_train)
        """
        if self.distance_metric == 'euclidean':
            # Using broadcasting: (m, 1, n) - (1, m_train, n) = (m, m_train, n)
            # Then sum over last dimension
            diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        elif self.distance_metric == 'manhattan':
            diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
            distances = np.sum(np.abs(diff), axis=2)
        
        elif self.distance_metric == 'cosine':
            # Cosine distance: 1 - cosine similarity
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            X_train_norm = self.X_train / (np.linalg.norm(self.X_train, axis=1, keepdims=True) + 1e-10)
            # Dot product: (m, n) @ (n, m_train) = (m, m_train)
            cosine_sim = X_norm @ X_train_norm.T
            distances = 1 - cosine_sim
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def predict(self, X):
        """
        Predict class labels (vectorized version).
        
        Parameters:
        -----------
        X : numpy array
            Query points of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels of shape (m,)
        """
        # Calculate all distances at once
        distances = self._calculate_distances_vectorized(X)
        
        # Get k nearest neighbors for each query point
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        k_distances = np.take_along_axis(distances, k_indices, axis=1)
        k_labels = self.y_train[k_indices]
        
        predictions = []
        classes = np.unique(self.y_train)
        
        for i in range(len(X)):
            if self.weights == 'uniform':
                # Majority voting
                label_counts = np.bincount(
                    np.searchsorted(classes, k_labels[i]),
                    minlength=len(classes)
                )
                predictions.append(classes[np.argmax(label_counts)])
            
            elif self.weights == 'distance':
                # Weighted voting
                epsilon = 1e-10
                weights = 1 / (k_distances[i] + epsilon)
                
                class_weights = np.zeros(len(classes))
                for label, weight in zip(k_labels[i], weights):
                    class_idx = np.where(classes == label)[0][0]
                    class_weights[class_idx] += weight
                
                predictions.append(classes[np.argmax(class_weights)])
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## Choosing the Optimal k

### Cross-Validation for k Selection

```python
from sklearn.model_selection import cross_val_score

def find_optimal_k(X_train, y_train, k_range=range(1, 21), cv=5):
    """
    Find optimal k using cross-validation.
    
    Parameters:
    -----------
    X_train : numpy array
        Training features
    y_train : numpy array
        Training labels
    k_range : range
        Range of k values to test
    cv : int
        Number of cross-validation folds
    
    Returns:
    --------
    best_k : int
        Optimal k value
    k_scores : dict
        Dictionary mapping k to average CV score
    """
    k_scores = {}
    
    for k in k_range:
        knn = KNNClassifier(k=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        k_scores[k] = np.mean(scores)
        print(f"k={k}: CV Accuracy = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    best_k = max(k_scores, key=k_scores.get)
    print(f"\nBest k: {best_k} with accuracy: {k_scores[best_k]:.4f}")
    
    return best_k, k_scores

# Usage
# best_k, scores = find_optimal_k(X_train, y_train, k_range=range(1, 21))
```

---

## Complete Example: Comparison of Distance Metrics

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Test different distance metrics
distance_metrics = ['euclidean', 'manhattan', 'cosine']
results = {}

for metric in distance_metrics:
    knn = KNNClassifier(k=5, distance_metric=metric, weights='uniform')
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    results[metric] = accuracy
    print(f"{metric.capitalize()} Distance: Accuracy = {accuracy:.4f}")

# Compare uniform vs distance-weighted
knn_uniform = KNNClassifier(k=5, weights='uniform')
knn_weighted = KNNClassifier(k=5, weights='distance')

knn_uniform.fit(X_train, y_train)
knn_weighted.fit(X_train, y_train)

acc_uniform = knn_uniform.score(X_test, y_test)
acc_weighted = knn_weighted.score(X_test, y_test)

print(f"\nUniform weights: {acc_uniform:.4f}")
print(f"Distance-weighted: {acc_weighted:.4f}")
```

---

## Key Takeaways

1. **Lazy Learning**: KNN doesn't build a model during training - it stores all data
2. **Distance Matters**: Choice of distance metric significantly affects performance
3. **k Selection**: 
   - Small k: Low bias, high variance (sensitive to noise)
   - Large k: High bias, low variance (smoother decision boundary)
4. **Weighted Voting**: Distance-weighted voting often improves performance
5. **Feature Scaling**: Essential for distance-based algorithms
6. **Computational Cost**: Prediction is O(m * n) where m = training samples, n = features

---

## When to Use KNN

| Scenario | Suitable? | Notes |
|----------|-----------|-------|
| **Small datasets** | ✅ Yes | Works well with limited data |
| **Non-linear boundaries** | ✅ Yes | Can model complex decision boundaries |
| **Multi-class classification** | ✅ Yes | Naturally handles multiple classes |
| **Large datasets** | ❌ No | Prediction becomes slow |
| **High-dimensional data** | ❌ No | Curse of dimensionality |
| **Real-time prediction** | ❌ No | Requires computing distances to all training points |
| **Noisy data** | ⚠️ Maybe | Use larger k to smooth out noise |

---

## Interview Tips

When implementing KNN in interviews:

1. **Explain Lazy Learning**: KNN is instance-based, no explicit model
2. **Discuss Distance Metrics**: Know when to use Euclidean vs Manhattan vs Cosine
3. **Handle Edge Cases**:
   - Ties in voting (random selection or weighted)
   - Zero distances (use epsilon)
   - Missing features
4. **Optimization**: Mention KD-trees or Ball trees for large datasets
5. **Feature Scaling**: Always normalize features before using KNN
6. **k Selection**: Explain bias-variance tradeoff
7. **Time Complexity**: O(m * n) for prediction, can be optimized with data structures

---

## Time Complexity

- **Training**: O(1) - just stores data (lazy learning)
- **Prediction**: O(m * n) where m = training samples, n = features
  - For each query: calculate distance to all m training points
- **Space Complexity**: O(m * n) to store all training data

### Optimization Techniques

1. **KD-Tree**: O(n log m) average case for prediction
2. **Ball Tree**: Similar to KD-tree, better for high dimensions
3. **Locality Sensitive Hashing (LSH)**: Approximate nearest neighbors
4. **Vectorization**: Use NumPy broadcasting for batch predictions

---

## Advantages

1. **Simple**: Easy to understand and implement
2. **No Training**: No model building phase
3. **Non-parametric**: Makes no assumptions about data distribution
4. **Multi-class**: Naturally handles multiple classes
5. **Interpretable**: Can explain predictions by showing nearest neighbors

---

## Disadvantages

1. **Slow Prediction**: Must compute distances to all training points
2. **Memory Intensive**: Stores all training data
3. **Sensitive to Irrelevant Features**: All features contribute equally to distance
4. **Curse of Dimensionality**: Performance degrades in high dimensions
5. **Sensitive to Scale**: Requires feature normalization
6. **No Model**: Can't extract patterns or insights from the model itself

---

## References

* Instance-based learning and lazy learning
* Distance metrics and similarity measures
* Bias-variance tradeoff in k selection
* KD-trees and spatial data structures
* Curse of dimensionality

