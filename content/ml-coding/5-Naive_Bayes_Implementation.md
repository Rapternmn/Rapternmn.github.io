+++
title = "Naive Bayes Implementation from Scratch"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 5
description = "Complete implementation of Naive Bayes classifier from scratch using Python and NumPy. Covers Gaussian, Multinomial, and Bernoulli Naive Bayes, Bayes' theorem, conditional probability, and practical examples."
+++

---

## Introduction

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with the "naive" assumption of feature independence. Despite this simplifying assumption, Naive Bayes often performs remarkably well and is particularly useful for text classification, spam filtering, and recommendation systems.

In this guide, we'll implement Naive Bayes from scratch using Python and NumPy, covering Gaussian, Multinomial, and Bernoulli variants.

---

## Mathematical Foundation

### Bayes' Theorem

```
P(y|X) = P(X|y) * P(y) / P(X)
```

Where:
- `P(y|X)` is the posterior probability (probability of class y given features X)
- `P(X|y)` is the likelihood (probability of features X given class y)
- `P(y)` is the prior probability (probability of class y)
- `P(X)` is the evidence (probability of features X)

### Naive Assumption

The "naive" assumption is that features are conditionally independent given the class:

```
P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y) = ∏ᵢ P(xᵢ|y)
```

This simplifies the calculation significantly.

### Classification Rule

For classification, we predict the class with the highest posterior probability:

```
ŷ = argmax_y P(y|X) = argmax_y P(y) * ∏ᵢ P(xᵢ|y)
```

Since `P(X)` is constant for all classes, we can ignore it in the maximization.

### Log Probability (Numerical Stability)

To avoid underflow with small probabilities, we use log probabilities:

```
log P(y|X) = log P(y) + Σᵢ log P(xᵢ|y)
```

---

## Implementation 1: Gaussian Naive Bayes

Gaussian Naive Bayes assumes that continuous features follow a Gaussian (normal) distribution.

### Mathematical Foundation

For each feature in each class, we estimate:
- Mean: `μᵧⱼ = (1/nᵧ) * Σᵢ:yᵢ=y xᵢⱼ`
- Variance: `σ²ᵧⱼ = (1/nᵧ) * Σᵢ:yᵢ=y (xᵢⱼ - μᵧⱼ)²`

The probability density function (PDF) for a feature value x given class y:

```
P(xⱼ|y) = (1/√(2πσ²ᵧⱼ)) * exp(-(xⱼ - μᵧⱼ)² / (2σ²ᵧⱼ))
```

In log form:

```
log P(xⱼ|y) = -0.5 * log(2πσ²ᵧⱼ) - (xⱼ - μᵧⱼ)² / (2σ²ᵧⱼ)
```

### Implementation

```python
import numpy as np
from collections import Counter

class GaussianNaiveBayes:
    def __init__(self, smoothing=1e-9):
        """
        Initialize Gaussian Naive Bayes classifier.
        
        Parameters:
        -----------
        smoothing : float
            Small value added to variance to prevent division by zero
        """
        self.smoothing = smoothing
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None
    
    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes classifier.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        m, n = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize arrays
        self.means_ = np.zeros((n_classes, n))
        self.variances_ = np.zeros((n_classes, n))
        self.class_priors_ = np.zeros(n_classes)
        
        # Calculate statistics for each class
        for i, c in enumerate(self.classes_):
            # Get samples for this class
            X_c = X[y == c]
            n_c = len(X_c)
            
            # Calculate prior probability
            self.class_priors_[i] = n_c / m
            
            # Calculate mean for each feature
            self.means_[i] = np.mean(X_c, axis=0)
            
            # Calculate variance for each feature
            self.variances_[i] = np.var(X_c, axis=0) + self.smoothing
        
        return self
    
    def _calculate_log_likelihood(self, X):
        """
        Calculate log likelihood for each class.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        log_likelihood : numpy array
            Log likelihood of shape (m, n_classes)
        """
        m, n = X.shape
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((m, n_classes))
        
        for i in range(n_classes):
            # Log PDF for Gaussian distribution
            # log P(x|y) = -0.5 * log(2πσ²) - (x - μ)² / (2σ²)
            diff = X - self.means_[i]
            variance = self.variances_[i]
            
            # Calculate log probability for each feature
            log_prob = -0.5 * np.log(2 * np.pi * variance)
            log_prob -= 0.5 * (diff ** 2) / variance
            
            # Sum log probabilities (product in probability space)
            log_likelihood[:, i] = np.sum(log_prob, axis=1)
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        probabilities : numpy array
            Class probabilities of shape (m, n_classes)
        """
        # Calculate log likelihood
        log_likelihood = self._calculate_log_likelihood(X)
        
        # Add log prior
        log_posterior = log_likelihood + np.log(self.class_priors_)
        
        # Normalize using log-sum-exp trick for numerical stability
        # P(y|X) = exp(log_posterior) / sum(exp(log_posterior))
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        log_posterior_shifted = log_posterior - log_posterior_max
        exp_log_posterior = np.exp(log_posterior_shifted)
        probabilities = exp_log_posterior / np.sum(exp_log_posterior, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels of shape (m,)
        """
        probabilities = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        return predictions
    
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

# Create and train model
model = GaussianNaiveBayes()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"\nPredicted probabilities for first 5 samples:")
print(y_proba[:5])
print(f"\nTrue labels: {y_test[:5]}")
print(f"Predicted labels: {y_pred[:5]}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## Implementation 2: Multinomial Naive Bayes

Multinomial Naive Bayes is used for discrete count data, commonly in text classification where features represent word counts.

### Mathematical Foundation

For each class y and feature j, we estimate:
- Count of feature j in class y: `Nᵧⱼ = Σᵢ:yᵢ=y xᵢⱼ`
- Total count of all features in class y: `Nᵧ = Σⱼ Nᵧⱼ`

The probability of feature j given class y (with Laplace smoothing):

```
P(xⱼ|y) = (Nᵧⱼ + α) / (Nᵧ + α * n)
```

Where:
- `α` is the smoothing parameter (typically 1.0 for Laplace smoothing)
- `n` is the number of features

### Implementation

```python
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize Multinomial Naive Bayes classifier.
        
        Parameters:
        -----------
        alpha : float
            Smoothing parameter (Laplace smoothing if alpha=1.0)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.feature_counts_ = None
        self.class_totals_ = None
    
    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes classifier.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n) with non-negative counts
        y : numpy array
            Target vector of shape (m,)
        """
        m, n = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize arrays
        self.feature_counts_ = np.zeros((n_classes, n))
        self.class_totals_ = np.zeros(n_classes)
        self.class_priors_ = np.zeros(n_classes)
        
        # Calculate counts for each class
        for i, c in enumerate(self.classes_):
            # Get samples for this class
            X_c = X[y == c]
            n_c = len(X_c)
            
            # Calculate prior probability
            self.class_priors_[i] = n_c / m
            
            # Sum feature counts for this class
            self.feature_counts_[i] = np.sum(X_c, axis=0)
            self.class_totals_[i] = np.sum(self.feature_counts_[i])
        
        return self
    
    def _calculate_log_likelihood(self, X):
        """
        Calculate log likelihood for each class.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        log_likelihood : numpy array
            Log likelihood of shape (m, n_classes)
        """
        m, n = X.shape
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((m, n_classes))
        
        for i in range(n_classes):
            # Calculate log probabilities: log P(xⱼ|y) = log((Nᵧⱼ + α) / (Nᵧ + α*n))
            # Using log properties: log(a/b) = log(a) - log(b)
            numerator = self.feature_counts_[i] + self.alpha
            denominator = self.class_totals_[i] + self.alpha * n
            
            log_feature_probs = np.log(numerator) - np.log(denominator)
            
            # For each sample, sum log probabilities weighted by feature counts
            # log P(X|y) = Σⱼ xⱼ * log P(xⱼ|y)
            log_likelihood[:, i] = X.dot(log_feature_probs)
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        probabilities : numpy array
            Class probabilities of shape (m, n_classes)
        """
        # Calculate log likelihood
        log_likelihood = self._calculate_log_likelihood(X)
        
        # Add log prior
        log_posterior = log_likelihood + np.log(self.class_priors_)
        
        # Normalize using log-sum-exp trick
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        log_posterior_shifted = log_posterior - log_posterior_max
        exp_log_posterior = np.exp(log_posterior_shifted)
        probabilities = exp_log_posterior / np.sum(exp_log_posterior, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels of shape (m,)
        """
        probabilities = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        return predictions
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

### Usage Example (Text Classification)

```python
# Example: Simple text classification
# Features represent word counts

# Sample documents (word counts for: "python", "data", "science", "machine", "learning")
X_train = np.array([
    [5, 3, 2, 1, 4],  # Document 1: high "python" and "learning"
    [1, 4, 5, 2, 3],  # Document 2: high "data" and "science"
    [2, 2, 1, 5, 4],  # Document 3: high "machine" and "learning"
    [4, 1, 1, 1, 2],  # Document 4: high "python"
    [1, 5, 4, 1, 2],  # Document 5: high "data" and "science"
])

y_train = np.array([0, 1, 0, 0, 1])  # 0: programming, 1: data science

# Test documents
X_test = np.array([
    [6, 2, 1, 1, 5],  # Should be class 0
    [1, 4, 5, 1, 2],  # Should be class 1
])

# Create and train model
model = MultinomialNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions:", predictions)
print("Probabilities:\n", probabilities)
```

---

## Implementation 3: Bernoulli Naive Bayes

Bernoulli Naive Bayes is used for binary features (present/absent), commonly in text classification where features indicate presence/absence of words.

### Mathematical Foundation

For each class y and feature j, we estimate:
- Probability that feature j is present in class y: `P(xⱼ=1|y)`
- Probability that feature j is absent in class y: `P(xⱼ=0|y) = 1 - P(xⱼ=1|y)`

With Laplace smoothing:

```
P(xⱼ=1|y) = (Nᵧⱼ + α) / (Nᵧ + 2α)
```

Where:
- `Nᵧⱼ` is the count of samples in class y where feature j is present
- `Nᵧ` is the total number of samples in class y
- `α` is the smoothing parameter

### Implementation

```python
class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize Bernoulli Naive Bayes classifier.
        
        Parameters:
        -----------
        alpha : float
            Smoothing parameter (Laplace smoothing if alpha=1.0)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.feature_probs_ = None  # P(xⱼ=1|y)
    
    def fit(self, X, y):
        """
        Train the Bernoulli Naive Bayes classifier.
        
        Parameters:
        -----------
        X : numpy array
            Binary feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        m, n = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize arrays
        self.feature_probs_ = np.zeros((n_classes, n))
        self.class_priors_ = np.zeros(n_classes)
        
        # Calculate probabilities for each class
        for i, c in enumerate(self.classes_):
            # Get samples for this class
            X_c = X[y == c]
            n_c = len(X_c)
            
            # Calculate prior probability
            self.class_priors_[i] = n_c / m
            
            # Calculate P(xⱼ=1|y) with Laplace smoothing
            # P(xⱼ=1|y) = (count of xⱼ=1 in class y + α) / (n_c + 2α)
            feature_counts = np.sum(X_c, axis=0)
            self.feature_probs_[i] = (feature_counts + self.alpha) / (n_c + 2 * self.alpha)
        
        return self
    
    def _calculate_log_likelihood(self, X):
        """
        Calculate log likelihood for each class.
        
        Parameters:
        -----------
        X : numpy array
            Binary feature matrix of shape (m, n)
        
        Returns:
        --------
        log_likelihood : numpy array
            Log likelihood of shape (m, n_classes)
        """
        m, n = X.shape
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((m, n_classes))
        
        for i in range(n_classes):
            # For each feature j:
            # If xⱼ = 1: log P(xⱼ=1|y)
            # If xⱼ = 0: log P(xⱼ=0|y) = log(1 - P(xⱼ=1|y))
            
            prob_present = self.feature_probs_[i]
            prob_absent = 1 - prob_present
            
            # Calculate log probabilities
            log_prob_present = np.log(prob_present + 1e-10)  # Add small value to avoid log(0)
            log_prob_absent = np.log(prob_absent + 1e-10)
            
            # For each sample, sum log probabilities
            # log P(X|y) = Σⱼ [xⱼ * log P(xⱼ=1|y) + (1-xⱼ) * log P(xⱼ=0|y)]
            log_likelihood[:, i] = np.sum(
                X * log_prob_present + (1 - X) * log_prob_absent,
                axis=1
            )
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : numpy array
            Binary feature matrix of shape (m, n)
        
        Returns:
        --------
        probabilities : numpy array
            Class probabilities of shape (m, n_classes)
        """
        # Calculate log likelihood
        log_likelihood = self._calculate_log_likelihood(X)
        
        # Add log prior
        log_posterior = log_likelihood + np.log(self.class_priors_)
        
        # Normalize using log-sum-exp trick
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        log_posterior_shifted = log_posterior - log_posterior_max
        exp_log_posterior = np.exp(log_posterior_shifted)
        probabilities = exp_log_posterior / np.sum(exp_log_posterior, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : numpy array
            Binary feature matrix of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels of shape (m,)
        """
        probabilities = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        return predictions
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

### Usage Example

```python
# Example: Binary feature classification
# Features represent presence (1) or absence (0) of words

X_train = np.array([
    [1, 1, 0, 0, 1],  # Document 1: has "python", "data", "learning"
    [0, 1, 1, 1, 0],  # Document 2: has "data", "science", "machine"
    [1, 0, 0, 1, 1],  # Document 3: has "python", "machine", "learning"
    [1, 0, 0, 0, 0],  # Document 4: has only "python"
    [0, 1, 1, 0, 0],  # Document 5: has "data", "science"
])

y_train = np.array([0, 1, 0, 0, 1])  # 0: programming, 1: data science

X_test = np.array([
    [1, 1, 0, 0, 1],  # Should be class 0
    [0, 1, 1, 0, 0],  # Should be class 1
])

# Create and train model
model = BernoulliNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions:", predictions)
print("Probabilities:\n", probabilities)
```

---

## Complete Example: Comparison of Naive Bayes Variants

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate continuous data for Gaussian NB
X_continuous, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Generate count data for Multinomial NB (convert to counts)
X_counts = (X_continuous * 10).astype(int)
X_counts = np.abs(X_counts)  # Ensure non-negative

# Generate binary data for Bernoulli NB
X_binary = (X_continuous > 0).astype(int)

# Split data
X_train_c, X_test_c, y_train, y_test = train_test_split(
    X_continuous, y, test_size=0.2, random_state=42
)
X_train_m, X_test_m, _, _ = train_test_split(
    X_counts, y, test_size=0.2, random_state=42
)
X_train_b, X_test_b, _, _ = train_test_split(
    X_binary, y, test_size=0.2, random_state=42
)

# Test all three variants
models = {
    'Gaussian NB': (GaussianNaiveBayes(), X_train_c, X_test_c),
    'Multinomial NB': (MultinomialNaiveBayes(), X_train_m, X_test_m),
    'Bernoulli NB': (BernoulliNaiveBayes(), X_train_b, X_test_b),
}

results = {}

for name, (model, X_train, X_test) in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred))

# Compare results
print("\n" + "="*50)
print("Accuracy Comparison:")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {acc:.4f}")
```

---

## Key Takeaways

1. **Gaussian Naive Bayes**: Best for continuous features that follow a normal distribution
2. **Multinomial Naive Bayes**: Best for count data (e.g., word counts in text)
3. **Bernoulli Naive Bayes**: Best for binary features (e.g., presence/absence of words)
4. **Laplace Smoothing**: Prevents zero probabilities and handles unseen features
5. **Log Probabilities**: Essential for numerical stability when dealing with small probabilities
6. **Feature Independence**: The "naive" assumption simplifies computation but may not hold in practice

---

## When to Use Each Variant

| Variant | Use Case | Feature Type |
|---------|----------|--------------|
| **Gaussian** | Continuous numerical data | Real-valued features |
| **Multinomial** | Count data, text classification | Non-negative integers |
| **Bernoulli** | Binary features, text with presence/absence | Binary (0/1) |

---

## Interview Tips

When implementing Naive Bayes in interviews:

1. **Explain Bayes' Theorem**: Show understanding of the mathematical foundation
2. **Discuss the Naive Assumption**: Explain why it's called "naive" and when it might fail
3. **Handle Edge Cases**: 
   - Zero probabilities (use smoothing)
   - Numerical underflow (use log probabilities)
   - Missing features
4. **Choose the Right Variant**: Justify why you chose Gaussian/Multinomial/Bernoulli
5. **Explain Smoothing**: Why Laplace smoothing is important
6. **Time Complexity**: O(m * n) for training, O(m * n * k) for prediction where k is number of classes

---

## Time Complexity

- **Training**: O(m * n) where m = samples, n = features
  - Calculate means/variances (Gaussian) or counts (Multinomial/Bernoulli) for each class
- **Prediction**: O(m * n * k) where k = number of classes
  - Calculate likelihood for each sample, feature, and class
- **Space Complexity**: O(k * n) to store class statistics

---

## Advantages

1. **Fast Training and Prediction**: Simple calculations, no iterative optimization
2. **Works Well with Small Datasets**: Requires less data than many other algorithms
3. **Handles Multiple Classes**: Naturally handles multi-class classification
4. **Probabilistic Output**: Provides probability estimates, not just predictions
5. **Robust to Irrelevant Features**: Can handle many features

---

## Disadvantages

1. **Feature Independence Assumption**: Rarely true in practice
2. **Sensitive to Feature Distribution**: Gaussian NB assumes normal distribution
3. **Zero Frequency Problem**: Without smoothing, unseen features cause issues
4. **Not Ideal for Regression**: Designed for classification

---

## References

* Bayes' theorem and conditional probability
* Gaussian distribution and probability density function
* Laplace smoothing and additive smoothing
* Text classification with Naive Bayes
* Log-sum-exp trick for numerical stability

