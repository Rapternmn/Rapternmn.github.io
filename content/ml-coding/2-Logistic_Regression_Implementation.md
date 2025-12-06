+++
title = "Logistic Regression Implementation from Scratch"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Complete implementation of logistic regression from scratch using Python and NumPy. Covers binary classification, multiclass classification, gradient descent, regularization, and practical examples with decision boundaries."
+++

---

## Introduction

Logistic Regression is a fundamental classification algorithm that models the probability of a binary outcome using a logistic (sigmoid) function. Despite its name, it's used for classification, not regression.

In this guide, we'll implement logistic regression from scratch using Python and NumPy, covering binary classification, multiclass classification, and various optimization techniques.

---

## Mathematical Foundation

### Hypothesis Function

Unlike linear regression, logistic regression uses the sigmoid function to output probabilities:

```
h(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

Where `σ(z)` is the sigmoid function:
```
σ(z) = 1 / (1 + e^(-z))
```

### Sigmoid Function Properties

- **Range**: (0, 1) - outputs probabilities
- **S-shape**: Smooth, differentiable curve
- **Decision Boundary**: When `h(x) ≥ 0.5`, predict class 1; otherwise predict class 0
- **Symmetric**: `σ(-z) = 1 - σ(z)`

### Cost Function (Log Loss / Cross-Entropy)

For binary classification:

```
J(θ) = -(1/m) * Σ[yᵢ * log(h(xᵢ)) + (1 - yᵢ) * log(1 - h(xᵢ))]
```

This is also called **Binary Cross-Entropy Loss**.

**Why this cost function?**
- Penalizes confident wrong predictions heavily
- Convex function (guarantees global minimum)
- Works well with gradient descent

### Gradient of Cost Function

The gradient is surprisingly similar to linear regression:

```
∂J/∂θⱼ = (1/m) * Σ(h(xᵢ) - yᵢ) * xᵢⱼ
```

In vectorized form:
```
∇J(θ) = (1/m) * Xᵀ(h(X) - y)
```

Where `h(X)` is the vector of predictions for all training examples.

### Sigmoid Derivative

The derivative of the sigmoid function is:

```
σ'(z) = σ(z) * (1 - σ(z))
```

This elegant property makes gradient computation efficient.

---

## Implementation 1: Binary Logistic Regression

### Basic Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent (default: 0.01)
        n_iterations : int
            Number of iterations for gradient descent (default: 1000)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """
        Compute sigmoid function.
        
        Parameters:
        -----------
        z : numpy array
            Input values
        
        Returns:
        --------
        sigmoid : numpy array
            Sigmoid of input values
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X):
        """Add bias term (column of ones) to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,) with values 0 or 1
        """
        # Add bias term
        X = self._add_bias(X)
        m, n = X.shape
        
        # Initialize parameters (theta) with zeros
        self.theta = np.zeros(n)
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Compute predictions (probabilities)
            z = X.dot(self.theta)
            h = self._sigmoid(z)
            
            # Calculate cost (log loss)
            cost = -(1 / m) * np.sum(
                y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15)
            )
            self.cost_history.append(cost)
            
            # Calculate gradient
            error = h - y
            gradient = (1 / m) * X.T.dot(error)
            # Note: gradient[0] is the bias gradient (since X's first column is all ones)
            #       gradient[1:] are the feature gradients
            
            # Update parameters (including bias)
            # self.theta[0] is the bias term, self.theta[1:] are feature weights
            self.theta -= self.learning_rate * gradient
            
            # Optional: Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Cost = {cost:.4f}")
    
    def predict_proba(self, X):
        """
        Predict probabilities for each class.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        probabilities : numpy array
            Probabilities of shape (m,) for class 1
        """
        X = self._add_bias(X)
        z = X.dot(self.theta)
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        threshold : float
            Decision threshold (default: 0.5)
        
        Returns:
        --------
        predictions : numpy array
            Binary predictions (0 or 1) of shape (m,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y, threshold=0.5):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            True target values
        threshold : float
            Decision threshold
        
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        y_pred = self.predict(X, threshold)
        return np.mean(y_pred == y)
```

### Usage Example

```python
# Generate sample data (two classes)
np.random.seed(42)
n_samples = 200

# Class 0: centered at (-2, -2)
X0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
y0 = np.zeros(n_samples // 2)

# Class 1: centered at (2, 2)
X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
y1 = np.ones(n_samples // 2)

# Combine
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# Shuffle
indices = np.random.permutation(n_samples)
X = X[indices]
y = y[indices]

# Create and train model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
probabilities = model.predict_proba(X)

# Evaluate
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
print(f"Learned parameters: θ₀ = {model.theta[0]:.4f}, θ₁ = {model.theta[1]:.4f}, θ₂ = {model.theta[2]:.4f}")

# Plot decision boundary
plt.figure(figsize=(12, 5))

# Plot 1: Cost history
plt.subplot(1, 2, 1)
plt.plot(model.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History')
plt.grid(True)

# Plot 2: Data and decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)

# Plot decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))
X_grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = model.predict_proba(X_grid).reshape(xx1.shape)
plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
plt.contourf(xx1, xx2, Z, levels=50, alpha=0.3, cmap='RdYlBu')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Decision Boundary')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Implementation 2: Regularized Logistic Regression

### L2 Regularization (Ridge)

Regularization prevents overfitting by penalizing large parameter values:

```
J(θ) = -(1/m) * Σ[yᵢ * log(h(xᵢ)) + (1 - yᵢ) * log(1 - h(xᵢ))] + (λ/2m) * Σθⱼ²
```

Note: We typically don't regularize `θ₀` (bias term).

### Implementation

```python
class RegularizedLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.1):
        """
        Initialize Regularized Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of iterations
        lambda_reg : float
            Regularization parameter (λ)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.theta = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Compute sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """
        Train the regularized logistic regression model.
        
        Note: We don't regularize θ₀ (bias term).
        """
        X = self._add_bias(X)
        m, n = X.shape
        
        # Initialize parameters
        self.theta = np.zeros(n)
        
        for i in range(self.n_iterations):
            # Compute predictions
            z = X.dot(self.theta)
            h = self._sigmoid(z)
            
            # Calculate cost (with regularization)
            cost = -(1 / m) * np.sum(
                y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15)
            )
            
            # Add regularization term (excluding bias θ₀)
            reg_term = (self.lambda_reg / (2 * m)) * np.sum(self.theta[1:] ** 2)
            cost += reg_term
            self.cost_history.append(cost)
            
            # Calculate gradient
            error = h - y
            gradient = (1 / m) * X.T.dot(error)
            # Note: gradient[0] is the bias gradient (since X's first column is all ones)
            #       gradient[1:] are the feature gradients
            
            # Add regularization to gradient (excluding bias)
            # We don't regularize the bias term, only the feature weights
            gradient[1:] += (self.lambda_reg / m) * self.theta[1:]
            
            # Update parameters (including bias)
            # self.theta[0] is the bias term (updated without regularization)
            # self.theta[1:] are feature weights (updated with regularization)
            self.theta -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        """Predict probabilities for each class."""
        X = self._add_bias(X)
        z = X.dot(self.theta)
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y, threshold=0.5):
        """Calculate accuracy score."""
        y_pred = self.predict(X, threshold)
        return np.mean(y_pred == y)
```

---

## Implementation 3: Multiclass Classification (One-vs-All)

For multiclass problems, we train multiple binary classifiers (one for each class).

### Implementation

```python
class MulticlassLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.1):
        """
        Initialize Multiclass Logistic Regression using One-vs-All.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of iterations
        lambda_reg : float
            Regularization parameter
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.classifiers = {}  # Store one classifier per class
        self.classes = None
    
    def _sigmoid(self, z):
        """Compute sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """
        Train multiple binary classifiers (one per class).
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,) with class labels
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        print(f"Training {n_classes} binary classifiers...")
        
        # Train one classifier for each class
        for i, class_label in enumerate(self.classes):
            print(f"Training classifier {i+1}/{n_classes} for class {class_label}...")
            
            # Create binary labels: 1 for current class, 0 for others
            y_binary = (y == class_label).astype(int)
            
            # Train binary classifier
            classifier = RegularizedLogisticRegression(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                lambda_reg=self.lambda_reg
            )
            classifier.fit(X, y_binary)
            
            self.classifiers[class_label] = classifier
    
    def predict_proba(self, X):
        """
        Predict probabilities for each class.
        
        Returns:
        --------
        probabilities : numpy array
            Probabilities of shape (m, n_classes)
        """
        m = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((m, n_classes))
        
        for i, class_label in enumerate(self.classes):
            classifier = self.classifiers[class_label]
            probabilities[:, i] = classifier.predict_proba(X)
        
        # Normalize probabilities (softmax-like, but for one-vs-all)
        # Each probability is independent, so we normalize to sum to 1
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        Predict class labels.
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels of shape (m,)
        """
        probabilities = self.predict_proba(X)
        # Return class with highest probability
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes[class_indices]
    
    def score(self, X, y):
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
```

### Usage Example

```python
# Generate multiclass data
np.random.seed(42)
n_samples_per_class = 100

# Three classes
X0 = np.random.randn(n_samples_per_class, 2) + np.array([-2, -2])
X1 = np.random.randn(n_samples_per_class, 2) + np.array([2, 0])
X2 = np.random.randn(n_samples_per_class, 2) + np.array([0, 2])

X = np.vstack([X0, X1, X2])
y = np.hstack([
    np.zeros(n_samples_per_class),
    np.ones(n_samples_per_class),
    np.full(n_samples_per_class, 2)
])

# Shuffle
indices = np.random.permutation(len(y))
X = X[indices]
y = y[indices]

# Create and train model
model = MulticlassLogisticRegression(
    learning_rate=0.1,
    n_iterations=1000,
    lambda_reg=0.1
)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
probabilities = model.predict_proba(X)

# Evaluate
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Plot results
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, class_label in enumerate(model.classes):
    mask = y == class_label
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                label=f'Class {class_label}', alpha=0.6, s=50)
    
    # Plot misclassified points
    misclassified = (y == class_label) & (y_pred != class_label)
    if np.any(misclassified):
        plt.scatter(X[misclassified, 0], X[misclassified, 1], 
                   c='black', marker='x', s=100, linewidths=2, 
                   label='Misclassified' if i == 0 else '')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title(f'Multiclass Classification (Accuracy: {accuracy:.4f})')
plt.grid(True)
plt.show()
```

---

## Implementation 4: Softmax Regression (Multinomial Logistic Regression)

Softmax regression is a more elegant approach for multiclass classification that directly outputs class probabilities.

### Mathematical Foundation

**Softmax Function:**
```
P(y = k | x) = e^(θₖᵀx) / Σⱼ e^(θⱼᵀx)
```

**Cost Function (Cross-Entropy):**
```
J(θ) = -(1/m) * Σᵢ Σₖ [yᵢₖ * log(P(yᵢ = k | xᵢ))]
```

Where `yᵢₖ = 1` if example `i` belongs to class `k`, else `0`.

### Implementation

```python
class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.1):
        """
        Initialize Softmax Regression (Multinomial Logistic Regression).
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of iterations
        lambda_reg : float
            Regularization parameter
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.theta = None  # Shape: (n_features, n_classes)
        self.classes = None
    
    def _softmax(self, z):
        """
        Compute softmax function.
        
        Parameters:
        -----------
        z : numpy array
            Input of shape (m, n_classes)
        
        Returns:
        --------
        softmax : numpy array
            Probabilities of shape (m, n_classes)
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """
        Train the softmax regression model.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,) with class labels
        """
        X = self._add_bias(X)
        m, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Create one-hot encoded labels
        y_onehot = np.zeros((m, n_classes))
        for i, class_label in enumerate(self.classes):
            y_onehot[y == class_label, i] = 1
        
        # Initialize parameters: (n_features, n_classes)
        self.theta = np.zeros((n_features, n_classes))
        
        for i in range(self.n_iterations):
            # Compute predictions (probabilities)
            z = X.dot(self.theta)  # Shape: (m, n_classes)
            h = self._softmax(z)  # Shape: (m, n_classes)
            
            # Calculate cost (cross-entropy)
            cost = -(1 / m) * np.sum(y_onehot * np.log(h + 1e-15))
            
            # Add regularization term
            reg_term = (self.lambda_reg / (2 * m)) * np.sum(self.theta[1:, :] ** 2)
            cost += reg_term
            self.cost_history.append(cost)
            
            # Calculate gradient
            error = h - y_onehot  # Shape: (m, n_classes)
            gradient = (1 / m) * X.T.dot(error)  # Shape: (n_features, n_classes)
            
            # Add regularization to gradient (excluding bias)
            gradient[1:, :] += (self.lambda_reg / m) * self.theta[1:, :]
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        """
        Predict probabilities for each class.
        
        Returns:
        --------
        probabilities : numpy array
            Probabilities of shape (m, n_classes)
        """
        X = self._add_bias(X)
        z = X.dot(self.theta)
        return self._softmax(z)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Returns:
        --------
        predictions : numpy array
            Predicted class labels of shape (m,)
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes[class_indices]
    
    def score(self, X, y):
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
```

---

## Advanced: Stochastic Gradient Descent

```python
class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        """
        Initialize Logistic Regression with Stochastic Gradient Descent.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_epochs : int
            Number of epochs (passes through entire dataset)
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.theta = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Compute sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """Train model using Stochastic Gradient Descent."""
        X = self._add_bias(X)
        m, n = X.shape
        
        # Initialize parameters
        self.theta = np.zeros(n)
        
        for epoch in range(self.n_epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            # Process each example
            for i in range(m):
                # Single example
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]
                
                # Prediction
                z_i = x_i.dot(self.theta)
                h_i = self._sigmoid(z_i)
                
                # Calculate error
                error = h_i - y_i
                
                # Update parameters (using single example)
                self.theta -= self.learning_rate * error * x_i
                
                # Accumulate cost
                epoch_cost += -(y_i * np.log(h_i + 1e-15) + 
                               (1 - y_i) * np.log(1 - h_i + 1e-15))
            
            # Average cost for this epoch
            avg_cost = epoch_cost / m
            self.cost_history.append(avg_cost)
    
    def predict_proba(self, X):
        """Predict probabilities for each class."""
        X = self._add_bias(X)
        z = X.dot(self.theta)
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
```

---

## Evaluation Metrics

### Confusion Matrix

```python
def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Returns:
    --------
    cm : numpy array
        Confusion matrix of shape (2, 2)
        [[TN, FP],
         [FN, TP]]
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp],
                     [fn, tp]])

def classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Returns:
    --------
    metrics : dict
        Dictionary containing accuracy, precision, recall, F1-score
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
```

### ROC Curve and AUC

```python
def roc_curve(y_true, y_scores):
    """
    Compute ROC curve.
    
    Parameters:
    -----------
    y_true : numpy array
        True binary labels
    y_scores : numpy array
        Predicted probabilities
    
    Returns:
    --------
    fpr : numpy array
        False positive rates
    tpr : numpy array
        True positive rates
    thresholds : numpy array
        Threshold values
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Calculate TPR and FPR for each threshold
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    return np.array(fpr), np.array(tpr), thresholds

def auc_score(fpr, tpr):
    """
    Calculate Area Under the ROC Curve (AUC).
    
    Uses trapezoidal rule for integration.
    """
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr_sorted, fpr_sorted)
    return auc
```

---

## Complete Example: End-to-End Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
n_samples = 500

# Class 0: centered at (-1, -1)
X0 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
y0 = np.zeros(n_samples // 2)

# Class 1: centered at (1, 1)
X1 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
y1 = np.ones(n_samples // 2)

# Combine
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# Shuffle
indices = np.random.permutation(n_samples)
X = X[indices]
y = y[indices]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Calculate metrics
train_metrics = classification_metrics(y_train, y_train_pred)
test_metrics = classification_metrics(y_test, y_test_pred)

print("\nTest Set Metrics:")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")
print(f"  F1-Score: {test_metrics['f1_score']:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
auc = auc_score(fpr, tpr)
print(f"\nAUC Score: {auc:.4f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Cost history
axes[0, 0].plot(model.cost_history)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Cost')
axes[0, 0].set_title('Cost History')
axes[0, 0].grid(True)

# Plot 2: Decision boundary (training data)
axes[0, 1].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
                   c='blue', label='Class 0', alpha=0.6)
axes[0, 1].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                   c='red', label='Class 1', alpha=0.6)

# Decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))
X_grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = model.predict_proba(X_grid).reshape(xx1.shape)
axes[0, 1].contour(xx1, xx2, Z, levels=[0.5], colors='black', 
                   linestyles='--', linewidths=2)
axes[0, 1].contourf(xx1, xx2, Z, levels=50, alpha=0.3, cmap='RdYlBu')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')
axes[0, 1].set_title(f'Training Data (Accuracy: {train_accuracy:.4f})')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Decision boundary (test data)
axes[1, 0].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], 
                   c='blue', label='Class 0', alpha=0.6)
axes[1, 0].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
                   c='red', label='Class 1', alpha=0.6)
axes[1, 0].contour(xx1, xx2, Z, levels=[0.5], colors='black', 
                   linestyles='--', linewidths=2)
axes[1, 0].contourf(xx1, xx2, Z, levels=50, alpha=0.3, cmap='RdYlBu')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')
axes[1, 0].set_title(f'Test Data (Accuracy: {test_accuracy:.4f})')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: ROC Curve
axes[1, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

1. **Sigmoid Function**: Maps linear combination to probabilities (0, 1)
2. **Log Loss**: Appropriate cost function for classification
3. **Gradient**: Similar form to linear regression, but uses sigmoid
4. **Decision Boundary**: Linear decision boundary (can be extended with polynomial features)
5. **Multiclass**: One-vs-All or Softmax approaches
6. **Regularization**: Prevents overfitting, especially with many features
7. **Evaluation**: Use accuracy, precision, recall, F1-score, ROC-AUC

---

## Interview Tips

When implementing logistic regression in interviews:

1. **Derive the Gradient**: Show you understand the math
   - Start with cost function: `J(θ) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]`
   - Apply chain rule: `∂J/∂θ = ∂J/∂h * ∂h/∂z * ∂z/∂θ`
   - Result: `(1/m) * Xᵀ(h - y)`

2. **Handle Numerical Stability**:
   - Clip sigmoid input to prevent overflow
   - Add small epsilon (1e-15) to log arguments

3. **Explain Decision Boundary**:
   - Linear: `θᵀx = 0`
   - Can be extended with polynomial features

4. **Compare with Linear Regression**:
   - Same gradient form, different hypothesis function
   - Different cost function (log loss vs MSE)

5. **Multiclass Approaches**:
   - One-vs-All: Train k binary classifiers
   - Softmax: Single model, more elegant

---

## Time Complexity

- **Gradient Descent**: O(k * m * n) where k = iterations, m = samples, n = features
- **SGD**: O(m * n) per epoch
- **One-vs-All**: O(k * m * n * c) where c = number of classes
- **Softmax**: O(k * m * n * c) but more efficient than One-vs-All

---

## Common Pitfalls

1. **Numerical Overflow**: Always clip sigmoid input
2. **Log of Zero**: Add small epsilon to prevent `log(0)`
3. **Feature Scaling**: Important for convergence
4. **Learning Rate**: Too high can cause divergence
5. **Imbalanced Classes**: Consider class weights or different thresholds

---

## Extensions

1. **Polynomial Features**: Create non-linear decision boundaries
2. **Regularization**: L1 (Lasso) or L2 (Ridge)
3. **Class Weights**: Handle imbalanced datasets
4. **Early Stopping**: Prevent overfitting
5. **Feature Selection**: Use L1 regularization

---

## References

- Sigmoid function and its properties
- Cross-entropy loss derivation
- Gradient calculation with chain rule
- Multiclass classification strategies
- Evaluation metrics for classification

