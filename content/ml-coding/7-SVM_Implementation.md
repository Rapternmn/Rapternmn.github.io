+++
title = "Support Vector Machines (SVM) Implementation from Scratch"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Complete implementation of Support Vector Machines from scratch using Python and NumPy. Covers linear SVM, soft margin, kernel trick (RBF, polynomial), hinge loss, and practical examples."
+++

---

## Introduction

Support Vector Machines (SVM) is a powerful supervised learning algorithm used for classification and regression. SVM finds the optimal hyperplane that separates classes with the maximum margin. It can handle both linear and non-linear classification using the kernel trick.

In this guide, we'll implement SVM from scratch using Python and NumPy, covering linear SVM, soft margin, and kernel methods.

---

## Mathematical Foundation

### Linear SVM (Hard Margin)

For linearly separable data, SVM finds the hyperplane that maximizes the margin between classes.

**Hyperplane equation:**
```
wᵀx + b = 0
```

**Decision function:**
```
f(x) = sign(wᵀx + b)
```

**Margin:**
```
margin = 2 / ||w||
```

**Optimization problem (primal):**
```
minimize: (1/2) ||w||²
subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i
```

### Soft Margin SVM

For non-separable data, we introduce slack variables ξᵢ:

```
minimize: (1/2) ||w||² + C * Σᵢ ξᵢ
subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

Where:
- `C` is the regularization parameter (larger C = harder margin)
- `ξᵢ` are slack variables allowing misclassification

### Hinge Loss

The hinge loss function:

```
L(y, f(x)) = max(0, 1 - y * f(x))
```

For correctly classified points (y * f(x) ≥ 1): loss = 0
For misclassified points: loss increases linearly

### Dual Form (Lagrangian)

The dual optimization problem:

```
maximize: Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ xᵢᵀxⱼ
subject to: 0 ≤ αᵢ ≤ C, Σᵢ αᵢ yᵢ = 0
```

Where:
- `αᵢ` are Lagrange multipliers (dual variables)
- Support vectors have αᵢ > 0

**Weight vector:**
```
w = Σᵢ αᵢ yᵢ xᵢ
```

**Bias:**
```
b = (1/|S|) Σᵢ∈S (yᵢ - wᵀxᵢ)
```

Where S is the set of support vectors.

### Kernel Trick

For non-linear classification, we map data to higher dimensions:

```
K(xᵢ, xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)
```

Common kernels:

**Linear:**
```
K(xᵢ, xⱼ) = xᵢᵀ xⱼ
```

**Polynomial:**
```
K(xᵢ, xⱼ) = (γ xᵢᵀ xⱼ + r)ᵈ
```

**RBF (Radial Basis Function):**
```
K(xᵢ, xⱼ) = exp(-γ ||xᵢ - xⱼ||²)
```

**Sigmoid:**
```
K(xᵢ, xⱼ) = tanh(γ xᵢᵀ xⱼ + r)
```

---

## Implementation 1: Linear SVM (Simplified Gradient Descent)

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        """
        Initialize Linear SVM.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        lambda_param : float
            Regularization parameter (equivalent to 1/C)
        n_iterations : int
            Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        """
        Train the Linear SVM using gradient descent on hinge loss.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,) with values {-1, 1}
        """
        # Ensure labels are -1 or 1
        y = np.where(y <= 0, -1, 1)
        
        m, n = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n)
        self.b = 0
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            for i, x_i in enumerate(X):
                condition = y[i] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Correctly classified: update only regularization term
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified: update weights and bias
                    self.w -= self.learning_rate * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y[i])
                    )
                    self.b -= self.learning_rate * y[i]
    
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
        predictions = np.sign(np.dot(X, self.w) - self.b)
        # Convert -1/1 back to 0/1 if needed
        return np.where(predictions <= 0, 0, 1)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

### Usage Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate linearly separable data
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train model
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
svm.fit(X_train, y_train)

# Evaluate
accuracy = svm.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Implementation 2: SVM with Kernel Support

```python
class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma='scale', degree=3, coef0=0.0, max_iter=1000):
        """
        Initialize SVM with kernel support.
        
        Parameters:
        -----------
        C : float
            Regularization parameter (inverse of lambda)
        kernel : str
            Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
        gamma : float or 'scale'
            Kernel coefficient for 'rbf', 'poly', 'sigmoid'
        degree : int
            Degree for polynomial kernel
        coef0 : float
            Independent term in polynomial/sigmoid kernel
        max_iter : int
            Maximum number of iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None  # α * y
        self.intercept_ = None
        self.X_train = None
        self.y_train = None
    
    def _kernel_function(self, x1, x2):
        """
        Compute kernel function between two vectors.
        
        Parameters:
        -----------
        x1, x2 : numpy arrays
            Feature vectors
        
        Returns:
        --------
        kernel_value : float
            Kernel value
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1, X2):
        """
        Compute kernel matrix between two sets of vectors.
        
        Parameters:
        -----------
        X1, X2 : numpy arrays
            Feature matrices
        
        Returns:
        --------
        K : numpy array
            Kernel matrix of shape (len(X1), len(X2))
        """
        K = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = self._kernel_function(x1, x2)
        return K
    
    def fit(self, X, y):
        """
        Train SVM using simplified SMO (Sequential Minimal Optimization) algorithm.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,) with values {-1, 1}
        """
        # Ensure labels are -1 or 1
        y = np.where(y <= 0, -1, 1)
        
        m, n = X.shape
        self.X_train = X
        self.y_train = y
        
        # Set gamma if 'scale'
        if self.gamma == 'scale':
            self.gamma = 1.0 / (n * X.var())
        elif self.gamma == 'auto':
            self.gamma = 1.0 / n
        
        # Initialize dual variables (Lagrange multipliers)
        alpha = np.zeros(m)
        
        # Simplified SMO algorithm
        for iteration in range(self.max_iter):
            alpha_prev = alpha.copy()
            
            for i in range(m):
                # Calculate prediction for sample i
                E_i = self._predict_single(X[i], alpha, X, y) - y[i]
                
                # Check KKT conditions
                if (y[i] * E_i < -0.01 and alpha[i] < self.C) or \
                   (y[i] * E_i > 0.01 and alpha[i] > 0):
                    
                    # Select second variable j randomly
                    j = np.random.randint(0, m)
                    if j == i:
                        continue
                    
                    E_j = self._predict_single(X[j], alpha, X, y) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta (second derivative)
                    K_ii = self._kernel_function(X[i], X[i])
                    K_jj = self._kernel_function(X[j], X[j])
                    K_ij = self._kernel_function(X[i], X[j])
                    eta = 2 * K_ij - K_ii - K_jj
                    
                    if eta >= 0:
                        continue
                    
                    # Update alpha[j]
                    alpha[j] = alpha[j] - (y[j] * (E_i - E_j)) / eta
                    
                    # Clip alpha[j]
                    alpha[j] = max(L, min(H, alpha[j]))
                    
                    # Update alpha[i]
                    alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
            
            # Check convergence
            if np.linalg.norm(alpha - alpha_prev) < 1e-3:
                break
        
        # Find support vectors (alpha > 0)
        support_vector_indices = alpha > 1e-5
        self.support_vectors_ = X[support_vector_indices]
        self.support_vector_labels_ = y[support_vector_indices]
        self.dual_coef_ = alpha[support_vector_indices] * y[support_vector_indices]
        
        # Calculate bias
        self.intercept_ = 0
        for i in range(len(self.support_vectors_)):
            self.intercept_ += self.support_vector_labels_[i]
            self.intercept_ -= np.sum(
                self.dual_coef_ * self._compute_kernel_matrix(
                    [self.support_vectors_[i]], self.support_vectors_
                )[0]
            )
        self.intercept_ /= len(self.support_vectors_)
    
    def _predict_single(self, x, alpha, X, y):
        """Helper function to predict a single sample during training."""
        prediction = 0
        for i in range(len(X)):
            prediction += alpha[i] * y[i] * self._kernel_function(X[i], x)
        return prediction
    
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
        predictions = []
        for x in X:
            prediction = 0
            for i in range(len(self.support_vectors_)):
                prediction += self.dual_coef_[i] * self._kernel_function(
                    self.support_vectors_[i], x
                )
            prediction += self.intercept_
            predictions.append(np.sign(prediction))
        
        # Convert -1/1 to 0/1
        return np.where(np.array(predictions) <= 0, 0, 1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (distance to hyperplane as proxy).
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        probabilities : numpy array
            Class probabilities of shape (m, 2)
        """
        # Calculate decision function values
        decision_values = []
        for x in X:
            value = 0
            for i in range(len(self.support_vectors_)):
                value += self.dual_coef_[i] * self._kernel_function(
                    self.support_vectors_[i], x
                )
            value += self.intercept_
            decision_values.append(value)
        
        decision_values = np.array(decision_values)
        
        # Convert to probabilities using sigmoid
        # P(y=1|x) = 1 / (1 + exp(-decision_value))
        prob_positive = 1 / (1 + np.exp(-decision_values))
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

### Usage Example

```python
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Example 1: Linear SVM
print("=== Linear SVM ===")
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_linear = SVM(C=1.0, kernel='linear')
svm_linear.fit(X_train, y_train)

accuracy = svm_linear.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"Number of support vectors: {len(svm_linear.support_vectors_)}")

# Example 2: RBF Kernel for non-linear data
print("\n=== RBF Kernel SVM ===")
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_rbf = SVM(C=1.0, kernel='rbf', gamma='scale')
svm_rbf.fit(X_train, y_train)

accuracy = svm_rbf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"Number of support vectors: {len(svm_rbf.support_vectors_)}")
```

---

## Implementation 3: Optimized SVM with Vectorized Kernels

```python
class SVMOptimized:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, coef0=0.0, max_iter=1000):
        """Initialize optimized SVM."""
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.X_train = None
        self.y_train = None
    
    def _rbf_kernel_vectorized(self, X1, X2):
        """
        Vectorized RBF kernel computation.
        
        Parameters:
        -----------
        X1, X2 : numpy arrays
            Feature matrices
        
        Returns:
        --------
        K : numpy array
            Kernel matrix
        """
        # ||x1 - x2||² = ||x1||² + ||x2||² - 2*x1*x2
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        K = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * K)
    
    def _linear_kernel_vectorized(self, X1, X2):
        """Vectorized linear kernel."""
        return np.dot(X1, X2.T)
    
    def _poly_kernel_vectorized(self, X1, X2):
        """Vectorized polynomial kernel."""
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
    
    def _compute_kernel_matrix(self, X1, X2):
        """Compute kernel matrix using vectorized operations."""
        if self.kernel == 'linear':
            return self._linear_kernel_vectorized(X1, X2)
        elif self.kernel == 'poly':
            return self._poly_kernel_vectorized(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel_vectorized(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Train SVM (simplified version for demonstration).
        For production, use libraries like sklearn or libsvm.
        """
        y = np.where(y <= 0, -1, 1)
        m, n = X.shape
        self.X_train = X
        self.y_train = y
        
        if self.gamma == 'scale':
            self.gamma = 1.0 / (n * X.var())
        elif self.gamma == 'auto':
            self.gamma = 1.0 / n
        
        # Simplified training (full SMO is more complex)
        # This is a placeholder - full implementation would use proper SMO
        alpha = np.random.rand(m) * 0.1  # Simplified initialization
        
        # Compute kernel matrix once
        K = self._compute_kernel_matrix(X, X)
        
        # Simplified optimization (full SMO needed for production)
        for iteration in range(self.max_iter):
            # Gradient of dual objective
            # ∇L = 1 - y * (K @ (alpha * y))
            gradient = 1 - y * (K @ (alpha * y))
            
            # Projected gradient descent
            alpha = alpha + 0.01 * gradient * y
            alpha = np.clip(alpha, 0, self.C)
            
            # Enforce constraint: Σ αᵢ yᵢ = 0
            alpha = alpha - (np.sum(alpha * y) / np.sum(y ** 2)) * y
        
        # Find support vectors
        support_vector_indices = alpha > 1e-5
        self.support_vectors_ = X[support_vector_indices]
        self.support_vector_labels_ = y[support_vector_indices]
        self.dual_coef_ = alpha[support_vector_indices] * y[support_vector_indices]
        
        # Calculate bias
        decision_values = self._compute_kernel_matrix(
            self.support_vectors_, self.support_vectors_
        ) @ self.dual_coef_
        self.intercept_ = np.mean(self.support_vector_labels_ - decision_values)
    
    def predict(self, X):
        """Predict class labels (vectorized)."""
        # Compute kernel between test and support vectors
        K_test = self._compute_kernel_matrix(X, self.support_vectors_)
        
        # Decision function: K_test @ dual_coef + intercept
        decision_values = K_test @ self.dual_coef_ + self.intercept_
        
        return np.where(decision_values <= 0, 0, 1)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## Complete Example: Comparison of Kernels

```python
from sklearn.datasets import make_classification, make_circles, make_moons
import matplotlib.pyplot as plt

def compare_kernels(X, y, title):
    """Compare different kernel types."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    kernels = ['linear', 'poly', 'rbf']
    results = {}
    
    for kernel in kernels:
        svm = SVM(C=1.0, kernel=kernel, gamma='scale', max_iter=1000)
        svm.fit(X_train, y_train)
        accuracy = svm.score(X_test, y_test)
        results[kernel] = {
            'accuracy': accuracy,
            'n_support_vectors': len(svm.support_vectors_)
        }
        print(f"{kernel.upper()} Kernel: Accuracy = {accuracy:.4f}, "
              f"Support Vectors = {results[kernel]['n_support_vectors']}")
    
    return results

# Test on different datasets
print("=== Linearly Separable Data ===")
X1, y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, random_state=42)
compare_kernels(X1, y1, "Linearly Separable")

print("\n=== Non-linear Data (Circles) ===")
X2, y2 = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
compare_kernels(X2, y2, "Circles")

print("\n=== Non-linear Data (Moons) ===")
X3, y3 = make_moons(n_samples=200, noise=0.1, random_state=42)
compare_kernels(X3, y3, "Moons")
```

---

## Key Takeaways

1. **Maximum Margin**: SVM finds the hyperplane with maximum margin between classes
2. **Support Vectors**: Only training points near the decision boundary matter
3. **Kernel Trick**: Maps data to higher dimensions without explicit transformation
4. **Regularization**: C parameter controls tradeoff between margin and misclassification
5. **Kernel Selection**:
   - Linear: For linearly separable data
   - RBF: Most versatile, good default
   - Polynomial: For specific polynomial relationships
6. **Sparse Solution**: Only support vectors are needed for prediction

---

## When to Use SVM

| Scenario | Suitable? | Notes |
|----------|-----------|-------|
| **High-dimensional data** | ✅ Yes | Works well with many features |
| **Non-linear boundaries** | ✅ Yes | With appropriate kernel |
| **Small to medium datasets** | ✅ Yes | Memory efficient (only stores support vectors) |
| **Large datasets** | ❌ No | Training becomes slow |
| **Multi-class** | ⚠️ Maybe | Requires one-vs-one or one-vs-rest |
| **Probabilistic output** | ⚠️ Maybe | Can use Platt scaling |
| **Interpretability** | ❌ No | Less interpretable than linear models |

---

## Interview Tips

When implementing SVM in interviews:

1. **Explain Maximum Margin**: Why maximizing margin improves generalization
2. **Discuss Support Vectors**: Only points near boundary matter
3. **Kernel Trick**: Explain how it enables non-linear classification
4. **C Parameter**: Larger C = harder margin, smaller C = softer margin
5. **Dual Form**: Understand why we solve dual instead of primal
6. **SMO Algorithm**: Mention Sequential Minimal Optimization (simplified version shown)
7. **Time Complexity**: O(m²) to O(m³) for training, O(n_sv) for prediction
8. **Edge Cases**: Handle ties, degenerate cases, numerical stability

---

## Time Complexity

- **Training**: 
  - Linear: O(m * n) with optimized algorithms
  - Non-linear: O(m² * n) to O(m³) depending on kernel
- **Prediction**: O(n_sv * n) where n_sv = number of support vectors
- **Space Complexity**: O(n_sv * n) to store support vectors

---

## Advantages

1. **Effective in High Dimensions**: Works well with many features
2. **Memory Efficient**: Only stores support vectors
3. **Versatile**: Can handle linear and non-linear problems
4. **Robust**: Less prone to overfitting with appropriate C
5. **Theoretically Sound**: Based on solid statistical learning theory

---

## Disadvantages

1. **Slow Training**: Especially for large datasets
2. **Kernel Selection**: Choosing right kernel and parameters can be tricky
3. **No Probabilistic Output**: Doesn't directly provide probabilities
4. **Sensitive to Feature Scaling**: Requires normalization
5. **Black Box**: Less interpretable than linear models
6. **Memory**: Kernel matrix can be large for big datasets

---

## Hyperparameter Tuning

### C Parameter

```python
from sklearn.model_selection import GridSearchCV

# Test different C values
C_values = [0.1, 1, 10, 100, 1000]
best_score = 0
best_C = None

for C in C_values:
    svm = SVM(C=C, kernel='rbf', gamma='scale')
    svm.fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_C = C
    print(f"C={C}: Accuracy = {score:.4f}")

print(f"\nBest C: {best_C} with accuracy: {best_score:.4f}")
```

### Gamma Parameter (for RBF)

```python
# Test different gamma values
gamma_values = [0.001, 0.01, 0.1, 1, 10]
best_score = 0
best_gamma = None

for gamma in gamma_values:
    svm = SVM(C=1.0, kernel='rbf', gamma=gamma)
    svm.fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_gamma = gamma
    print(f"gamma={gamma}: Accuracy = {score:.4f}")

print(f"\nBest gamma: {best_gamma} with accuracy: {best_score:.4f}")
```

---

## References

* Maximum margin classification
* Kernel methods and the kernel trick
* Sequential Minimal Optimization (SMO) algorithm
* Support vectors and margin
* Hinge loss and regularization
* Statistical learning theory

