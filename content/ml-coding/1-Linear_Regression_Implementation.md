+++
title = "Linear Regression Implementation from Scratch"
date = 2025-01-15T10:00:00+05:30
draft = false
weight = 1
description = "Complete implementation of linear regression from scratch using Python and NumPy. Covers gradient descent, normal equation, regularization (Ridge, Lasso), feature scaling, and practical examples."
+++

---

## Introduction

Linear Regression is one of the most fundamental algorithms in machine learning. It models the relationship between a dependent variable (target) and one or more independent variables (features) using a linear equation.

In this guide, we'll implement linear regression from scratch using Python and NumPy, covering multiple approaches and variations.

---

## Mathematical Foundation

### Hypothesis Function

For a single feature:
```
h(x) = θ₀ + θ₁x
```

For multiple features (multivariate):
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

In vectorized form:
```
h(x) = θᵀx
```

Where:
- `θ₀` is the bias term (intercept)
- `θ₁, θ₂, ..., θₙ` are the weights (coefficients)
- `x₁, x₂, ..., xₙ` are the feature values

### Cost Function (Mean Squared Error)

```
J(θ) = (1/2m) * Σ(h(xᵢ) - yᵢ)²
```

Where:
- `m` is the number of training examples
- `h(xᵢ)` is the predicted value
- `yᵢ` is the actual value

In vectorized form:
```
J(θ) = (1/2m) * (Xθ - y)ᵀ(Xθ - y)
```

### Gradient of Cost Function

Partial derivative with respect to each parameter:

```
∂J/∂θⱼ = (1/m) * Σ(h(xᵢ) - yᵢ) * xᵢⱼ
```

In vectorized form:
```
∇J(θ) = (1/m) * Xᵀ(Xθ - y)
```

---

## Implementation 1: Gradient Descent

### Basic Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize Linear Regression model.
        
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
    
    def _add_bias(self, X):
        """Add bias term (column of ones) to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        # Add bias term
        X = self._add_bias(X)
        m, n = X.shape
        
        # Initialize parameters (theta) with zeros
        self.theta = np.zeros(n)
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            predictions = X.dot(self.theta)
            
            # Calculate cost
            error = predictions - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            # Calculate gradient
            gradient = (1 / m) * X.T.dot(error)
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Optional: Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Cost = {cost:.4f}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        predictions : numpy array
            Predicted values of shape (m,)
        """
        X = self._add_bias(X)
        return X.dot(self.theta)
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        
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
        r2 = 1 - (ss_res / ss_tot)
        return r2
```

### Usage Example

```python
# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 2 * X.flatten() + 3 + np.random.randn(100) * 2

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate
r2_score = model.score(X, y)
print(f"R² Score: {r2_score:.4f}")
print(f"Learned parameters: θ₀ = {model.theta[0]:.4f}, θ₁ = {model.theta[1]:.4f}")

# Plot cost history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(model.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History')
plt.grid(True)

# Plot predictions
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Actual')
plt.plot(X, predictions, 'r-', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Fit')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Implementation 2: Normal Equation (Closed-Form Solution)

The normal equation provides an analytical solution without iteration:

```
θ = (XᵀX)⁻¹Xᵀy
```

### Implementation

```python
class LinearRegressionNormal:
    def __init__(self):
        """Initialize Linear Regression model using normal equation."""
        self.theta = None
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """
        Train the model using normal equation.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        y : numpy array
            Target vector of shape (m,)
        """
        X = self._add_bias(X)
        
        # Normal equation: θ = (XᵀX)⁻¹Xᵀy
        XTX = X.T.dot(X)
        
        # Check if matrix is invertible
        if np.linalg.det(XTX) == 0:
            raise ValueError("Matrix XᵀX is singular (not invertible). "
                           "Try using gradient descent or add regularization.")
        
        self.theta = np.linalg.inv(XTX).dot(X.T).dot(y)
    
    def predict(self, X):
        """Make predictions on new data."""
        X = self._add_bias(X)
        return X.dot(self.theta)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
```

### When to Use Normal Equation vs Gradient Descent

| Aspect | Normal Equation | Gradient Descent |
|--------|-----------------|------------------|
| **Time Complexity** | O(n³) | O(kn²) where k = iterations |
| **Works for large n?** | Slow if n > 10,000 | Yes, scales well |
| **Need to choose α?** | No | Yes (learning rate) |
| **Need iterations?** | No | Yes |
| **Works with non-invertible?** | No | Yes |

**Rule of thumb**: Use normal equation when n < 10,000, otherwise use gradient descent.

---

## Implementation 3: Ridge Regression (L2 Regularization)

Ridge regression adds L2 penalty to prevent overfitting:

```
J(θ) = (1/2m) * Σ(h(xᵢ) - yᵢ)² + λ * Σθⱼ²
```

Where `λ` (lambda) is the regularization parameter.

### Implementation

```python
class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.1):
        """
        Initialize Ridge Regression model.
        
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
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """
        Train the Ridge Regression model.
        
        Note: We don't regularize θ₀ (bias term).
        """
        X = self._add_bias(X)
        m, n = X.shape
        
        # Initialize parameters
        self.theta = np.zeros(n)
        
        for i in range(self.n_iterations):
            # Predictions
            predictions = X.dot(self.theta)
            
            # Calculate cost (with regularization)
            error = predictions - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            
            # Add regularization term (excluding bias θ₀)
            reg_term = (self.lambda_reg / (2 * m)) * np.sum(self.theta[1:] ** 2)
            cost += reg_term
            self.cost_history.append(cost)
            
            # Calculate gradient
            gradient = (1 / m) * X.T.dot(error)
            
            # Add regularization to gradient (excluding bias)
            gradient[1:] += (self.lambda_reg / m) * self.theta[1:]
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
    
    def predict(self, X):
        """Make predictions on new data."""
        X = self._add_bias(X)
        return X.dot(self.theta)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
```

---

## Implementation 4: Lasso Regression (L1 Regularization)

Lasso regression uses L1 penalty, which can drive some coefficients to exactly zero (feature selection):

```
J(θ) = (1/2m) * Σ(h(xᵢ) - yᵢ)² + λ * Σ|θⱼ|
```

### Implementation

```python
class LassoRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.1):
        """
        Initialize Lasso Regression model.
        
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
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def _soft_threshold(self, x, threshold):
        """
        Soft thresholding function for L1 regularization.
        sign(x) * max(|x| - threshold, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        """
        Train the Lasso Regression model using coordinate descent.
        
        Note: L1 regularization is non-differentiable, so we use
        coordinate descent or subgradient method.
        """
        X = self._add_bias(X)
        m, n = X.shape
        
        # Initialize parameters
        self.theta = np.zeros(n)
        
        for i in range(self.n_iterations):
            # Predictions
            predictions = X.dot(self.theta)
            
            # Calculate cost
            error = predictions - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            
            # Add L1 regularization term (excluding bias)
            reg_term = (self.lambda_reg / m) * np.sum(np.abs(self.theta[1:]))
            cost += reg_term
            self.cost_history.append(cost)
            
            # Coordinate descent: update each parameter separately
            for j in range(n):
                # Calculate residual without feature j
                residual = y - (X.dot(self.theta) - X[:, j] * self.theta[j])
                
                # Update θⱼ
                if j == 0:  # Bias term (no regularization)
                    self.theta[j] = np.mean(residual)
                else:  # Regularized features
                    # Soft thresholding
                    threshold = self.lambda_reg / m
                    z_j = np.mean(X[:, j] * residual)
                    self.theta[j] = self._soft_threshold(z_j, threshold)
    
    def predict(self, X):
        """Make predictions on new data."""
        X = self._add_bias(X)
        return X.dot(self.theta)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
```

---

## Feature Scaling

Feature scaling is crucial for gradient descent to converge faster. Two common methods:

### 1. Standardization (Z-score normalization)

```
x_scaled = (x - μ) / σ
```

### 2. Min-Max Scaling

```
x_scaled = (x - min) / (max - min)
```

### Implementation

```python
class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """Compute mean and std for later scaling."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Perform standardization."""
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

class MinMaxScaler:
    """Transform features by scaling each feature to a given range."""
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        """Compute min and max for later scaling."""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        """Perform min-max scaling."""
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        # Scale to feature_range
        min_range, max_range = self.feature_range
        return X_scaled * (max_range - min_range) + min_range
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
```

### Updated Linear Regression with Scaling

```python
class LinearRegressionScaled(LinearRegression):
    """Linear Regression with automatic feature scaling."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, scale_features=True):
        super().__init__(learning_rate, n_iterations)
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
    
    def fit(self, X, y):
        """Train model with optional feature scaling."""
        if self.scale_features:
            X = self.scaler.fit_transform(X)
        super().fit(X, y)
    
    def predict(self, X):
        """Make predictions with feature scaling."""
        if self.scale_features:
            X = self.scaler.transform(X)
        return super().predict(X)
```

---

## Advanced: Stochastic Gradient Descent (SGD)

SGD updates parameters using a single training example at a time, making it faster for large datasets:

```python
class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        """
        Initialize Linear Regression with Stochastic Gradient Descent.
        
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
                prediction = x_i.dot(self.theta)
                
                # Calculate error
                error = prediction - y_i
                
                # Update parameters (using single example)
                self.theta -= self.learning_rate * error * x_i
                
                # Accumulate cost
                epoch_cost += error ** 2
            
            # Average cost for this epoch
            avg_cost = epoch_cost / (2 * m)
            self.cost_history.append(avg_cost)
    
    def predict(self, X):
        """Make predictions on new data."""
        X = self._add_bias(X)
        return X.dot(self.theta)
```

---

## Advanced: Mini-Batch Gradient Descent

Mini-batch GD uses small batches of data, combining benefits of batch and stochastic GD:

```python
class LinearRegressionMiniBatch:
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=32):
        """
        Initialize Linear Regression with Mini-Batch Gradient Descent.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_epochs : int
            Number of epochs
        batch_size : int
            Size of each mini-batch
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.theta = None
        self.cost_history = []
    
    def _add_bias(self, X):
        """Add bias term to feature matrix."""
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack([bias, X])
    
    def fit(self, X, y):
        """Train model using Mini-Batch Gradient Descent."""
        X = self._add_bias(X)
        m, n = X.shape
        
        # Initialize parameters
        self.theta = np.zeros(n)
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            # Process in mini-batches
            for i in range(0, m, self.batch_size):
                # Get mini-batch
                end_idx = min(i + self.batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                batch_size_actual = X_batch.shape[0]
                
                # Predictions for batch
                predictions = X_batch.dot(self.theta)
                
                # Calculate error
                error = predictions - y_batch
                
                # Calculate gradient for batch
                gradient = (1 / batch_size_actual) * X_batch.T.dot(error)
                
                # Update parameters
                self.theta -= self.learning_rate * gradient
                
                # Accumulate cost
                epoch_cost += np.sum(error ** 2)
            
            # Average cost for this epoch
            avg_cost = epoch_cost / (2 * m)
            self.cost_history.append(avg_cost)
    
    def predict(self, X):
        """Make predictions on new data."""
        X = self._add_bias(X)
        return X.dot(self.theta)
```

---

## Complete Example: Comparison of Methods

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 3) * 5
true_theta = np.array([2.5, 1.8, -0.9, 3.2])  # [bias, w1, w2, w3]
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
y = X_with_bias.dot(true_theta) + np.random.randn(n_samples) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare different methods
methods = {
    'Gradient Descent': LinearRegression(learning_rate=0.01, n_iterations=1000),
    'Normal Equation': LinearRegressionNormal(),
    'Ridge Regression': RidgeRegression(learning_rate=0.01, n_iterations=1000, lambda_reg=0.1),
    'SGD': LinearRegressionSGD(learning_rate=0.01, n_epochs=100),
    'Mini-Batch GD': LinearRegressionMiniBatch(learning_rate=0.01, n_epochs=100, batch_size=32)
}

results = {}

for name, model in methods.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    
    results[name] = {
        'MSE': mse,
        'R²': r2,
        'theta': model.theta
    }
    
    print(f"\n{name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Parameters: {model.theta}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Cost history for iterative methods
iterative_methods = ['Gradient Descent', 'Ridge Regression', 'SGD', 'Mini-Batch GD']
for i, name in enumerate(iterative_methods):
    if name in methods and hasattr(methods[name], 'cost_history'):
        ax = axes[i // 2, i % 2]
        ax.plot(methods[name].cost_history)
        ax.set_title(f'{name} - Cost History')
        ax.set_xlabel('Iteration/Epoch')
        ax.set_ylabel('Cost')
        ax.grid(True)

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

1. **Gradient Descent**: Iterative optimization, works for any dataset size
2. **Normal Equation**: Analytical solution, fast for small datasets (n < 10,000)
3. **Ridge Regression**: L2 regularization prevents overfitting, shrinks coefficients
4. **Lasso Regression**: L1 regularization performs feature selection, can zero out coefficients
5. **Feature Scaling**: Essential for gradient descent convergence
6. **SGD/Mini-Batch**: Faster for large datasets, more noisy but can escape local minima

---

## Interview Tips

When implementing linear regression in interviews:

1. **Start Simple**: Begin with basic gradient descent
2. **Explain Math**: Derive the cost function and gradient
3. **Handle Edge Cases**: Empty data, single feature, perfect correlation
4. **Optimize**: Use vectorization, consider feature scaling
5. **Extend**: Add regularization, different optimizers
6. **Test**: Validate with known examples

---

## Time Complexity

- **Gradient Descent**: O(k * m * n) where k = iterations, m = samples, n = features
- **Normal Equation**: O(n³) for matrix inversion
- **SGD**: O(m * n) per epoch
- **Mini-Batch**: O(b * n) per batch where b = batch_size

---

## References

- Cost function derivation
- Gradient calculation
- Regularization effects
- Optimization algorithms

