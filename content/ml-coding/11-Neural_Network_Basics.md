+++
title = "Neural Network Basics"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 11
description = "Complete implementation of feedforward neural networks from scratch using Python and NumPy. Covers forward propagation, backpropagation, activation functions, gradient descent, and multi-layer networks."
+++

---

## Introduction

Neural Networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that can learn complex patterns from data through a process called backpropagation.

In this guide, we'll implement a multi-layer feedforward neural network from scratch using Python and NumPy, covering forward propagation, backpropagation, various activation functions, and optimization techniques.

---

## Mathematical Foundation

### Single Neuron (Perceptron)

A neuron computes:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
a = σ(z)
```

Where:
- `z` is the weighted sum (pre-activation)
- `a` is the output (activation)
- `σ` is the activation function
- `w` are weights
- `b` is bias

### Multi-Layer Network

For a network with L layers:

**Forward Propagation:**

```
z[l] = W[l] * a[l-1] + b[l]
a[l] = σ(z[l])
```

Where:
- `a[0] = X` (input)
- `a[L] = ŷ` (output)

### Loss Function

**Mean Squared Error (MSE):**
```
L = (1/2m) * Σ(y - ŷ)²
```

**Cross-Entropy Loss (for classification):**
```
L = -(1/m) * Σ[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

### Backpropagation

**Output Layer:**
```
∂L/∂a[L] = (a[L] - y) / m
∂L/∂z[L] = ∂L/∂a[L] * σ'(z[L])
```

**Hidden Layers (l = L-1, L-2, ..., 1):**
```
∂L/∂a[l] = (W[l+1])ᵀ * ∂L/∂z[l+1]
∂L/∂z[l] = ∂L/∂a[l] * σ'(z[l])
```

**Weight and Bias Gradients:**
```
∂L/∂W[l] = (1/m) * ∂L/∂z[l] * (a[l-1])ᵀ
∂L/∂b[l] = (1/m) * Σ ∂L/∂z[l]
```

**Weight Updates:**
```
W[l] = W[l] - α * ∂L/∂W[l]
b[l] = b[l] - α * ∂L/∂b[l]
```

Where `α` is the learning rate.

---

## Activation Functions

### Sigmoid

```
σ(z) = 1 / (1 + e^(-z))
σ'(z) = σ(z) * (1 - σ(z))
```

### Tanh

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
tanh'(z) = 1 - tanh²(z)
```

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
ReLU'(z) = {1 if z > 0, 0 if z ≤ 0}
```

### Leaky ReLU

```
LeakyReLU(z) = max(0.01z, z)
LeakyReLU'(z) = {1 if z > 0, 0.01 if z ≤ 0}
```

---

## Implementation 1: Basic Neural Network

```python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid', learning_rate=0.01):
        """
        Initialize Neural Network.
        
        Parameters:
        -----------
        layers : list
            List of integers representing neurons in each layer
            e.g., [2, 4, 3, 1] for 2 input, 4 hidden, 3 hidden, 1 output
        activation : str
            Activation function: 'sigmoid', 'tanh', 'relu', 'leaky_relu'
        learning_rate : float
            Learning rate for gradient descent
        """
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # He initialization for better convergence
        for i in range(1, self.num_layers):
            # Weight matrix: (neurons_current, neurons_previous)
            w = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2.0 / layers[i-1])
            b = np.zeros((layers[i], 1))
            self.weights.append(w)
            self.biases.append(b)
        
        # Store activations and pre-activations for backprop
        self.activations = []
        self.z_values = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    def _sigmoid_derivative(self, z):
        """Derivative of sigmoid."""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _tanh(self, z):
        """Tanh activation function."""
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        """Derivative of tanh."""
        return 1 - np.tanh(z) ** 2
    
    def _relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)
    
    def _leaky_relu(self, z, alpha=0.01):
        """Leaky ReLU activation function."""
        return np.where(z > 0, z, alpha * z)
    
    def _leaky_relu_derivative(self, z, alpha=0.01):
        """Derivative of Leaky ReLU."""
        return np.where(z > 0, 1, alpha)
    
    def _activate(self, z):
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        elif self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'leaky_relu':
            return self._leaky_relu(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activate_derivative(self, z):
        """Apply activation function derivative."""
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return self._tanh_derivative(z)
        elif self.activation == 'relu':
            return self._relu_derivative(z)
        elif self.activation == 'leaky_relu':
            return self._leaky_relu_derivative(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, X):
        """
        Forward propagation.
        
        Parameters:
        -----------
        X : numpy array
            Input data of shape (m, n_features)
        
        Returns:
        --------
        output : numpy array
            Network output of shape (m, n_output)
        """
        # Store for backprop
        self.activations = [X.T]  # Transpose for easier matrix operations
        self.z_values = []
        
        # Forward pass through each layer
        a = X.T
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            self.z_values.append(z)
            a = self._activate(z)
            self.activations.append(a)
        
        return a.T  # Transpose back
    
    def backward(self, X, y, output):
        """
        Backward propagation.
        
        Parameters:
        -----------
        X : numpy array
            Input data
        y : numpy array
            True labels
        output : numpy array
            Network output from forward pass
        """
        m = X.shape[0]
        
        # Gradients for weights and biases
        dW = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        
        # Output layer error
        # For binary classification with sigmoid output
        if self.activation == 'sigmoid':
            dz = (output.T - y.T) / m
        else:
            # For other activations, use MSE derivative
            dz = (output.T - y.T) / m
        
        # Backpropagate through layers
        for l in range(self.num_layers - 2, -1, -1):
            # Gradient for weights
            dW[l] = np.dot(dz, self.activations[l].T)
            # Gradient for biases
            db[l] = np.sum(dz, axis=1, keepdims=True)
            
            # Error for previous layer
            if l > 0:
                dz = np.dot(self.weights[l].T, dz)
                dz = dz * self._activate_derivative(self.z_values[l-1])
        
        # Update weights and biases
        for l in range(self.num_layers - 1):
            self.weights[l] -= self.learning_rate * dW[l]
            self.biases[l] -= self.learning_rate * db[l]
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute loss (MSE for regression, cross-entropy for classification).
        
        Parameters:
        -----------
        y_true : numpy array
            True labels
        y_pred : numpy array
            Predicted labels
        
        Returns:
        --------
        loss : float
            Loss value
        """
        m = y_true.shape[0]
        if self.activation == 'sigmoid':
            # Cross-entropy loss
            epsilon = 1e-15  # Small value to prevent log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # MSE loss
            loss = np.mean((y_true - y_pred) ** 2)
        return loss
    
    def fit(self, X, y, epochs=1000, verbose=True, print_every=100):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : numpy array
            Training features
        y : numpy array
            Training labels
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print training progress
        print_every : int
            Print loss every N epochs
        """
        history = {'loss': []}
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            history['loss'].append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : numpy array
            Input data
        
        Returns:
        --------
        predictions : numpy array
            Predicted values
        """
        output = self.forward(X)
        if self.activation == 'sigmoid':
            # Binary classification: threshold at 0.5
            return (output > 0.5).astype(int)
        return output
    
    def score(self, X, y):
        """Calculate accuracy (for classification) or R² (for regression)."""
        predictions = self.predict(X)
        if self.activation == 'sigmoid':
            return np.mean(predictions == y)
        else:
            # R² score for regression
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
```

### Usage Example

```python
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example 1: Binary Classification
print("=== Binary Classification ===")
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Reshape y for neural network
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train network
nn = NeuralNetwork(
    layers=[2, 4, 4, 1],
    activation='sigmoid',
    learning_rate=0.1
)

history = nn.fit(X_train, y_train, epochs=1000, print_every=200)

# Evaluate
accuracy = nn.score(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Plot training loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Example 2: Regression
print("\n=== Regression ===")
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=3,
    n_informative=3,
    noise=10,
    random_state=42
)

y_reg = y_reg.reshape(-1, 1)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# Scale target for regression
y_scaler = StandardScaler()
y_train_reg = y_scaler.fit_transform(y_train_reg)
y_test_reg = y_scaler.transform(y_test_reg)

nn_reg = NeuralNetwork(
    layers=[3, 5, 5, 1],
    activation='relu',
    learning_rate=0.01
)

history_reg = nn_reg.fit(X_train_reg, y_train_reg, epochs=1000, print_every=200)

# Evaluate
r2 = nn_reg.score(X_test_reg, y_test_reg)
print(f"\nTest R² Score: {r2:.4f}")

plt.subplot(1, 2, 2)
plt.plot(history_reg['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (Regression)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Implementation 2: Neural Network with Multiple Activation Functions

```python
class NeuralNetworkAdvanced:
    def __init__(self, layers, hidden_activation='relu', output_activation='sigmoid', learning_rate=0.01):
        """
        Initialize Neural Network with different activations for hidden and output layers.
        
        Parameters:
        -----------
        layers : list
            List of layer sizes
        hidden_activation : str
            Activation for hidden layers
        output_activation : str
            Activation for output layer
        learning_rate : float
            Learning rate
        """
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # Initialize weights
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            w = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2.0 / layers[i-1])
            b = np.zeros((layers[i], 1))
            self.weights.append(w)
            self.biases.append(b)
        
        self.activations = []
        self.z_values = []
    
    def _apply_activation(self, z, activation_type):
        """Apply activation function based on type."""
        if activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif activation_type == 'tanh':
            return np.tanh(z)
        elif activation_type == 'relu':
            return np.maximum(0, z)
        elif activation_type == 'leaky_relu':
            return np.where(z > 0, z, 0.01 * z)
        elif activation_type == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {activation_type}")
    
    def _apply_activation_derivative(self, z, activation_type):
        """Apply activation derivative."""
        if activation_type == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif activation_type == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif activation_type == 'relu':
            return (z > 0).astype(float)
        elif activation_type == 'leaky_relu':
            return np.where(z > 0, 1, 0.01)
        elif activation_type == 'linear':
            return np.ones_like(z)
        else:
            raise ValueError(f"Unknown activation: {activation_type}")
    
    def forward(self, X):
        """Forward propagation."""
        self.activations = [X.T]
        self.z_values = []
        
        a = X.T
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            self.z_values.append(z)
            
            # Use output activation for last layer, hidden activation otherwise
            if i == self.num_layers - 2:
                a = self._apply_activation(z, self.output_activation)
            else:
                a = self._apply_activation(z, self.hidden_activation)
            
            self.activations.append(a)
        
        return a.T
    
    def backward(self, X, y, output):
        """Backward propagation."""
        m = X.shape[0]
        dW = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        
        # Output layer error
        if self.output_activation == 'sigmoid':
            dz = (output.T - y.T) / m
        else:
            dz = (output.T - y.T) / m
        
        # Backpropagate
        for l in range(self.num_layers - 2, -1, -1):
            dW[l] = np.dot(dz, self.activations[l].T)
            db[l] = np.sum(dz, axis=1, keepdims=True)
            
            if l > 0:
                dz = np.dot(self.weights[l].T, dz)
                # Use appropriate activation derivative
                if l == self.num_layers - 2:
                    activation_deriv = self._apply_activation_derivative(
                        self.z_values[l-1], self.hidden_activation
                    )
                else:
                    activation_deriv = self._apply_activation_derivative(
                        self.z_values[l-1], self.hidden_activation
                    )
                dz = dz * activation_deriv
        
        # Update weights
        for l in range(self.num_layers - 1):
            self.weights[l] -= self.learning_rate * dW[l]
            self.biases[l] -= self.learning_rate * db[l]
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """Train the network."""
        history = {'loss': []}
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            history['loss'].append(loss)
            self.backward(X, y, output)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        return self.forward(X)
```

---

## Key Takeaways

1. **Forward Propagation**: Compute activations layer by layer
2. **Backpropagation**: Compute gradients using chain rule
3. **Activation Functions**: Choose based on problem type (sigmoid for classification, ReLU for hidden layers)
4. **Weight Initialization**: Proper initialization (He, Xavier) is crucial
5. **Learning Rate**: Critical hyperparameter affecting convergence
6. **Loss Functions**: MSE for regression, cross-entropy for classification
7. **Gradient Descent**: Update weights using computed gradients

---

## Interview Tips

When implementing neural networks in interviews:

1. **Explain Forward Pass**: How input flows through network
2. **Derive Backpropagation**: Show understanding of chain rule
3. **Activation Functions**: Know derivatives and when to use each
4. **Weight Initialization**: Explain why proper initialization matters
5. **Vanishing Gradients**: Understand problem and solutions (ReLU, batch norm)
6. **Overfitting**: Mention regularization (dropout, L2)
7. **Hyperparameters**: Learning rate, batch size, architecture

---

## Time Complexity

- **Forward Pass**: O(m * n * h) where m = samples, n = features, h = hidden units
- **Backward Pass**: O(m * n * h) (similar to forward)
- **Training**: O(epochs * m * n * h)
- **Space Complexity**: O(n * h) for weights and activations

---

## Advantages

1. **Universal Approximator**: Can approximate any continuous function
2. **Non-linear**: Can learn complex non-linear patterns
3. **Feature Learning**: Automatically learns features
4. **Flexible**: Can handle various problem types

---

## Disadvantages

1. **Black Box**: Hard to interpret
2. **Hyperparameter Tuning**: Many hyperparameters to tune
3. **Computational Cost**: Training can be slow
4. **Overfitting**: Prone to overfitting without regularization
5. **Data Requirements**: Often needs large amounts of data

---

## References

* Backpropagation algorithm and chain rule
* Activation functions and their derivatives
* Weight initialization strategies
* Gradient descent and optimization
* Universal approximation theorem

