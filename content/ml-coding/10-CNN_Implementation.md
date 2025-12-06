+++
title = "Convolutional Neural Network (CNN) Implementation from Scratch"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 10
description = "Complete implementation of Convolutional Neural Networks from scratch using Python and NumPy. Covers convolutional layers, pooling, backpropagation through convolutions, and image classification."
+++

---

## Introduction

Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies of features, making them highly effective for image recognition, object detection, and computer vision tasks.

In this guide, we'll implement CNN from scratch using Python and NumPy, covering convolutional layers, pooling operations, and backpropagation through convolutions.

---

## Mathematical Foundation

### Convolution Operation

For a 2D convolution:

```
O[i, j] = Σ Σ I[i+m, j+n] * K[m, n]
```

Where:
- `I` is the input feature map
- `K` is the kernel (filter)
- `O` is the output feature map

### Convolution with Padding

To maintain spatial dimensions:

```
Output_size = (Input_size + 2*Padding - Kernel_size) / Stride + 1
```

### Convolution with Multiple Channels

For input with C channels:

```
O[i, j, k] = Σ_c Σ_m Σ_n I[i+m, j+n, c] * K[m, n, c, k]
```

Where `k` is the output channel index.

### Max Pooling

```
O[i, j] = max(I[i*stride : i*stride+pool_size, j*stride : j*stride+pool_size])
```

### Average Pooling

```
O[i, j] = mean(I[i*stride : i*stride+pool_size, j*stride : j*stride+pool_size])
```

---

## Implementation 1: Basic Convolutional Layer

```python
import numpy as np
import matplotlib.pyplot as plt

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize 2D Convolutional Layer.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int or tuple
            Size of convolutional kernel
        stride : int
            Stride of convolution
        padding : int
            Padding size
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (He initialization)
        self.weights = np.random.randn(
            out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]
        ) * np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        
        self.bias = np.zeros(out_channels)
        
        # Store for backprop
        self.input = None
        self.output = None
    
    def _pad_input(self, X):
        """Add padding to input."""
        if self.padding == 0:
            return X
        
        return np.pad(
            X,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )
    
    def forward(self, X):
        """
        Forward pass through convolutional layer.
        
        Parameters:
        -----------
        X : numpy array
            Input of shape (batch_size, in_channels, height, width)
        
        Returns:
        --------
        output : numpy array
            Output of shape (batch_size, out_channels, out_height, out_width)
        """
        self.input = X
        batch_size, in_channels, in_height, in_width = X.shape
        
        # Add padding
        X_padded = self._pad_input(X)
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input region
                        ih_start = oh * self.stride
                        ih_end = ih_start + self.kernel_size[0]
                        iw_start = ow * self.stride
                        iw_end = iw_start + self.kernel_size[1]
                        
                        # Extract input region
                        input_region = X_padded[b, :, ih_start:ih_end, iw_start:iw_end]
                        
                        # Convolve
                        output[b, oc, oh, ow] = np.sum(
                            input_region * self.weights[oc]
                        ) + self.bias[oc]
        
        self.output = output
        return output
    
    def backward(self, d_output, learning_rate=0.01):
        """
        Backward pass through convolutional layer.
        
        Parameters:
        -----------
        d_output : numpy array
            Gradient of loss with respect to output
        learning_rate : float
            Learning rate for weight updates
        
        Returns:
        --------
        d_input : numpy array
            Gradient of loss with respect to input
        """
        batch_size, out_channels, out_height, out_width = d_output.shape
        _, in_channels, in_height, in_width = self.input.shape
        
        # Initialize gradients
        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)
        d_input = np.zeros_like(self.input)
        
        # Pad input for gradient calculation
        d_input_padded = np.zeros((
            batch_size, in_channels,
            in_height + 2 * self.padding, in_width + 2 * self.padding
        ))
        
        # Backpropagate gradients
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input region
                        ih_start = oh * self.stride
                        ih_end = ih_start + self.kernel_size[0]
                        iw_start = ow * self.stride
                        iw_end = iw_start + self.kernel_size[1]
                        
                        # Gradient w.r.t. weights
                        input_region = self._pad_input(self.input[b:b+1])[0, :, ih_start:ih_end, iw_start:iw_end]
                        d_weights[oc] += input_region * d_output[b, oc, oh, ow]
                        
                        # Gradient w.r.t. bias
                        d_bias[oc] += d_output[b, oc, oh, ow]
                        
                        # Gradient w.r.t. input (full convolution with flipped kernel)
                        d_input_padded[:, :, ih_start:ih_end, iw_start:iw_end] += \
                            self.weights[oc] * d_output[b, oc, oh, ow]
        
        # Remove padding from input gradient
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_input_padded
        
        # Update weights and biases
        self.weights -= learning_rate * d_weights / batch_size
        self.bias -= learning_rate * d_bias / batch_size
        
        return d_input
```

---

## Implementation 2: Max Pooling Layer

```python
class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize Max Pooling Layer.
        
        Parameters:
        -----------
        pool_size : int
            Size of pooling window
        stride : int
            Stride of pooling operation
        """
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.max_indices = None
    
    def forward(self, X):
        """
        Forward pass through max pooling layer.
        
        Parameters:
        -----------
        X : numpy array
            Input of shape (batch_size, channels, height, width)
        
        Returns:
        --------
        output : numpy array
            Output of shape (batch_size, channels, out_height, out_width)
        """
        self.input = X
        batch_size, channels, in_height, in_width = X.shape
        
        # Calculate output dimensions
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input region
                        ih_start = oh * self.stride
                        ih_end = ih_start + self.pool_size
                        iw_start = ow * self.stride
                        iw_end = iw_start + self.pool_size
                        
                        # Extract region
                        region = X[b, c, ih_start:ih_end, iw_start:iw_end]
                        
                        # Find max and its index
                        max_val = np.max(region)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        
                        output[b, c, oh, ow] = max_val
                        self.max_indices[b, c, oh, ow] = [
                            ih_start + max_idx[0],
                            iw_start + max_idx[1]
                        ]
        
        return output
    
    def backward(self, d_output):
        """
        Backward pass through max pooling layer.
        
        Parameters:
        -----------
        d_output : numpy array
            Gradient of loss with respect to output
        
        Returns:
        --------
        d_input : numpy array
            Gradient of loss with respect to input
        """
        batch_size, channels, out_height, out_width = d_output.shape
        _, _, in_height, in_width = self.input.shape
        
        d_input = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Get index of max value
                        max_idx = self.max_indices[b, c, oh, ow]
                        # Only the max position gets the gradient
                        d_input[b, c, max_idx[0], max_idx[1]] += d_output[b, c, oh, ow]
        
        return d_input
```

---

## Implementation 3: Average Pooling Layer

```python
class AvgPool2D:
    def __init__(self, pool_size=2, stride=2):
        """Initialize Average Pooling Layer."""
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
    
    def forward(self, X):
        """Forward pass through average pooling layer."""
        self.input = X
        batch_size, channels, in_height, in_width = X.shape
        
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        ih_start = oh * self.stride
                        ih_end = ih_start + self.pool_size
                        iw_start = ow * self.stride
                        iw_end = iw_start + self.pool_size
                        
                        region = X[b, c, ih_start:ih_end, iw_start:iw_end]
                        output[b, c, oh, ow] = np.mean(region)
        
        return output
    
    def backward(self, d_output):
        """Backward pass through average pooling layer."""
        batch_size, channels, out_height, out_width = d_output.shape
        _, _, in_height, in_width = self.input.shape
        
        d_input = np.zeros_like(self.input)
        pool_area = self.pool_size * self.pool_size
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        ih_start = oh * self.stride
                        ih_end = ih_start + self.pool_size
                        iw_start = ow * self.stride
                        iw_end = iw_start + self.pool_size
                        
                        # Distribute gradient equally to all positions in pool
                        gradient = d_output[b, c, oh, ow] / pool_area
                        d_input[b, c, ih_start:ih_end, iw_start:iw_end] += gradient
        
        return d_input
```

---

## Implementation 4: Flatten Layer

```python
class Flatten:
    def __init__(self):
        """Initialize Flatten Layer."""
        self.input_shape = None
    
    def forward(self, X):
        """
        Flatten input from (batch, channels, height, width) to (batch, features).
        
        Parameters:
        -----------
        X : numpy array
            Input of shape (batch_size, channels, height, width)
        
        Returns:
        --------
        output : numpy array
            Flattened output of shape (batch_size, channels*height*width)
        """
        self.input_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, d_output):
        """
        Reshape gradient back to original shape.
        
        Parameters:
        -----------
        d_output : numpy array
            Gradient of shape (batch_size, features)
        
        Returns:
        --------
        d_input : numpy array
            Gradient of shape (batch_size, channels, height, width)
        """
        return d_output.reshape(self.input_shape)
```

---

## Implementation 5: Complete CNN

```python
class CNN:
    def __init__(self, learning_rate=0.01):
        """Initialize CNN."""
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_history = []
    
    def add_conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """Add convolutional layer."""
        layer = Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.layers.append(layer)
        return self
    
    def add_pool_layer(self, pool_type='max', pool_size=2, stride=2):
        """Add pooling layer."""
        if pool_type == 'max':
            layer = MaxPool2D(pool_size, stride)
        else:
            layer = AvgPool2D(pool_size, stride)
        self.layers.append(layer)
        return self
    
    def add_flatten(self):
        """Add flatten layer."""
        self.layers.append(Flatten())
        return self
    
    def add_dense(self, input_size, output_size, activation='relu'):
        """Add dense (fully connected) layer."""
        # Simplified dense layer (can use NeuralNetwork class from previous file)
        layer = DenseLayer(input_size, output_size, activation)
        self.layers.append(layer)
        return self
    
    def forward(self, X):
        """Forward pass through all layers."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, d_output):
        """Backward pass through all layers."""
        gradient = d_output
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                gradient = layer.backward(gradient, self.learning_rate)
            else:
                gradient = layer.backward(gradient)
        return gradient
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y, epochs=10, verbose=True):
        """Train the CNN."""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
            
            # Backward pass
            d_output = (output - y) / y.shape[0]
            self.backward(d_output)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return self.loss_history
    
    def predict(self, X):
        """Make predictions."""
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Dense layer for CNN
class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        """Initialize Dense Layer."""
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.input = None
        self.output = None
    
    def _relu(self, z):
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass."""
        self.input = X.T if X.ndim > 2 else X
        z = np.dot(self.weights, self.input) + self.bias
        
        if self.activation == 'relu':
            self.output = self._relu(z)
        elif self.activation == 'sigmoid':
            self.output = self._sigmoid(z)
        else:
            self.output = z
        
        return self.output.T
    
    def backward(self, d_output, learning_rate=0.01):
        """Backward pass."""
        if self.activation == 'relu':
            d_output = d_output.T * self._relu_derivative(self.output)
        elif self.activation == 'sigmoid':
            d_output = d_output.T * self._sigmoid_derivative(self.output)
        else:
            d_output = d_output.T
        
        d_weights = np.dot(d_output, self.input.T)
        d_bias = np.sum(d_output, axis=1, keepdims=True)
        d_input = np.dot(self.weights.T, d_output)
        
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        
        return d_input.T
```

---

## Usage Example

```python
# Simple example with synthetic image data
# Generate random image data (batch_size=4, channels=1, height=28, width=28)
X = np.random.randn(4, 1, 28, 28)
y = np.array([[1], [0], [1], [0]])  # Binary labels

# Create CNN
cnn = CNN(learning_rate=0.01)
cnn.add_conv_layer(in_channels=1, out_channels=8, kernel_size=3, padding=1)
cnn.add_pool_layer(pool_type='max', pool_size=2, stride=2)
cnn.add_conv_layer(in_channels=8, out_channels=16, kernel_size=3, padding=1)
cnn.add_pool_layer(pool_type='max', pool_size=2, stride=2)
cnn.add_flatten()
cnn.add_dense(input_size=16*7*7, output_size=64, activation='relu')
cnn.add_dense(input_size=64, output_size=1, activation='sigmoid')

# Train
history = cnn.fit(X, y, epochs=10)

# Predict
predictions = cnn.predict(X)
print(f"Predictions: {predictions.flatten()}")
```

---

## Key Takeaways

1. **Convolution**: Learns spatial features through sliding filters
2. **Pooling**: Reduces spatial dimensions and provides translation invariance
3. **Feature Maps**: Each filter produces a feature map detecting different patterns
4. **Hierarchical Features**: Early layers detect edges, later layers detect complex patterns
5. **Parameter Sharing**: Same filter applied across all spatial locations
6. **Translation Invariance**: Pooling makes network robust to small translations

---

## Interview Tips

When implementing CNNs in interviews:

1. **Explain Convolution**: How filters slide across input
2. **Padding and Stride**: Understand their effects on output size
3. **Backpropagation**: Know how to compute gradients through convolution
4. **Pooling**: Explain max vs average pooling and their use cases
5. **Architecture**: Discuss common architectures (LeNet, AlexNet, VGG, ResNet)
6. **Receptive Field**: Understand how receptive field grows with depth
7. **Parameter Efficiency**: Explain why CNNs have fewer parameters than fully connected

---

## Time Complexity

- **Convolution**: O(batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w)
- **Pooling**: O(batch * channels * out_h * out_w * pool_size²)
- **Training**: O(epochs * (conv + pool + dense operations))

---

## Advantages

1. **Spatial Feature Learning**: Automatically learns spatial hierarchies
2. **Parameter Sharing**: Fewer parameters than fully connected networks
3. **Translation Invariant**: Robust to translations
4. **Effective for Images**: State-of-the-art for image tasks

---

## Disadvantages

1. **Computational Cost**: Convolutions can be expensive
2. **Fixed Input Size**: Often requires fixed input dimensions
3. **Limited Receptive Field**: Early layers have small receptive fields
4. **Hyperparameter Tuning**: Many hyperparameters (filters, kernel size, etc.)

---

## References

* Convolution operation and feature maps
* Backpropagation through convolutional layers
* Pooling operations and their effects
* CNN architectures (LeNet, AlexNet, VGG, ResNet)
* Image classification and computer vision

