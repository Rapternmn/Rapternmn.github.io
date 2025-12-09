+++
title = "RNN and LSTM"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 13
description = "Complete implementation of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) from scratch using Python and NumPy. Covers forward propagation, backpropagation through time, and sequence modeling."
+++

---

## Introduction

Recurrent Neural Networks (RNNs) are designed to process sequential data by maintaining hidden states that capture information from previous time steps. Long Short-Term Memory (LSTM) networks are a special type of RNN that address the vanishing gradient problem and can learn long-term dependencies.

In this guide, we'll implement RNN and LSTM from scratch using Python and NumPy, covering forward propagation, backpropagation through time (BPTT), and sequence modeling.

---

## Mathematical Foundation

### Basic RNN

**Forward Pass:**

```
h[t] = tanh(W_hh * h[t-1] + W_xh * x[t] + b_h)
y[t] = W_hy * h[t] + b_y
```

Where:
- `h[t]` is the hidden state at time t
- `x[t]` is the input at time t
- `y[t]` is the output at time t
- `W_hh`, `W_xh`, `W_hy` are weight matrices
- `b_h`, `b_y` are bias vectors

### LSTM

LSTM uses gates to control information flow:

**Forget Gate:**
```
f[t] = σ(W_f * [h[t-1], x[t]] + b_f)
```

**Input Gate:**
```
i[t] = σ(W_i * [h[t-1], x[t]] + b_i)
C̃[t] = tanh(W_C * [h[t-1], x[t]] + b_C)
```

**Cell State Update:**
```
C[t] = f[t] * C[t-1] + i[t] * C̃[t]
```

**Output Gate:**
```
o[t] = σ(W_o * [h[t-1], x[t]] + b_o)
h[t] = o[t] * tanh(C[t])
```

**Output:**
```
y[t] = W_hy * h[t] + b_y
```

Where `σ` is sigmoid activation.

---

## Implementation 1: Basic RNN

```python
import numpy as np
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize RNN.
        
        Parameters:
        -----------
        input_size : int
            Size of input vector
        hidden_size : int
            Size of hidden state
        output_size : int
            Size of output vector
        learning_rate : float
            Learning rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights (Xavier initialization)
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))
        
        # Initialize biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # Store states for backpropagation
        self.hidden_states = []
        self.inputs = []
        self.outputs = []
    
    def _tanh(self, x):
        """Tanh activation."""
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        """Tanh derivative."""
        return 1 - np.tanh(x) ** 2
    
    def forward(self, X):
        """
        Forward pass through RNN.
        
        Parameters:
        -----------
        X : numpy array
            Input sequence of shape (sequence_length, input_size, batch_size)
            or (sequence_length, input_size) for single sample
        
        Returns:
        --------
        outputs : list
            List of outputs for each time step
        """
        # Handle single sample vs batch
        if X.ndim == 2:
            X = X[:, :, np.newaxis]  # Add batch dimension
        
        sequence_length, input_size, batch_size = X.shape
        self.inputs = X
        self.hidden_states = []
        self.outputs = []
        
        # Initialize hidden state
        h = np.zeros((self.hidden_size, batch_size))
        
        # Process each time step
        for t in range(sequence_length):
            # Current input
            x_t = X[t]
            
            # Update hidden state
            # h[t] = tanh(W_hh * h[t-1] + W_xh * x[t] + b_h)
            h = self._tanh(
                np.dot(self.W_hh, h) + np.dot(self.W_xh, x_t) + self.b_h
            )
            
            # Compute output
            # y[t] = W_hy * h[t] + b_y
            y_t = np.dot(self.W_hy, h) + self.b_y
            
            self.hidden_states.append(h.copy())
            self.outputs.append(y_t)
        
        return self.outputs
    
    def backward(self, d_outputs):
        """
        Backpropagation Through Time (BPTT).
        
        Parameters:
        -----------
        d_outputs : list
            Gradients of loss w.r.t. outputs for each time step
        """
        sequence_length = len(d_outputs)
        batch_size = d_outputs[0].shape[1]
        
        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # Initialize gradient w.r.t. hidden state
        dh_next = np.zeros((self.hidden_size, batch_size))
        
        # Backpropagate through time
        for t in reversed(range(sequence_length)):
            # Gradient w.r.t. output
            dy = d_outputs[t]
            
            # Gradient w.r.t. W_hy and b_y
            dW_hy += np.dot(dy, self.hidden_states[t].T)
            db_y += np.sum(dy, axis=1, keepdims=True)
            
            # Gradient w.r.t. hidden state (from output and next time step)
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # Gradient w.r.t. pre-activation (before tanh)
            dh_raw = dh * self._tanh_derivative(self.hidden_states[t])
            
            # Gradient w.r.t. W_xh, W_hh, b_h
            if t > 0:
                dW_hh += np.dot(dh_raw, self.hidden_states[t-1].T)
            else:
                # First time step has zero previous hidden state
                dW_hh += np.dot(dh_raw, np.zeros((self.hidden_size, batch_size)).T)
            
            dW_xh += np.dot(dh_raw, self.inputs[t].T)
            db_h += np.sum(dh_raw, axis=1, keepdims=True)
            
            # Gradient for previous time step
            dh_next = np.dot(self.W_hh.T, dh_raw)
        
        # Update weights
        self.W_xh -= self.learning_rate * dW_xh / batch_size
        self.W_hh -= self.learning_rate * dW_hh / batch_size
        self.W_hy -= self.learning_rate * dW_hy / batch_size
        self.b_h -= self.learning_rate * db_h / batch_size
        self.b_y -= self.learning_rate * db_y / batch_size
    
    def compute_loss(self, y_true, y_pred):
        """Compute mean squared error loss."""
        loss = 0
        for t in range(len(y_pred)):
            loss += np.mean((y_true[t] - y_pred[t]) ** 2)
        return loss / len(y_pred)
    
    def fit(self, X, y, epochs=100, verbose=True):
        """Train the RNN."""
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(X)
            
            # Compute loss and gradients
            loss = self.compute_loss(y, outputs)
            loss_history.append(loss)
            
            # Compute gradients w.r.t. outputs
            d_outputs = []
            for t in range(len(outputs)):
                d_output = 2 * (outputs[t] - y[t]) / len(outputs)
                d_outputs.append(d_output)
            
            # Backward pass
            self.backward(d_outputs)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return loss_history
    
    def predict(self, X):
        """Make predictions."""
        outputs = self.forward(X)
        return outputs
```

---

## Implementation 2: LSTM

```python
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize LSTM.
        
        Parameters:
        -----------
        input_size : int
            Size of input vector
        hidden_size : int
            Size of hidden state
        output_size : int
            Size of output vector
        learning_rate : float
            Learning rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Weight matrices for gates
        # Forget gate
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b_f = np.zeros((hidden_size, 1))
        
        # Input gate
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b_i = np.zeros((hidden_size, 1))
        
        # Candidate values
        self.W_C = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b_C = np.zeros((hidden_size, 1))
        
        # Output gate
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b_o = np.zeros((hidden_size, 1))
        
        # Output layer
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b_y = np.zeros((output_size, 1))
        
        # Store states
        self.hidden_states = []
        self.cell_states = []
        self.inputs = []
        self.outputs = []
        self.gates = []  # Store gate values for backprop
    
    def _sigmoid(self, x):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        """Sigmoid derivative."""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _tanh(self, x):
        """Tanh activation."""
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        """Tanh derivative."""
        return 1 - np.tanh(x) ** 2
    
    def forward(self, X):
        """
        Forward pass through LSTM.
        
        Parameters:
        -----------
        X : numpy array
            Input sequence of shape (sequence_length, input_size, batch_size)
            or (sequence_length, input_size) for single sample
        
        Returns:
        --------
        outputs : list
            List of outputs for each time step
        """
        # Handle single sample vs batch
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        
        sequence_length, input_size, batch_size = X.shape
        self.inputs = X
        self.hidden_states = []
        self.cell_states = []
        self.outputs = []
        self.gates = []
        
        # Initialize states
        h = np.zeros((self.hidden_size, batch_size))
        C = np.zeros((self.hidden_size, batch_size))
        
        # Process each time step
        for t in range(sequence_length):
            x_t = X[t]
            
            # Concatenate input and hidden state
            concat = np.vstack([x_t, h])
            
            # Forget gate
            f_t = self._sigmoid(np.dot(self.W_f, concat) + self.b_f)
            
            # Input gate
            i_t = self._sigmoid(np.dot(self.W_i, concat) + self.b_i)
            
            # Candidate cell state
            C_tilde = self._tanh(np.dot(self.W_C, concat) + self.b_C)
            
            # Update cell state
            C = f_t * C + i_t * C_tilde
            
            # Output gate
            o_t = self._sigmoid(np.dot(self.W_o, concat) + self.b_o)
            
            # Update hidden state
            h = o_t * self._tanh(C)
            
            # Compute output
            y_t = np.dot(self.W_hy, h) + self.b_y
            
            # Store states
            self.hidden_states.append(h.copy())
            self.cell_states.append(C.copy())
            self.gates.append({
                'f': f_t, 'i': i_t, 'C_tilde': C_tilde, 'o': o_t
            })
            self.outputs.append(y_t)
        
        return self.outputs
    
    def backward(self, d_outputs):
        """
        Backpropagation Through Time for LSTM.
        
        Parameters:
        -----------
        d_outputs : list
            Gradients of loss w.r.t. outputs
        """
        sequence_length = len(d_outputs)
        batch_size = d_outputs[0].shape[1]
        
        # Initialize gradients
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_C = np.zeros_like(self.W_C)
        dW_o = np.zeros_like(self.W_o)
        dW_hy = np.zeros_like(self.W_hy)
        
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_C = np.zeros_like(self.b_C)
        db_o = np.zeros_like(self.b_o)
        db_y = np.zeros_like(self.b_y)
        
        # Initialize gradients for hidden and cell states
        dh_next = np.zeros((self.hidden_size, batch_size))
        dC_next = np.zeros((self.hidden_size, batch_size))
        
        # Backpropagate through time
        for t in reversed(range(sequence_length)):
            # Gradient w.r.t. output
            dy = d_outputs[t]
            
            # Gradient w.r.t. W_hy and b_y
            dW_hy += np.dot(dy, self.hidden_states[t].T)
            db_y += np.sum(dy, axis=1, keepdims=True)
            
            # Gradient w.r.t. hidden state
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # Gradient w.r.t. cell state (from hidden state and next time step)
            dC = dh * self.gates[t]['o'] * self._tanh_derivative(self.cell_states[t]) + dC_next
            
            # Gradient w.r.t. output gate
            do = dh * self._tanh(self.cell_states[t])
            do_raw = do * self._sigmoid_derivative(
                np.dot(self.W_o, np.vstack([self.inputs[t], 
                      self.hidden_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size))])) + self.b_o
            )
            
            # Gradient w.r.t. cell state (from forget and input gates)
            dC_tilde = dC * self.gates[t]['i']
            di = dC * self.gates[t]['C_tilde']
            df = dC * (self.cell_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size)))
            
            # Compute gate gradients
            if t > 0:
                concat = np.vstack([self.inputs[t], self.hidden_states[t-1]])
            else:
                concat = np.vstack([self.inputs[t], np.zeros((self.hidden_size, batch_size))])
            
            di_raw = di * self._sigmoid_derivative(np.dot(self.W_i, concat) + self.b_i)
            df_raw = df * self._sigmoid_derivative(np.dot(self.W_f, concat) + self.b_f)
            dC_tilde_raw = dC_tilde * self._tanh_derivative(self.gates[t]['C_tilde'])
            
            # Update weight gradients
            dW_o += np.dot(do_raw, concat.T)
            dW_i += np.dot(di_raw, concat.T)
            dW_f += np.dot(df_raw, concat.T)
            dW_C += np.dot(dC_tilde_raw, concat.T)
            
            db_o += np.sum(do_raw, axis=1, keepdims=True)
            db_i += np.sum(di_raw, axis=1, keepdims=True)
            db_f += np.sum(df_raw, axis=1, keepdims=True)
            db_C += np.sum(dC_tilde_raw, axis=1, keepdims=True)
            
            # Gradients for previous time step
            dconcat = (np.dot(self.W_o.T, do_raw) + 
                      np.dot(self.W_i.T, di_raw) + 
                      np.dot(self.W_f.T, df_raw) + 
                      np.dot(self.W_C.T, dC_tilde_raw))
            
            dh_next = dconcat[self.input_size:]
            dC_next = dC * self.gates[t]['f']
        
        # Update weights
        self.W_f -= self.learning_rate * dW_f / batch_size
        self.W_i -= self.learning_rate * dW_i / batch_size
        self.W_C -= self.learning_rate * dW_C / batch_size
        self.W_o -= self.learning_rate * dW_o / batch_size
        self.W_hy -= self.learning_rate * dW_hy / batch_size
        
        self.b_f -= self.learning_rate * db_f / batch_size
        self.b_i -= self.learning_rate * db_i / batch_size
        self.b_C -= self.learning_rate * db_C / batch_size
        self.b_o -= self.learning_rate * db_o / batch_size
        self.b_y -= self.learning_rate * db_y / batch_size
    
    def compute_loss(self, y_true, y_pred):
        """Compute mean squared error loss."""
        loss = 0
        for t in range(len(y_pred)):
            loss += np.mean((y_true[t] - y_pred[t]) ** 2)
        return loss / len(y_pred)
    
    def fit(self, X, y, epochs=100, verbose=True):
        """Train the LSTM."""
        loss_history = []
        
        for epoch in range(epochs):
            outputs = self.forward(X)
            loss = self.compute_loss(y, outputs)
            loss_history.append(loss)
            
            d_outputs = []
            for t in range(len(outputs)):
                d_output = 2 * (outputs[t] - y[t]) / len(outputs)
                d_outputs.append(d_output)
            
            self.backward(d_outputs)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return loss_history
    
    def predict(self, X):
        """Make predictions."""
        return self.forward(X)
```

---

## Usage Example

```python
# Example: Sequence prediction
# Generate simple sequence data
sequence_length = 10
input_size = 3
hidden_size = 8
output_size = 1

# Create synthetic sequence data
X = np.random.randn(sequence_length, input_size, 1)
y = [np.random.randn(output_size, 1) for _ in range(sequence_length)]

# Test RNN
print("=== RNN ===")
rnn = RNN(input_size, hidden_size, output_size, learning_rate=0.01)
history_rnn = rnn.fit(X, y, epochs=50, verbose=True)

# Test LSTM
print("\n=== LSTM ===")
lstm = LSTM(input_size, hidden_size, output_size, learning_rate=0.01)
history_lstm = lstm.fit(X, y, epochs=50, verbose=True)

# Compare training curves
plt.figure(figsize=(10, 5))
plt.plot(history_rnn, label='RNN')
plt.plot(history_lstm, label='LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss: RNN vs LSTM')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Key Takeaways

1. **RNN**: Maintains hidden state to process sequences
2. **BPTT**: Backpropagation through time unrolls the network
3. **Vanishing Gradients**: RNNs suffer from vanishing gradient problem
4. **LSTM**: Uses gates to control information flow and solve vanishing gradients
5. **Gates**: Forget, input, and output gates control cell state
6. **Long-term Dependencies**: LSTM can learn long-term dependencies better than RNN
7. **Sequence Modeling**: Both are used for time series, NLP, and sequential data

---

## Interview Tips

When implementing RNN/LSTM in interviews:

1. **Explain Recurrence**: How hidden state carries information forward
2. **BPTT**: Understand backpropagation through time
3. **Vanishing Gradients**: Explain problem and how LSTM solves it
4. **Gates**: Understand purpose of each LSTM gate
5. **Cell State**: Explain how cell state differs from hidden state
6. **Applications**: Time series, NLP, speech recognition
7. **Variants**: Mention GRU, attention mechanisms

---

## Time Complexity

- **Forward Pass**: O(sequence_length * (input_size + hidden_size) * hidden_size)
- **Backward Pass**: O(sequence_length * hidden_size²)
- **LSTM**: Similar but with 4 gates (approximately 4x RNN cost)

---

## Advantages

1. **Sequence Processing**: Natural for sequential data
2. **Variable Length**: Can handle variable-length sequences
3. **Memory**: Maintains information across time steps
4. **LSTM**: Solves vanishing gradient problem

---

## Disadvantages

1. **Slow Training**: BPTT is computationally expensive
2. **Vanishing Gradients**: RNNs struggle with long sequences
3. **Sequential Processing**: Can't parallelize across time steps
4. **Memory**: Requires storing all hidden states

---

## References

* Recurrent neural networks and backpropagation through time
* Long short-term memory networks and gating mechanisms
* Vanishing and exploding gradient problems
* Sequence modeling and time series prediction
* Natural language processing with RNNs

