+++
title = "Deep Neural Networks (DNN): Fundamentals"
date = 2025-11-22T11:00:00+05:30
draft = false
description = "Comprehensive guide to Deep Neural Networks covering feedforward propagation, backpropagation, loss functions, optimization algorithms (SGD, Adam), regularization techniques (dropout, batch normalization), and mathematical foundations."
+++

# 🧠 Deep Neural Networks (DNN): Fundamentals

Deep Neural Networks (DNNs) are multi-layer neural networks that can learn complex, hierarchical representations from data. This document covers fundamental DNN algorithms, their mathematical formulations, loss functions, gradient calculations, and training procedures.

**Key Concepts:**
- Feedforward propagation
- Backpropagation algorithm
- Loss functions and optimization
- Gradient computation
- Regularization techniques

---

## 🌳 1. Linear Regression (Foundation)

Linear Regression is the simplest form of neural network (a single neuron with linear activation).

### Mathematical Formulation

**Model:**
```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
```

Where:
- `w = [w₁, w₂, ..., wₙ]ᵀ` = weight vector
- `x = [x₁, x₂, ..., xₙ]ᵀ` = input features
- `b` = bias term
- `ŷ` = predicted output

**Matrix Form (for batch of m samples):**
```
Ŷ = XW + b
```

Where:
- `X` = (m × n) input matrix
- `W` = (n × 1) weight vector
- `Ŷ` = (m × 1) predictions

### Loss Function: Mean Squared Error (MSE)

**Formula:**
```
L = (1/2m) * Σ(ŷᵢ - yᵢ)²
```

**Vectorized Form:**
```
L = (1/2m) * ||Ŷ - Y||²
```

### Gradient Calculation

**Gradient w.r.t. weights:**
```
∂L/∂wⱼ = (1/m) * Σ(ŷᵢ - yᵢ) * xᵢⱼ
```

**Vectorized:**
```
∇w L = (1/m) * Xᵀ(Ŷ - Y)
```

**Gradient w.r.t. bias:**
```
∂L/∂b = (1/m) * Σ(ŷᵢ - yᵢ)
```

**Vectorized:**
```
∇b L = (1/m) * Σ(Ŷ - Y)
```

### Training Algorithm

**Gradient Descent Update:**
```
w := w - α * ∇w L
b := b - α * ∇b L
```

Where `α` is the learning rate.

### Properties

- **Convex optimization:** Guaranteed global minimum
- **Closed-form solution:** Can be solved analytically using normal equation
- **Fast training:** Simple gradient computation
- **Interpretable:** Weights show feature importance

### Limitations

- **Linear relationships only:** Cannot model non-linear patterns
- **Assumes linearity:** May fail on complex data

---

## 🟦 2. Logistic Regression

Logistic Regression is a binary classification algorithm using a sigmoid activation function.

### Mathematical Formulation

**Model:**
```
z = wᵀx + b
ŷ = σ(z) = 1 / (1 + e^(-z))
```

Where `σ(z)` is the sigmoid function.

**Output Interpretation:**
- `ŷ` represents the probability that `y = 1`
- `P(y=1|x) = ŷ`
- `P(y=0|x) = 1 - ŷ`

### Loss Function: Binary Cross-Entropy

**Formula:**
```
L = -(1/m) * Σ[yᵢ * log(ŷᵢ) + (1-yᵢ) * log(1-ŷᵢ)]
```

**For a single sample:**
```
L = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

**Properties:**
- Derived from maximum likelihood estimation
- Penalizes confident wrong predictions heavily
- Well-suited for probability outputs

### Gradient Calculation

**Step 1: Gradient w.r.t. output ŷ**
```
∂L/∂ŷ = -(y/ŷ) + (1-y)/(1-ŷ) = (ŷ - y) / [ŷ(1-ŷ)]
```

**Step 2: Gradient w.r.t. z (before sigmoid)**
```
∂L/∂z = ∂L/∂ŷ * ∂ŷ/∂z
```

Since `∂ŷ/∂z = σ(z)(1-σ(z)) = ŷ(1-ŷ)`:
```
∂L/∂z = (ŷ - y) / [ŷ(1-ŷ)] * ŷ(1-ŷ) = ŷ - y
```

**Step 3: Gradient w.r.t. weights**
```
∂L/∂wⱼ = ∂L/∂z * ∂z/∂wⱼ = (ŷ - y) * xⱼ
```

**Vectorized:**
```
∇w L = (1/m) * Xᵀ(Ŷ - Y)
```

**Step 4: Gradient w.r.t. bias**
```
∂L/∂b = ∂L/∂z * ∂z/∂b = (ŷ - y)
```

**Vectorized:**
```
∇b L = (1/m) * Σ(Ŷ - Y)
```

### Training Algorithm

**Gradient Descent Update:**
```
w := w - α * ∇w L
b := b - α * ∇b L
```

**Complete Algorithm:**
1. Initialize weights `w` and bias `b`
2. For each iteration:
   - Forward pass: Compute `ŷ = σ(wᵀx + b)`
   - Compute loss: `L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]`
   - Backward pass: Compute gradients
   - Update parameters: `w := w - α * ∇w L`, `b := b - α * ∇b L`
3. Repeat until convergence

### Decision Boundary

**Classification Rule:**
```
Predict y = 1 if ŷ ≥ 0.5 (i.e., z ≥ 0)
Predict y = 0 if ŷ < 0.5 (i.e., z < 0)
```

**Decision Boundary:**
```
wᵀx + b = 0
```

This is a linear decision boundary (hyperplane).

### Properties

- **Probabilistic output:** Provides probability estimates
- **Interpretable:** Weights indicate feature importance
- **Convex loss:** Guaranteed convergence to global minimum
- **Efficient:** Fast training and prediction

### Limitations

- **Linear decision boundary:** Cannot handle non-linearly separable data
- **Binary classification only:** Requires extension for multi-class

### Multi-class Extension: Softmax Regression

**Model:**
```
zⱼ = wⱼᵀx + bⱼ  (for each class j)
ŷⱼ = e^(zⱼ) / Σₖ e^(zₖ)  (softmax)
```

**Loss Function: Categorical Cross-Entropy**
```
L = -(1/m) * Σᵢ Σⱼ yᵢⱼ * log(ŷᵢⱼ)
```

**Gradient:**
```
∂L/∂zⱼ = ŷⱼ - yⱼ
```

---

## 🟩 3. Perceptron

The Perceptron is the simplest neural network unit, a binary classifier with a step activation function.

### Mathematical Formulation

**Model:**
```
z = wᵀx + b
ŷ = {
    1,  if z ≥ 0
    0,  if z < 0
}
```

This is a **step function** (Heaviside function).

### Loss Function: Perceptron Loss

**Formula:**
```
L = {
    0,        if y = ŷ (correct prediction)
    -(wᵀx + b) * y,  if y ≠ ŷ (misclassification)
}
```

**Alternative Formulation:**
```
L = max(0, -y * (wᵀx + b))
```

This is similar to the **hinge loss**.

### Gradient Calculation

**When misclassified (y = 1, ŷ = 0):**
```
∂L/∂wⱼ = -y * xⱼ = -xⱼ
∂L/∂b = -y = -1
```

**When misclassified (y = 0, ŷ = 1):**
```
∂L/∂wⱼ = -y * xⱼ = 0
∂L/∂b = -y = 0
```

**When correctly classified:**
```
∂L/∂wⱼ = 0
∂L/∂b = 0
```

### Training Algorithm: Perceptron Learning Rule

**Algorithm:**
1. Initialize weights `w` and bias `b` (typically to zeros or small random values)
2. For each training sample `(x, y)`:
   - Compute prediction: `ŷ = step(wᵀx + b)`
   - If `ŷ ≠ y` (misclassified):
     - Update: `w := w + α * y * x`
     - Update: `b := b + α * y`
3. Repeat until all samples are correctly classified or max iterations reached

**Update Rule:**
```
w := w + α * y * x  (if misclassified)
b := b + α * y      (if misclassified)
```

### Convergence Theorem

**Perceptron Convergence Theorem:**
- If data is **linearly separable**, the Perceptron algorithm will converge in a finite number of steps
- The number of mistakes is bounded by `(R/γ)²`, where:
  - `R` = maximum norm of training examples
  - `γ` = margin (distance from decision boundary to closest point)

### Properties

- **Simple and fast:** Very efficient training
- **Online learning:** Can update with each sample
- **Guaranteed convergence:** For linearly separable data

### Limitations

- **Only linearly separable data:** Cannot learn XOR function
- **No probabilistic output:** Only binary predictions
- **May not converge:** If data is not linearly separable

---

## 🟪 4. Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron is a feedforward neural network with one or more hidden layers.

### Architecture

**Structure:**
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Example (2-layer MLP):**
```
x → [Linear + ReLU] → h → [Linear + Activation] → ŷ
```

### Mathematical Formulation

**Forward Pass:**

**Layer 1 (Hidden):**
```
z¹ = W¹x + b¹
h = σ¹(z¹)  (activation function, e.g., ReLU)
```

**Layer 2 (Output):**
```
z² = W²h + b²
ŷ = σ²(z²)  (activation function depends on task)
```

**General Form (L layers):**
```
zˡ = Wˡaˡ⁻¹ + bˡ
aˡ = σˡ(zˡ)
```

Where:
- `a⁰ = x` (input)
- `aᴸ = ŷ` (output)
- `Wˡ` = weight matrix for layer `l`
- `bˡ` = bias vector for layer `l`
- `σˡ` = activation function for layer `l`

### Loss Functions

**For Regression:**
```
L = (1/2m) * Σ(ŷᵢ - yᵢ)²  (MSE)
```

**For Binary Classification:**
```
L = -(1/m) * Σ[yᵢ * log(ŷᵢ) + (1-yᵢ) * log(1-ŷᵢ)]  (BCE)
```

**For Multi-class Classification:**
```
L = -(1/m) * Σᵢ Σⱼ yᵢⱼ * log(ŷᵢⱼ)  (CCE)
```

### Backpropagation Algorithm

Backpropagation computes gradients using the chain rule of calculus.

#### Algorithm Steps

**1. Forward Pass:**
```
For l = 1 to L:
    zˡ = Wˡaˡ⁻¹ + bˡ
    aˡ = σˡ(zˡ)
```

**2. Compute Output Error:**
```
δᴸ = ∂L/∂aᴸ * σ'ᴸ(zᴸ)
```

**For MSE loss:**
```
δᴸ = (aᴸ - y) * σ'ᴸ(zᴸ)
```

**For BCE loss (with sigmoid):**
```
δᴸ = aᴸ - y
```

**For CCE loss (with softmax):**
```
δᴸ = aᴸ - y
```

**3. Backward Pass (Error Propagation):**
```
For l = L-1 to 1:
    δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'ˡ(zˡ)
```

Where `⊙` denotes element-wise multiplication (Hadamard product).

**4. Compute Gradients:**
```
∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ
∂L/∂bˡ = δˡ
```

**5. Update Parameters:**
```
Wˡ := Wˡ - α * ∂L/∂Wˡ
bˡ := bˡ - α * ∂L/∂bˡ
```

### Detailed Gradient Derivation

**For a 2-layer MLP:**

**Layer 2 (Output) Gradients:**
```
∂L/∂W² = (1/m) * δ²(h)ᵀ
∂L/∂b² = (1/m) * Σδ²
```

Where `δ² = (ŷ - y) * σ'²(z²)` or `δ² = ŷ - y` (if using cross-entropy with softmax/sigmoid).

**Layer 1 (Hidden) Gradients:**
```
δ¹ = (W²)ᵀδ² ⊙ σ'¹(z¹)
∂L/∂W¹ = (1/m) * δ¹(x)ᵀ
∂L/∂b¹ = (1/m) * Σδ¹
```

### Activation Function Derivatives

**ReLU:**
```
σ'(z) = {
    1,  if z > 0
    0,  if z ≤ 0
}
```

**Sigmoid:**
```
σ'(z) = σ(z)(1 - σ(z))
```

**Tanh:**
```
σ'(z) = 1 - tanh²(z)
```

**Linear:**
```
σ'(z) = 1
```

### Training Algorithm

**Complete Training Loop:**
1. Initialize all weights and biases (e.g., Xavier/Glorot or He initialization)
2. For each epoch:
   - For each batch:
     - **Forward Pass:** Compute predictions
     - **Compute Loss:** Calculate loss on batch
     - **Backward Pass:** Compute gradients using backpropagation
     - **Update Parameters:** Apply gradient descent update
3. Repeat until convergence

### Properties

- **Universal Approximation:** Can approximate any continuous function (with sufficient capacity)
- **Non-linear:** Can learn complex decision boundaries
- **Flexible:** Can handle various tasks (regression, classification)

### Limitations

- **Vanishing/Exploding Gradients:** Deep networks can suffer from gradient problems
- **Overfitting:** Can memorize training data
- **Local Minima:** Non-convex optimization
- **Computational Cost:** Training can be slow for large networks

---

## 🟧 5. Feedforward Neural Networks

Feedforward Neural Networks are the general class of DNNs where information flows in one direction (input → output).

### Architecture Types

**1. Fully Connected (Dense) Networks:**
- Every neuron in layer `l` connects to every neuron in layer `l+1`
- Most common type of DNN

**2. Deep Networks:**
- Multiple hidden layers (typically 3+)
- Can learn hierarchical features

### Forward Propagation

**General Formula:**
```
a⁰ = x
For l = 1 to L:
    zˡ = Wˡaˡ⁻¹ + bˡ
    aˡ = σˡ(zˡ)
ŷ = aᴸ
```

**Vectorized (for batch):**
```
A⁰ = X
For l = 1 to L:
    Zˡ = Aˡ⁻¹(Wˡ)ᵀ + bˡ  (broadcast)
    Aˡ = σˡ(Zˡ)
Ŷ = Aᴸ
```

### Backward Propagation

**General Formula:**
```
δᴸ = ∇aᴸ L ⊙ σ'ᴸ(zᴸ)
For l = L-1 to 1:
    δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'ˡ(zˡ)
    ∂L/∂Wˡ = (1/m) * δˡ(aˡ⁻¹)ᵀ
    ∂L/∂bˡ = (1/m) * Σδˡ
```

### Computational Graph

**Example for 2-layer network:**
```
x → [W¹, b¹] → z¹ → [σ¹] → h → [W², b²] → z² → [σ²] → ŷ → [Loss] → L
```

**Backpropagation traces backward:**
```
L → ∂L/∂ŷ → ∂L/∂z² → ∂L/∂W², ∂L/∂b²
         → ∂L/∂h → ∂L/∂z¹ → ∂L/∂W¹, ∂L/∂b¹
```

### Matrix Dimensions

**For a network with:**
- Input size: `n₀`
- Hidden layer sizes: `n₁, n₂, ..., nₗ₋₁`
- Output size: `nₗ`
- Batch size: `m`

**Weight Matrices:**
- `W¹`: (n₁ × n₀)
- `W²`: (n₂ × n₁)
- ...
- `Wᴸ`: (nₗ × nₗ₋₁)

**Activations:**
- `A⁰`: (m × n₀)
- `A¹`: (m × n₁)
- ...
- `Aᴸ`: (m × nₗ)

**Gradients:**
- `∂L/∂Wˡ`: (nˡ × nˡ⁻¹)
- `∂L/∂bˡ`: (nˡ,)
- `δˡ`: (m × nˡ)

---

## 🟨 6. Regularization Techniques

Regularization prevents overfitting by constraining model complexity.

### 6.1 L2 Regularization (Weight Decay)

**Modified Loss Function:**
```
L_reg = L + (λ/2) * ||W||²
```

Where:
- `L` = original loss
- `λ` = regularization strength (hyperparameter)
- `||W||² = Σᵢⱼ W²ᵢⱼ` = sum of squared weights

**Gradient Update:**
```
∂L_reg/∂W = ∂L/∂W + λW
```

**Weight Update:**
```
W := W - α(∂L/∂W + λW)
    = (1 - αλ)W - α * ∂L/∂W
```

**Properties:**
- Penalizes large weights
- Encourages smooth functions
- Prevents overfitting
- `(1 - αλ)` term causes weight decay

### 6.2 L1 Regularization (Lasso)

**Modified Loss Function:**
```
L_reg = L + λ * ||W||₁
```

Where `||W||₁ = Σᵢⱼ |Wᵢⱼ|` = sum of absolute weights.

**Gradient Update:**
```
∂L_reg/∂W = ∂L/∂W + λ * sign(W)
```

**Properties:**
- Encourages sparsity (many weights become exactly zero)
- Feature selection effect
- More aggressive than L2

### 6.3 Dropout

**During Training:**
- Randomly set a fraction `p` of neurons to zero
- Each neuron is kept with probability `(1-p)`
- Scales remaining activations by `1/(1-p)`

**Mathematical Formulation:**
```
r ~ Bernoulli(1-p)
ã = r ⊙ a / (1-p)
```

Where:
- `r` = binary mask
- `a` = original activation
- `ã` = masked activation

**During Inference:**
- Use all neurons (no dropout)
- Scale outputs by `(1-p)` if training scaling was used

**Properties:**
- Prevents co-adaptation of neurons
- Acts as ensemble of sub-networks
- Effective regularization technique
- Common `p` values: 0.2-0.5

### 6.4 Early Stopping

**Algorithm:**
1. Split data into training and validation sets
2. Monitor validation loss during training
3. Stop training when validation loss stops improving
4. Restore weights from best validation performance

**Properties:**
- Simple and effective
- Prevents overfitting
- No additional computational cost during inference

### 6.5 Batch Normalization

**Normalization:**
```
μ = (1/m) * Σxᵢ
σ² = (1/m) * Σ(xᵢ - μ)²
x̂ = (x - μ) / √(σ² + ε)
```

**Scale and Shift:**
```
y = γx̂ + β
```

Where `γ` and `β` are learnable parameters.

**Properties:**
- Normalizes activations
- Allows higher learning rates
- Reduces internal covariate shift
- Acts as regularization

---

## 🟩 7. Optimization Algorithms

### 7.1 Stochastic Gradient Descent (SGD)

**Update Rule:**
```
θ := θ - α * ∇θ L(θ; xᵢ, yᵢ)
```

**Properties:**
- Updates after each sample (or mini-batch)
- Noisy gradients (helps escape local minima)
- Fast convergence
- May oscillate near optimum

### 7.2 Batch Gradient Descent

**Update Rule:**
```
θ := θ - α * (1/m) * Σ∇θ L(θ; xᵢ, yᵢ)
```

**Properties:**
- Uses all training data
- Smooth gradients
- Slow for large datasets
- Guaranteed convergence (for convex functions)

### 7.3 Mini-batch Gradient Descent

**Update Rule:**
```
θ := θ - α * (1/b) * Σ∇θ L(θ; xᵢ, yᵢ)
```

Where `b` is the batch size.

**Properties:**
- Balance between SGD and batch GD
- Most common in practice
- Typical batch sizes: 32, 64, 128, 256

### 7.4 Momentum

**Update Rule:**
```
v := βv + (1-β) * ∇θ L
θ := θ - α * v
```

Where:
- `v` = velocity (exponentially weighted average of gradients)
- `β` = momentum coefficient (typically 0.9)

**Properties:**
- Accumulates gradient history
- Reduces oscillations
- Faster convergence
- Helps escape local minima

### 7.5 RMSprop

**Update Rule:**
```
s := βs + (1-β) * (∇θ L)²
θ := θ - α * ∇θ L / (√s + ε)
```

Where:
- `s` = exponentially weighted average of squared gradients
- `β` = decay rate (typically 0.9)
- `ε` = small constant (e.g., 1e-8)

**Properties:**
- Adapts learning rate per parameter
- Reduces oscillations in directions with large gradients
- Good for non-stationary objectives

### 7.6 Adam (Adaptive Moment Estimation)

**Update Rule:**
```
m := β₁m + (1-β₁) * ∇θ L      (first moment)
v := β₂v + (1-β₂) * (∇θ L)²   (second moment)
m̂ := m / (1 - β₁ᵗ)            (bias correction)
v̂ := v / (1 - β₂ᵗ)             (bias correction)
θ := θ - α * m̂ / (√v̂ + ε)
```

Where:
- `m` = first moment (mean)
- `v` = second moment (variance)
- `β₁` = first moment decay (typically 0.9)
- `β₂` = second moment decay (typically 0.999)
- `t` = iteration number

**Properties:**
- Combines Momentum and RMSprop
- Adaptive learning rates
- Bias correction for early iterations
- Most popular optimizer in practice
- Default hyperparameters work well

### 7.7 Learning Rate Scheduling

**Fixed Learning Rate:**
- Constant `α` throughout training

**Step Decay:**
```
α = α₀ * γ^(floor(epoch / step_size))
```

**Exponential Decay:**
```
α = α₀ * e^(-k * epoch)
```

**Cosine Annealing:**
```
α = α_min + (α_max - α_min) * (1 + cos(π * epoch / T)) / 2
```

---

## 📊 Summary: Algorithm Comparison

| Algorithm | Update Rule | Key Features | Use Cases |
|-----------|-------------|--------------|-----------|
| **SGD** | `θ := θ - α∇L` | Simple, noisy | Large datasets, online learning |
| **Momentum** | `v := βv + (1-β)∇L; θ := θ - αv` | Reduces oscillations | Faster convergence |
| **RMSprop** | `s := βs + (1-β)(∇L)²; θ := θ - α∇L/√s` | Adaptive per-parameter LR | Non-stationary objectives |
| **Adam** | Combines momentum + RMSprop | Adaptive, bias correction | **Default choice** for most cases |

---

## 🎯 Best Practices

1. **Initialization:**
   - Use Xavier/Glorot for tanh/sigmoid
   - Use He initialization for ReLU
   - Avoid initializing all weights to zero

2. **Activation Functions:**
   - Use ReLU or variants for hidden layers
   - Use task-specific activations for output layer

3. **Regularization:**
   - Start with L2 regularization
   - Add dropout for deeper networks
   - Use early stopping

4. **Optimization:**
   - Start with Adam optimizer
   - Use learning rate scheduling
   - Monitor training/validation loss

5. **Architecture:**
   - Start simple, then increase complexity
   - Use batch normalization for deeper networks
   - Consider residual connections for very deep networks

---

## 🔬 Mathematical Foundations

### Chain Rule

**For composite functions:**
```
If z = f(y) and y = g(x), then:
dz/dx = (dz/dy) * (dy/dx)
```

**For backpropagation:**
```
∂L/∂Wˡ = (∂L/∂aˡ) * (∂aˡ/∂zˡ) * (∂zˡ/∂Wˡ)
```

### Universal Approximation Theorem

**Statement:**
A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of ℝⁿ, given appropriate activation functions and weights.

**Implications:**
- Neural networks are universal function approximators
- One hidden layer is theoretically sufficient
- In practice, deeper networks are often more efficient

---

## 📚 Key Takeaways

1. **Linear Regression:** Foundation, convex optimization
2. **Logistic Regression:** Binary classification with probabilistic outputs
3. **Perceptron:** Simple binary classifier, converges for linearly separable data
4. **MLP:** Multi-layer networks with backpropagation
5. **Backpropagation:** Efficient gradient computation using chain rule
6. **Regularization:** Prevents overfitting (L1, L2, Dropout, Early Stopping)
7. **Optimization:** Various algorithms (SGD, Momentum, Adam) for efficient training

**Next Steps:**
- Convolutional Neural Networks (CNNs) for image data
- Recurrent Neural Networks (RNNs) for sequential data
- Advanced architectures (ResNet, Transformer, etc.)

