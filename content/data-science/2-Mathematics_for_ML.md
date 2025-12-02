+++
title = "Mathematics for Machine Learning"
date = 2025-11-22T10:00:00+05:30
draft = false
weight = 2
description = "Essential mathematical foundations for machine learning. Covers linear algebra, calculus, probability, statistics, and optimization. Includes derivatives, gradients, matrix operations, and commonly used equations in ML algorithms."
+++

## 1. Introduction

Machine learning is built on mathematical foundations. Understanding these concepts is crucial for:
- Implementing algorithms from scratch
- Understanding how models learn and optimize
- Debugging and improving model performance
- Choosing appropriate algorithms for problems

This guide covers the essential mathematics needed for ML, with practical examples and applications.

---

## 2. Linear Algebra

### Vectors

**Definition**: A vector is an ordered list of numbers, representing a point in n-dimensional space.

**Notation**: 
```
v = [v₁, v₂, ..., vₙ]
```

**Operations**:

**Vector Addition**:
```
u + v = [u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ]
```

**Scalar Multiplication**:
```
c · v = [c·v₁, c·v₂, ..., c·vₙ]
```

**Dot Product**:
```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ = Σ(i=1 to n) uᵢvᵢ
```

**Vector Norm (Length)**:
```
||v|| = √(v₁² + v₂² + ... + vₙ²) = √(v · v)
```

**Use Cases in ML**:
- Feature vectors
- Weight vectors in neural networks
- Representing data points

### Matrices

**Definition**: A matrix is a rectangular array of numbers arranged in rows and columns.

**Notation**: 
```
A = [aᵢⱼ] where i = row, j = column
```

**Matrix Operations**:

**Matrix Addition**:
```
(A + B)ᵢⱼ = aᵢⱼ + bᵢⱼ
```

**Matrix Multiplication**:
```
(AB)ᵢⱼ = Σ(k=1 to p) aᵢₖbₖⱼ
```
Where A is m×p and B is p×n, result is m×n.

**Matrix Transpose**:
```
(Aᵀ)ᵢⱼ = aⱼᵢ
```

**Matrix Inverse** (for square matrices):
```
A⁻¹ such that A·A⁻¹ = I (identity matrix)
```

**Use Cases in ML**:
- Weight matrices in neural networks
- Data transformations
- Principal Component Analysis (PCA)
- Linear regression: `y = Xw + b`

### Eigenvalues and Eigenvectors

**Definition**: For a square matrix A, an eigenvector v and eigenvalue λ satisfy:
```
Av = λv
```

**Properties**:
- Eigenvalues represent variance in PCA
- Used in dimensionality reduction
- Important for understanding matrix transformations

**Use Cases in ML**:
- Principal Component Analysis (PCA)
- Spectral clustering
- Understanding model behavior

---

## 3. Calculus

### Derivatives

**Definition**: The derivative measures how a function changes as its input changes.

**Notation**: 
```
f'(x) = df/dx = lim(h→0) [f(x+h) - f(x)] / h
```

**Common Derivatives**:

**Power Rule**:
```
d/dx(xⁿ) = nxⁿ⁻¹
```

**Exponential**:
```
d/dx(eˣ) = eˣ
d/dx(aˣ) = aˣ ln(a)
```

**Logarithm**:
```
d/dx(ln(x)) = 1/x
d/dx(log_a(x)) = 1/(x ln(a))
```

**Trigonometric**:
```
d/dx(sin(x)) = cos(x)
d/dx(cos(x)) = -sin(x)
```

**Chain Rule**:
```
d/dx(f(g(x))) = f'(g(x)) · g'(x)
```

**Product Rule**:
```
d/dx(f(x)·g(x)) = f'(x)·g(x) + f(x)·g'(x)
```

**Quotient Rule**:
```
d/dx(f(x)/g(x)) = [f'(x)·g(x) - f(x)·g'(x)] / [g(x)]²
```

### Partial Derivatives

**Definition**: For a function of multiple variables, the partial derivative measures change with respect to one variable while keeping others constant.

**Notation**:
```
∂f/∂xᵢ = derivative of f with respect to xᵢ
```

**Example**:
```
f(x, y) = x²y + 3xy²
∂f/∂x = 2xy + 3y²
∂f/∂y = x² + 6xy
```

### Gradient

**Definition**: The gradient is a vector of all partial derivatives.

**Notation**:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Properties**:
- Points in direction of steepest ascent
- Zero gradient indicates critical point (min/max/saddle)
- Used in optimization algorithms

**Use Cases in ML**:
- Gradient descent optimization
- Backpropagation in neural networks
- Finding optimal parameters

### Gradient Descent

**Mathematical Formulation**:
```
θ_new = θ_old - α · ∇J(θ_old)
```

Where:
- `θ` = parameters (weights)
- `α` = learning rate
- `J(θ)` = cost/loss function
- `∇J(θ)` = gradient of cost function

**Intuition**: Move in the direction opposite to the gradient (steepest descent) to minimize the cost function.

---

## 4. Probability and Statistics

### Probability Basics

**Probability of Event A**:
```
P(A) = number of favorable outcomes / total outcomes
```

**Conditional Probability**:
```
P(A|B) = P(A ∩ B) / P(B)
```

**Bayes' Theorem**:
```
P(A|B) = [P(B|A) · P(A)] / P(B)
```

**Independence**:
```
P(A ∩ B) = P(A) · P(B)  (if A and B are independent)
```

### Expected Value

**Definition**: The expected value is the average value of a random variable.

**Discrete**:
```
E[X] = Σ(i=1 to n) xᵢ · P(xᵢ)
```

**Continuous**:
```
E[X] = ∫ x · f(x) dx
```

**Properties**:
```
E[aX + b] = aE[X] + b
E[X + Y] = E[X] + E[Y]
```

### Variance and Standard Deviation

**Variance**:
```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

**Standard Deviation**:
```
σ = √Var(X)
```

**Properties**:
```
Var(aX + b) = a²Var(X)
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)
```

### Covariance and Correlation

**Covariance**:
```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

**Correlation Coefficient**:
```
ρ = Cov(X, Y) / (σₓ · σᵧ)
```

**Range**: -1 to +1
- +1: Perfect positive correlation
- 0: No correlation
- -1: Perfect negative correlation

---

## 5. Optimization

### Cost Functions

**Mean Squared Error (MSE)**:
```
MSE = (1/n) Σ(i=1 to n) (yᵢ - ŷᵢ)²
```

**Mean Absolute Error (MAE)**:
```
MAE = (1/n) Σ(i=1 to n) |yᵢ - ŷᵢ|
```

**Cross-Entropy Loss** (for classification):
```
L = -Σ(i=1 to n) [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Hinge Loss** (for SVM):
```
L = max(0, 1 - yᵢ(w·xᵢ + b))
```

### Gradient Descent Variants

**Batch Gradient Descent**:
```
θ = θ - α · (1/m) Σ(i=1 to m) ∇J(θ; xᵢ, yᵢ)
```

**Stochastic Gradient Descent (SGD)**:
```
θ = θ - α · ∇J(θ; xᵢ, yᵢ)  (for single example i)
```

**Mini-Batch Gradient Descent**:
```
θ = θ - α · (1/b) Σ(i=1 to b) ∇J(θ; xᵢ, yᵢ)  (b = batch size)
```

### Learning Rate

**Fixed Learning Rate**:
```
α = constant (e.g., 0.01, 0.001)
```

**Adaptive Learning Rate** (Adam optimizer):
```
m_t = β₁m_(t-1) + (1-β₁)g_t
v_t = β₂v_(t-1) + (1-β₂)g_t²
θ_t = θ_(t-1) - α · m_t / (√v_t + ε)
```

Where:
- `g_t` = gradient at time t
- `β₁, β₂` = momentum parameters (typically 0.9, 0.999)
- `ε` = small constant (e.g., 10⁻⁸)

---

## 6. Commonly Used Equations in ML

### Linear Regression

**Hypothesis**:
```
h(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx + b
```

**Cost Function**:
```
J(w) = (1/2m) Σ(i=1 to m) (h(xᵢ) - yᵢ)²
```

**Gradient**:
```
∂J/∂wⱼ = (1/m) Σ(i=1 to m) (h(xᵢ) - yᵢ) · xᵢⱼ
```

### Logistic Regression

**Sigmoid Function**:
```
σ(z) = 1 / (1 + e^(-z))
```

**Hypothesis**:
```
h(x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))
```

**Cost Function**:
```
J(w) = -(1/m) Σ(i=1 to m) [yᵢ log(h(xᵢ)) + (1-yᵢ) log(1-h(xᵢ))]
```

**Gradient**:
```
∂J/∂wⱼ = (1/m) Σ(i=1 to m) (h(xᵢ) - yᵢ) · xᵢⱼ
```

### Neural Networks

**Forward Propagation**:
```
z^[l] = W^[l]a^[l-1] + b^[l]
a^[l] = g(z^[l])
```

Where:
- `l` = layer number
- `W^[l]` = weight matrix for layer l
- `a^[l]` = activation for layer l
- `g` = activation function

**Backpropagation**:
```
δ^[L] = ∇_a J · g'(z^[L])  (output layer)
δ^[l] = (W^[l+1])ᵀδ^[l+1] · g'(z^[l])  (hidden layers)
∂J/∂W^[l] = δ^[l](a^[l-1])ᵀ
∂J/∂b^[l] = δ^[l]
```

### Support Vector Machine (SVM)

**Objective Function**:
```
minimize: (1/2)||w||² + C Σ(i=1 to m) ξᵢ
subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

Where:
- `C` = regularization parameter
- `ξᵢ` = slack variables

**Decision Function**:
```
f(x) = sign(wᵀx + b)
```

### Principal Component Analysis (PCA)

**Covariance Matrix**:
```
Σ = (1/n) XᵀX
```

**Eigendecomposition**:
```
Σ = PΛPᵀ
```

Where:
- `P` = eigenvectors (principal components)
- `Λ` = eigenvalues (variance explained)

**Projection**:
```
Y = XP_k
```

Where `P_k` contains top k eigenvectors.

---

## 7. Activation Functions

### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
```

**Derivative**: `σ'(x) = σ(x)(1 - σ(x))`

**Derivation**:

Using the quotient rule: `d/dx(u/v) = (u'v - uv') / v²`

Let `u = 1` and `v = 1 + e^(-x)`

Then:
- `u' = 0`
- `v' = -e^(-x)`

```
σ'(x) = [0 · (1 + e^(-x)) - 1 · (-e^(-x))] / (1 + e^(-x))²
      = e^(-x) / (1 + e^(-x))²
```

Now, factor out `1 / (1 + e^(-x))`:
```
σ'(x) = [1 / (1 + e^(-x))] · [e^(-x) / (1 + e^(-x))]
      = σ(x) · [e^(-x) / (1 + e^(-x))]
```

Note that:
```
e^(-x) / (1 + e^(-x)) = (1 + e^(-x) - 1) / (1 + e^(-x))
                      = 1 - 1/(1 + e^(-x))
                      = 1 - σ(x)
```

Therefore:
```
σ'(x) = σ(x) · (1 - σ(x))
```

### Tanh
```
tanh(x) = (eˣ - e^(-x)) / (eˣ + e^(-x))
```
**Derivative**: `tanh'(x) = 1 - tanh²(x)`

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x) = {x if x > 0, 0 if x ≤ 0}
```
**Derivative**: `ReLU'(x) = {1 if x > 0, 0 if x ≤ 0}`

### Leaky ReLU
```
LeakyReLU(x) = max(αx, x) = {x if x > 0, αx if x ≤ 0}
```
Where `α` is a small constant (e.g., 0.01)

### Softmax (for multi-class classification)
```
softmax(xᵢ) = e^(xᵢ) / Σ(j=1 to k) e^(xⱼ)
```

---

## 8. Regularization

### L1 Regularization (Lasso)
```
J(w) = MSE + λ Σ(i=1 to n) |wᵢ|
```

**Gradient**:
```
∂J/∂wᵢ = ∂MSE/∂wᵢ + λ · sign(wᵢ)
```

### L2 Regularization (Ridge)
```
J(w) = MSE + λ Σ(i=1 to n) wᵢ²
```

**Gradient**:
```
∂J/∂wᵢ = ∂MSE/∂wᵢ + 2λwᵢ
```

### Elastic Net
```
J(w) = MSE + λ₁ Σ(i=1 to n) |wᵢ| + λ₂ Σ(i=1 to n) wᵢ²
```

---

## 9. Distance Metrics

### Euclidean Distance
```
d(x, y) = √(Σ(i=1 to n) (xᵢ - yᵢ)²)
```

### Manhattan Distance (L1)
```
d(x, y) = Σ(i=1 to n) |xᵢ - yᵢ|
```

### Cosine Similarity
```
cos(θ) = (x · y) / (||x|| · ||y||)
```

### Minkowski Distance
```
d(x, y) = (Σ(i=1 to n) |xᵢ - yᵢ|^p)^(1/p)
```
- p=1: Manhattan
- p=2: Euclidean
- p=∞: Chebyshev

---

## 10. Information Theory

### Entropy
```
H(X) = -Σ(i=1 to n) P(xᵢ) log₂(P(xᵢ))
```

### Cross-Entropy
```
H(P, Q) = -Σ(i=1 to n) P(xᵢ) log(Q(xᵢ))
```

### Kullback-Leibler Divergence
```
KL(P||Q) = Σ(i=1 to n) P(xᵢ) log(P(xᵢ) / Q(xᵢ))
```

### Mutual Information
```
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

---

## 11. Matrix Decompositions

### Singular Value Decomposition (SVD)
```
A = UΣVᵀ
```

Where:
- `U` = left singular vectors
- `Σ` = singular values (diagonal matrix)
- `V` = right singular vectors

**Use Cases**:
- Dimensionality reduction
- Matrix approximation
- Recommender systems

### QR Decomposition
```
A = QR
```

Where:
- `Q` = orthogonal matrix
- `R` = upper triangular matrix

**Use Cases**:
- Solving linear systems
- Least squares problems

---

## 12. Summary

### Key Mathematical Concepts for ML

1. **Linear Algebra**: Vectors, matrices, eigenvalues - foundation for data representation
2. **Calculus**: Derivatives, gradients - essential for optimization
3. **Probability**: Distributions, Bayes' theorem - understanding uncertainty
4. **Statistics**: Mean, variance, correlation - data analysis
5. **Optimization**: Gradient descent, cost functions - model training

### Common Patterns

- **Gradient-based optimization**: Use derivatives to find minima
- **Matrix operations**: Efficient computation for large datasets
- **Probability distributions**: Model uncertainty and make predictions
- **Distance metrics**: Measure similarity between data points

### Practical Tips

1. **Understand the intuition** behind each concept
2. **Practice with examples** - implement algorithms from scratch
3. **Visualize** - graphs help understand gradients, distributions
4. **Start simple** - master basics before advanced topics
5. **Apply to real problems** - use math to solve ML challenges

Mastering these mathematical foundations will help you understand, implement, and improve machine learning algorithms effectively.

