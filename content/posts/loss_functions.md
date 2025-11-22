+++
title = 'Loss Functions'
date = 2025-11-21T19:56:11+05:30
draft = false
+++

## 🧠 1. What is a Loss Function? (High-Level Overview)

A loss function measures the difference between the model's prediction and the ground truth.

It answers:
- ➡ How wrong is the model?
- ➡ In which direction should the parameters change?

During training, optimization tries to minimize the loss by updating weights (using gradients).

**Loss → Gradient → Weight Update → Better Predictions.**

---

## 🌳 2. Why do we need Loss Functions?

### ✔️ They quantify model error

Without a loss, we cannot measure how good or bad the model is.

### ✔️ They guide backpropagation

The derivative of the loss tells each weight:

> "Increase or decrease to reduce error?"

### ✔️ Different problems need different loss surfaces

**Example:**

- **Classification** → probability losses
- **Regression** → distance losses
- **Ranking** → pairwise or listwise losses
- **Sequence** → token-level cross entropy
- **Generative Models** → reconstruction losses, KL divergence, adversarial losses

---

## 🧰 How Are Loss Functions Used in Training?

**Training Loop:**

1. **Forward Pass:** Take input `x` → model predicts `ŷ = f(x; θ)`
2. **Compute Loss:** Calculate `L(y, ŷ)` where `y` is ground truth
3. **Backward Pass:** Compute gradients `∂L/∂θ` using backpropagation
4. **Update Weights:** Optimizer updates parameters: `θ ← θ - α * ∇L`
5. **Repeat:** Iterate over batches and epochs until convergence

**Key Components:**
- **Loss Function:** Quantifies error
- **Gradient:** Direction of steepest increase in loss
- **Optimizer:** Algorithm to update weights (SGD, Adam, etc.)
- **Learning Rate:** Step size for weight updates

**Mathematical Formulation:**
```
θ* = argmin_θ E[L(y, f(x; θ))]
```

The goal is to find parameters `θ` that minimize expected loss over the data distribution.

---

## 🧩 3. Types of Loss Functions in Deep Learning

We categorize them into 8 main groups.

### 🟦 1. Regression Losses

Used when predicting continuous values.

#### Mean Squared Error (MSE)

**Formula:**
```
L_MSE = (1/n) * Σ(y_i - ŷ_i)²
```

**Properties:**
- Penalizes large errors more heavily (quadratic penalty)
- Differentiable everywhere
- Sensitive to outliers
- Gradient: `∂L/∂ŷ = -2(y - ŷ)`

**When to use:**
- Errors are normally distributed
- Large errors should be heavily penalized
- Smooth gradients are needed

**Limitations:**
- Can be dominated by outliers
- May converge slowly near the optimum

---

#### Mean Absolute Error (MAE)

**Formula:**
```
L_MAE = (1/n) * Σ|y_i - ŷ_i|
```

**Properties:**
- Linear penalty for all errors
- Robust to outliers
- Less sensitive to large errors than MSE
- Gradient: `∂L/∂ŷ = -sign(y - ŷ)` (not smooth at zero)

**When to use:**
- Outliers are present in the data
- All errors should be treated equally
- Need robustness to noise

**Limitations:**
- Not differentiable at zero
- May converge slower than MSE

---

#### Huber Loss

**Formula:**
```
L_Huber = {
    (1/2) * (y - ŷ)²          if |y - ŷ| ≤ δ
    δ * |y - ŷ| - (1/2) * δ²  if |y - ŷ| > δ
}
```

Where `δ` (delta) is a hyperparameter (typically `1.0`).

**Properties:**
- Combines MSE (for small errors) and MAE (for large errors)
- Smooth and differentiable everywhere
- Robust to outliers like MAE
- Gradient: 
  - `∂L/∂ŷ = -(y - ŷ)` if `|y - ŷ| ≤ δ`
  - `∂L/∂ŷ = -δ * sign(y - ŷ)` if `|y - ŷ| > δ`

**When to use:**
- Want balanced behavior between MSE and MAE
- Need smooth gradients but also robustness
- Common in robust regression tasks

---

### 🟥 2. Classification Losses

Used when predicting discrete labels.

#### Binary Cross Entropy (BCE)

**Formula:**
```
L_BCE = -[y * log(p) + (1-y) * log(1-p)]
```

For a batch of n samples:
```
L_BCE = -(1/n) * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

**Properties:**
- Derived from maximum likelihood estimation
- Works with sigmoid activation (outputs probabilities)
- Gradient: `∂L/∂p = -(y/p) + (1-y)/(1-p) = (p - y)/(p(1-p))`
- Well-suited for binary classification

**When to use:**
- Binary classification problems
- Output layer uses sigmoid activation
- Need probabilistic outputs

**Note:** Add epsilon (`ε`) to log arguments to avoid `log(0) = -∞`

---

#### Categorical Cross Entropy (CCE)

**Formula:**
```
L_CCE = -Σ(y_i * log(p_i))
```

For a batch:
```
L_CCE = -(1/n) * Σ Σ(y_ij * log(p_ij))
```
where `i` indexes samples and `j` indexes classes.

**Properties:**
- Requires one-hot encoded labels
- Works with softmax activation
- Gradient: `∂L/∂p_i = -y_i/p_i`
- Minimizes the KL divergence between true and predicted distributions

**When to use:**
- Multi-class classification
- Output layer uses softmax activation
- Labels are one-hot encoded

---

#### Sparse Categorical Cross Entropy

**Formula:**
```
L_Sparse_CCE = -log(p_k)
```
where `k` is the true class index (integer label, not one-hot).

**Properties:**
- Same as CCE but accepts integer labels instead of one-hot
- More memory efficient (no need to one-hot encode)
- Computationally equivalent to CCE

**When to use:**
- Multi-class classification with integer labels
- Want to avoid one-hot encoding overhead
- Large number of classes (memory savings)

---

#### Focal Loss

**Formula:**
```
FL = -α * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t = p` if `y = 1`, else `p_t = 1 - p`
- `α` (alpha) = weighting factor for rare class (typically `0.25`)
- `γ` (gamma) = focusing parameter (typically `2.0`)

**Properties:**
- Addresses class imbalance by down-weighting easy examples
- `(1 - p_t)^γ` term reduces loss contribution from well-classified examples
- When `γ = 0`, reduces to standard cross-entropy
- Higher `γ` focuses more on hard examples

**When to use:**
- Highly imbalanced datasets
- Object detection (many background vs few objects)
- Need to focus on hard-to-classify examples

**Example:** In object detection, `γ=2` means easy negatives contribute `100x` less to loss than hard examples.

---

### 🟩 3. Ranking / Metric Learning Losses

Used for retrieval, search, similarity tasks.

#### Contrastive Loss

**Formula:**
```
L_contrastive = {
    (1/2) * d²                    if y = 1 (similar)
    (1/2) * max(0, margin - d)²   if y = 0 (dissimilar)
}
```

Where:
- `d = ||f(x_a) - f(x_b)||` (distance between embeddings)
- `margin` = minimum distance for dissimilar pairs (hyperparameter, typically `1.0`)

**Properties:**
- Pulls similar pairs together, pushes dissimilar pairs apart
- Used in Siamese networks
- Requires pairs of samples (similar/dissimilar labels)
- Gradient encourages embedding space separation

**When to use:**
- Learning similarity metrics
- Siamese network architectures
- Face recognition, signature verification
- Need to learn meaningful embeddings

---

#### Triplet Loss

**Formula:**
```
L_triplet = max(0, d(a,p) - d(a,n) + margin)
```

Where:
- `a` = anchor sample
- `p` = positive sample (same class as anchor)
- `n` = negative sample (different class from anchor)
- `d(a,p)` = distance between anchor and positive
- `d(a,n)` = distance between anchor and negative
- `margin` = minimum desired separation (typically `0.2-1.0`)

**Properties:**
- Ensures: `d(a,p) + margin < d(a,n)`
- Requires triplets: (anchor, positive, negative)
- Used in FaceNet, person re-identification
- More efficient than contrastive (one loss per triplet vs two pairs)

**When to use:**
- Face recognition (FaceNet)
- Person re-identification
- Learning embeddings where relative distances matter
- Need to ensure positive is closer than negative by margin

**Note:** Hard negative mining is crucial - easy triplets contribute zero loss.

---

#### Pairwise Ranking / Hinge Loss

**Formula:**
```
L_hinge = max(0, margin - (s_pos - s_neg))
```

Where:
- `s_pos` = score for positive/relevant item
- `s_neg` = score for negative/irrelevant item
- `margin` = desired score difference (typically `1.0`)

**Properties:**
- Used in Learning-to-Rank
- Encourages positive items to score higher than negatives
- Zero loss when score difference exceeds margin
- Common in recommendation systems, search ranking

**When to use:**
- Learning-to-Rank problems
- Recommendation systems
- Search result ranking
- Need to order items by relevance

---

### 🟪 4. Sequence Modeling Losses

Used in NLP, Transformers, LLMs.

#### Cross Entropy Loss (Token-level)

**Formula:**
```
L_CE = -(1/T) * Σ log(p(y_t | x_<t))
```

Where:
- `T` = sequence length
- `y_t` = true token at position `t`
- `x_<t` = context up to position `t`

**Properties:**
- Standard loss for language modeling
- Applied per token in sequence
- Used in GPT, BERT, T5, translation models
- Same as categorical cross-entropy but applied token-wise

**When to use:**
- Language modeling (next token prediction)
- Machine translation
- Text generation
- Any autoregressive sequence model

**Note:** Often combined with teacher forcing during training.

---

#### Label Smoothing Loss

**Formula:**
```
L_smooth = -(1/T) * Σ [y_t * log(p_t) + (1-y_t) * log(1-p_t)]
```

Where the true distribution is modified:
- Hard labels: `y_t = 1` for true class, `0` otherwise
- Smoothed labels: `y_t = (1 - α)` for true class, `α/(K-1)` for others
- `α` = smoothing factor (typically `0.1`)
- `K` = number of classes

**Properties:**
- Prevents overconfidence (logits don't become too extreme)
- Regularization effect
- Improves generalization
- Used in BERT, GPT-2, many modern LLMs

**When to use:**
- Models becoming overconfident
- Need better calibration
- Want regularization without dropout
- Large language models

**Example:** With `α=0.1` and `K=1000`, true class gets `0.9`, others get `0.1/999 ≈ 0.0001`

---

#### KL Divergence

**Formula:**
```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```

For continuous distributions:
```
KL(P || Q) = ∫ p(x) * log(p(x) / q(x)) dx
```

**Properties:**
- Measures difference between two probability distributions
- Asymmetric: `KL(P||Q) ≠ KL(Q||P)`
- Always non-negative, zero when `P = Q`
- Used in VAEs (regularization term) and knowledge distillation

**When to use:**
- **VAEs:** Regularize latent space to match prior (typically N(0,1))
- **Knowledge Distillation:** Match student to teacher distribution
- **Variational inference:** Approximate posterior to prior

**VAE Example:**
```
L_VAE = Reconstruction_Loss + β * KL(q(z|x) || p(z))
```
where `β` controls regularization strength (β-VAE).

---

### 🟫 5. Image Losses

#### Dice Loss

**Formula:**
```
Dice_Score = (2 * |X ∩ Y|) / (|X| + |Y|)
Dice_Loss = 1 - Dice_Score
```

In terms of predictions and ground truth:
```
Dice_Loss = 1 - (2 * Σ(p_i * y_i) + ε) / (Σ(p_i) + Σ(y_i) + ε)
```

Where:
- `p_i` = predicted probability for pixel `i`
- `y_i` = ground truth label for pixel `i` (`0` or `1`)
- `ε` = small constant to avoid division by zero (typically `1e-5`)

**Properties:**
- Measures overlap between predicted and true masks
- Range: [0, 1], where 0 = perfect overlap
- Handles class imbalance well (focuses on foreground)
- Differentiable approximation of Dice coefficient

**When to use:**
- Medical image segmentation
- Imbalanced segmentation tasks (small objects)
- UNet, DeepLab architectures
- When IoU is the evaluation metric

**Advantages:**
- Less sensitive to class imbalance than BCE
- Directly optimizes overlap metric

---

#### IoU Loss (Jaccard Loss)

**Formula:**
```
IoU = |X ∩ Y| / |X ∪ Y|
IoU_Loss = 1 - IoU
```

In terms of predictions:
```
IoU_Loss = 1 - (Σ(p_i * y_i) + ε) / (Σ(p_i + y_i - p_i * y_i) + ε)
```

**Properties:**
- Intersection over Union metric
- Range: [0, 1]
- Standard evaluation metric for segmentation
- Penalizes both false positives and false negatives

**When to use:**
- Segmentation tasks where IoU is the evaluation metric
- Need to directly optimize IoU
- Object detection (bounding box IoU)

**Comparison with Dice:**
- IoU is more strict (penalizes more for errors)
- Dice is more forgiving for small errors
- Both handle class imbalance

---

#### Combined Losses: BCE + Dice / BCE + IoU

**Formula:**
```
L_combined = α * L_BCE + (1 - α) * L_Dice
```

Or:
```
L_combined = α * L_BCE + (1 - α) * L_IoU
```

Where `α` is a weighting factor (typically `0.5`).

**Properties:**
- Combines pixel-wise (BCE) and region-wise (Dice/IoU) losses
- BCE provides stable gradients
- Dice/IoU directly optimizes overlap metric
- Common in UNet, DeepLab, and modern segmentation models

**When to use:**
- Medical image segmentation
- UNet-based architectures
- Want benefits of both losses
- Need stable training with direct metric optimization

**Typical values:** `α = 0.5` (equal weighting) or `α = 0.7` (more weight on BCE)

---

### 🟧 6. Generative / Autoencoder Losses

#### Minimax Loss (Original GAN)

**Formula:**

**Generator Loss:**
```
L_G = -E[log(D(G(z)))]
```

**Discriminator Loss:**
```
L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]
```

Where:
- `D(x)` = discriminator output for real data
- `G(z)` = generator output from noise `z`
- `D(G(z))` = discriminator output for fake data

**Properties:**
- Two-player minimax game
- Generator tries to fool discriminator
- Discriminator tries to distinguish real from fake
- Training can be unstable (vanishing gradients)

**When to use:**
- Original GAN formulation
- Understanding GAN fundamentals
- Not recommended for production (use improved versions)

**Issues:**
- Vanishing gradients when discriminator is too good
- Mode collapse (generator produces limited diversity)
- Training instability

---

#### Least Squares GAN (LSGAN)

**Formula:**

**Generator Loss:**
```
L_G = (1/2) * E[(D(G(z)) - 1)²]
```

**Discriminator Loss:**
```
L_D = (1/2) * E[(D(x) - 1)²] + (1/2) * E[D(G(z))²]
```

**Properties:**
- Uses L2 loss instead of cross-entropy
- More stable gradients
- Penalizes samples far from decision boundary
- Reduces mode collapse compared to original GAN

**When to use:**
- Need more stable GAN training
- Want to avoid vanishing gradients
- Image generation tasks

**Advantages:**
- Smoother loss landscape
- Better gradient flow
- More stable training

---

#### Wasserstein Loss (WGAN)

**Formula:**

**Generator Loss:**
```
L_G = -E[D(G(z))]
```

**Discriminator Loss (Critic):**
```
L_D = E[D(G(z))] - E[D(x)]
```

With gradient penalty (WGAN-GP):
```
L_D = E[D(G(z))] - E[D(x)] + λ * E[(||∇D(x̂)|| - 1)²]
```

Where:
- `x̂` = random interpolation between real and fake samples
- `λ` = gradient penalty weight (typically `10`)
- Discriminator (called "critic") must be `1-Lipschitz`

**Properties:**
- Measures Wasserstein-1 distance (Earth Mover's Distance)
- Provides meaningful training signal even when discriminator is optimal
- More stable than original GAN
- Requires weight clipping (WGAN) or gradient penalty (WGAN-GP)

**When to use:**
- Need stable GAN training
- Want to avoid mode collapse
- Image generation, style transfer
- When discriminator accuracy is not meaningful

**Advantages:**
- Stable gradients throughout training
- Meaningful loss value (correlates with sample quality)
- Less mode collapse
- Better convergence properties

---

### 🟨 7. Reinforcement Learning Losses

#### Policy Gradient Loss (REINFORCE)

**Formula:**
```
L_PG = -E[log(π(a|s)) * A(s,a)]
```

Where:
- `π(a|s)` = policy probability of action `a` given state `s`
- `A(s,a) = Q(s,a) - V(s)` = advantage function
- `Q(s,a)` = action-value function
- `V(s)` = state-value function (baseline)

**Properties:**
- Maximizes expected return
- Uses advantage to reduce variance
- On-policy algorithm
- High variance (requires many samples)

**When to use:**
- Policy gradient methods (REINFORCE, Actor-Critic)
- Continuous/discrete action spaces
- Need to learn stochastic policies

**Note:** Baseline (V(s)) reduces variance without introducing bias.

---

#### Value Function Loss

**Formula:**
```
L_V = E[(V(s) - R)^2]
```

Where:
- `V(s)` = predicted state value
- `R` = actual return (discounted sum of rewards)

For TD learning:
```
L_V = E[(V(s) - (r + γ * V(s')))^2]
```

Where:
- `r` = immediate reward
- `γ` = discount factor
- `s'` = next state

**Properties:**
- Regression loss (typically MSE)
- Learns to predict expected returns
- Used in Actor-Critic, DQN, A3C
- Provides baseline for policy gradient

**When to use:**
- Value-based methods (DQN)
- Actor-Critic architectures
- Need to estimate state/action values
- Baseline estimation for policy gradients

---

#### PPO Loss (Clipped)

**Formula:**
```
L_PPO = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
```

Where:
- `r(θ) = π_θ(a|s) / π_θ_old(a|s)` (importance sampling ratio)
- `A` = advantage estimate
- `ε` = clipping parameter (typically `0.1-0.3`)

**Properties:**
- Clips policy updates to prevent large changes
- Prevents policy from moving too far from old policy
- More stable than vanilla policy gradient
- State-of-the-art for many RL tasks

**When to use:**
- Need stable policy gradient training
- Continuous control tasks
- When sample efficiency matters
- Modern RL applications (robotics, games)

**Advantages:**
- Prevents destructive policy updates
- Better sample efficiency
- Easier to tune than TRPO
- Works well with function approximation

**Typical hyperparameters:**
- `ε = 0.2` (clipping range)
- Learning rate = `3e-4`
- Multiple epochs per batch (typically `4-10`)

---

### 🟩 8. Autoencoder & Reconstruction Losses

#### 8.1 Reconstruction Loss

**Formula:**

**MSE Reconstruction:**
```
L_recon_MSE = (1/n) * Σ(x_i - x̂_i)²
```

**L1 Reconstruction:**
```
L_recon_L1 = (1/n) * Σ|x_i - x̂_i|
```

Where:
- `x_i` = original input
- `x̂_i` = reconstructed output

**Properties:**
- Measures how well the model reconstructs input
- MSE: smooth, penalizes large errors quadratically
- L1: robust to outliers, linear penalty
- Used in standard autoencoders, denoising autoencoders

**When to use:**
- **MSE:** When input is continuous, normally distributed errors
- **L1:** When input has outliers, need robustness
- Standard autoencoders, denoising tasks
- Compression, dimensionality reduction

**Note:** For images, can also use perceptual losses (VGG features) instead of pixel-wise losses.

---

#### 8.2 KL Divergence (VAE)

**Formula:**

**VAE Total Loss:**
```
L_VAE = L_recon + β * L_KL
```

Where:
```
L_KL = KL(q(z|x) || p(z))
```

For Gaussian prior and posterior:
```
L_KL = (1/2) * Σ(σ² + μ² - 1 - log(σ²))
```

Where:
- `q(z|x)` = encoder output (posterior): `N(μ, σ²)`
- `p(z)` = prior: `N(0, I)`
- `μ, σ` = encoder outputs (mean and log-variance)
- `β` = regularization strength (β-VAE, typically `1.0`)

**Properties:**
- Regularizes latent space to match prior distribution
- Encourages smooth, continuous latent space
- Prevents posterior collapse (encoder ignores input)
- Enables sampling and interpolation in latent space

**When to use:**
- Variational Autoencoders (VAEs)
- Need to sample from latent space
- Want smooth latent representations
- Generative modeling with continuous latent variables

**β-VAE:**
- `β > 1`: stronger regularization, better disentanglement
- `β < 1`: weaker regularization, better reconstruction
- `β = 1`: standard VAE

**Common Issues:**
- **Posterior collapse:** `β` too large → encoder ignores input
- **Blurry reconstructions:** Common with MSE reconstruction
- **KL vanishing:** `β` too small → latent space not regularized

---

## 📊 Summary: Quick Reference Guide

### Loss Function Selection by Task

| Task Type | Recommended Loss | Key Considerations |
|-----------|-----------------|-------------------|
| **Regression** | MSE, MAE, Huber | MSE for normal errors, MAE for outliers, Huber for balance |
| **Binary Classification** | Binary Cross-Entropy | Use with sigmoid activation |
| **Multi-class Classification** | Categorical Cross-Entropy | Use with softmax activation |
| **Imbalanced Classification** | Focal Loss | Adjust γ and α for class imbalance |
| **Image Segmentation** | Dice Loss, IoU Loss, BCE+Dice | Dice/IoU for overlap, combined for stability |
| **Language Modeling** | Cross-Entropy (token-level) | Standard for next-token prediction |
| **Similarity Learning** | Contrastive Loss, Triplet Loss | Contrastive for pairs, Triplet for relative distances |
| **GAN Training** | Wasserstein Loss (WGAN-GP) | Most stable, use gradient penalty |
| **VAE** | Reconstruction + KL Divergence | Balance β for reconstruction vs regularization |
| **Reinforcement Learning** | PPO Loss | Clipped for stability |

### Key Properties to Consider

1. **Differentiability:** Most losses need to be differentiable for backpropagation
2. **Robustness:** Some losses (MAE, Huber) are more robust to outliers
3. **Gradient Behavior:** Smooth gradients (MSE) vs non-smooth (MAE at zero)
4. **Scale Sensitivity:** Some losses are scale-dependent, others are scale-invariant
5. **Class Imbalance:** Focal Loss, Dice Loss handle imbalance better
6. **Direct Optimization:** Dice/IoU directly optimize evaluation metrics

### Common Patterns

- **Combined Losses:** Often combine multiple losses (e.g., BCE + Dice, Reconstruction + KL)
- **Weighted Losses:** Use weighting factors (α, β) to balance different objectives
- **Adaptive Losses:** Some losses adapt during training (e.g., curriculum learning)
- **Task-Specific:** Choose loss that matches your evaluation metric when possible

### Best Practices

1. **Match Loss to Metric:** If evaluating with IoU, use IoU loss
2. **Handle Edge Cases:** Add epsilon to logarithms, handle division by zero
3. **Normalize Appropriately:** Consider batch normalization, loss normalization
4. **Monitor Training:** Watch for vanishing/exploding gradients
5. **Experiment:** Try different losses and combinations for your specific problem
