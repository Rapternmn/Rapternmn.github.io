+++
title = "Convolutional Neural Networks (CNN): Architectures & Algorithms"
date = 2025-11-22T11:00:00+05:30
draft = false
weight = 1
description = "Comprehensive guide to Convolutional Neural Networks covering convolution operations, pooling, classic architectures (LeNet, AlexNet, VGG), modern architectures (ResNet, Inception, DenseNet, MobileNet), and advanced techniques for image processing."
+++

# 🖼️ Convolutional Neural Networks (CNN): Architectures & Algorithms

Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed for processing grid-like data such as images. This document covers fundamental CNN concepts, key architectures, mathematical formulations, algorithms, and modern techniques.

**Key Concepts:**
- Convolution operation
- Pooling layers
- Feature hierarchies
- Transfer learning
- Modern architectures (ResNet, Inception, etc.)

---

## 🔷 1. Basic Convolution Operation

Convolution is the fundamental operation in CNNs that applies learnable filters to input data to extract features.

### Mathematical Formulation

**2D Convolution (for images):**

Given an input image `I` of size `[H × W]` and a filter/kernel `K` of size `[k × k]`:

```
(I * K)[i, j] = Σₘ Σₙ I[i + m, j + n] · K[m, n]
```

Where:
- `*` = convolution operator
- `[i, j]` = output position
- `[m, n]` = filter position
- Summation is over all valid positions where filter overlaps with input

**For multi-channel input (e.g., RGB image):**

Input: `I` of size `[H × W × C_in]`
Filter: `K` of size `[k × k × C_in]`

```
(I * K)[i, j] = Σ_c Σₘ Σₙ I[i + m, j + n, c] · K[m, n, c]
```

**For multiple filters (output channels):**

With `C_out` filters, output size: `[H_out × W_out × C_out]`

### Convolution Parameters

**1. Padding (P):**
- **Valid (no padding):** `P = 0`
- **Same (zero padding):** `P = (k - 1) / 2` (for odd kernel size)
- Output size: `H_out = H + 2P - k + 1`

**2. Stride (S):**
- Controls step size of filter movement
- Output size: `H_out = ⌊(H + 2P - k) / S⌋ + 1`

**3. Dilation (D):**
- Introduces gaps between filter elements
- Effective kernel size: `k_eff = k + (k - 1) · (D - 1)`
- Output size: `H_out = ⌊(H + 2P - k_eff) / S⌋ + 1`

### Output Size Formula

**General formula:**
```
H_out = ⌊(H_in + 2P - k) / S⌋ + 1
W_out = ⌊(W_in + 2P - k) / S⌋ + 1
```

**Example:**
- Input: `224 × 224`
- Kernel: `3 × 3`
- Padding: `P = 1`
- Stride: `S = 1`
- Output: `⌊(224 + 2 - 3) / 1⌋ + 1 = 224 × 224`

### Parameter Count

For a convolutional layer:
- Input: `[H × W × C_in]`
- Output: `[H_out × W_out × C_out]`
- Kernel: `[k × k × C_in × C_out]`

**Parameters:** `k × k × C_in × C_out + C_out` (including biases)

**Example:**
- `C_in = 3`, `C_out = 64`, `k = 3`
- Parameters: `3 × 3 × 3 × 64 + 64 = 1,792`

### Properties

**Advantages:**
- **Parameter sharing:** Same filter applied across all spatial locations
- **Translation invariance:** Can detect features regardless of position
- **Sparse connectivity:** Each output depends on small local region
- **Efficient:** Much fewer parameters than fully connected layers

**Key Insight:**
Convolution exploits the **stationarity** assumption: features useful in one location are useful everywhere.

---

## 🔺 2. Pooling Layers

Pooling layers reduce spatial dimensions while maintaining depth, providing translation invariance and reducing computational cost.

### Max Pooling

**Operation:**
```
y[i, j] = max(I[i·S : i·S + k, j·S : j·S + k])
```

Where:
- `k` = pool size (typically `2 × 2`)
- `S` = stride (typically equals pool size)

**Properties:**
- **Translation invariance:** Small shifts don't affect output
- **Dimensionality reduction:** Reduces spatial size
- **Non-parametric:** No learnable parameters

### Average Pooling

**Operation:**
```
y[i, j] = (1/k²) · Σₘ Σₙ I[i·S + m, j·S + n]
```

**Properties:**
- **Smoother:** Less sensitive to outliers than max pooling
- **Preserves average:** Maintains average activation

### Global Average Pooling (GAP)

**Operation:**
```
y[c] = (1/(H × W)) · Σᵢ Σⱼ I[i, j, c]
```

**Properties:**
- **Reduces to 1D:** `[H × W × C] → [C]`
- **No parameters:** Fully parameter-free
- **Prevents overfitting:** Used in modern architectures (ResNet, etc.)

### Adaptive Pooling

**Operation:**
Dynamically adjusts pool size to produce fixed output size:
```
Output size: [H_out × W_out] (fixed)
Pool size: k = ⌈H_in / H_out⌉
```

**Properties:**
- **Flexible input sizes:** Can handle variable input dimensions
- **Fixed output:** Always produces same output size

---

## 🏗️ 3. CNN Architecture Components

### Typical CNN Structure

```
Input → [Conv → ReLU → Pool] × N → [FC → ReLU] × M → Output
```

**Common pattern:**
1. **Feature extraction:** Convolutional layers extract hierarchical features
2. **Dimensionality reduction:** Pooling layers reduce spatial size
3. **Classification:** Fully connected layers perform final classification

### Activation Functions in CNNs

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
```

**Properties:**
- **Sparsity:** Produces sparse activations
- **Non-saturating:** Avoids vanishing gradients
- **Fast:** Simple computation

**Variants:**
- **Leaky ReLU:** `f(x) = max(αx, x)` where `α ≈ 0.01`
- **PReLU:** Learnable `α` parameter
- **ELU:** `f(x) = {x if x > 0, α(e^x - 1) if x ≤ 0}`

### Batch Normalization

**Operation:**
```
μ_B = (1/m) · Σᵢ xᵢ
σ²_B = (1/m) · Σᵢ (xᵢ - μ_B)²
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
yᵢ = γ · x̂ᵢ + β
```

Where:
- `γ`, `β` = learnable parameters
- `ε` = small constant (typically `1e-5`)

**Properties:**
- **Internal covariate shift:** Reduces distribution shift during training
- **Faster training:** Allows higher learning rates
- **Regularization:** Acts as implicit regularization
- **Placement:** Usually after convolution, before activation

---

## 📐 4. Classic CNN Architectures

### LeNet-5 (1998)

**Architecture:**
```
Input (32×32) 
→ Conv1 (6 filters, 5×5) → AvgPool (2×2)
→ Conv2 (16 filters, 5×5) → AvgPool (2×2)
→ FC1 (120) → FC2 (84) → Output (10)
```

**Key Features:**
- First successful CNN for digit recognition
- Used average pooling (now less common)
- Small architecture by modern standards

**Parameters:** ~60K

### AlexNet (2012)

**Architecture:**
```
Input (224×224×3)
→ Conv1 (96, 11×11, stride=4) → ReLU → MaxPool (3×3)
→ Conv2 (256, 5×5, pad=2) → ReLU → MaxPool (3×3)
→ Conv3 (384, 3×3, pad=1) → ReLU
→ Conv4 (384, 3×3, pad=1) → ReLU
→ Conv5 (256, 3×3, pad=1) → ReLU → MaxPool (3×3)
→ FC1 (4096) → Dropout → FC2 (4096) → Dropout → FC3 (1000)
```

**Key Innovations:**
- **ReLU activation:** First major use of ReLU
- **Dropout:** Regularization technique
- **Data augmentation:** Increased effective dataset size
- **GPU training:** Enabled training on GPUs
- **Local Response Normalization (LRN):** Now less common

**Parameters:** ~60M

**Impact:**
- Won ImageNet 2012 with significant margin
- Sparked deep learning revolution

### VGG (2014)

**Architecture Variants:**
- VGG-11, VGG-13, VGG-16, VGG-19

**VGG-16 Structure:**
```
Input (224×224×3)
→ Conv Block 1: [Conv(64, 3×3)]×2 → MaxPool
→ Conv Block 2: [Conv(128, 3×3)]×2 → MaxPool
→ Conv Block 3: [Conv(256, 3×3)]×3 → MaxPool
→ Conv Block 4: [Conv(512, 3×3)]×3 → MaxPool
→ Conv Block 5: [Conv(512, 3×3)]×3 → MaxPool
→ FC1 (4096) → FC2 (4096) → FC3 (1000)
```

**Key Features:**
- **Small filters:** Only 3×3 convolutions (more efficient than large filters)
- **Deep architecture:** Up to 19 layers
- **Uniform structure:** Simple, repetitive blocks
- **Proven design:** Still used as backbone today

**Key Insight:**
Two 3×3 convolutions have same receptive field as one 5×5, but with fewer parameters:
- 5×5: `25C²` parameters
- Two 3×3: `2 × 9C² = 18C²` parameters

**Parameters:** VGG-16: ~138M, VGG-19: ~144M

---

## 🚀 5. Modern CNN Architectures

### ResNet (2015) - Residual Networks

**Problem:** Deeper networks are harder to train (degradation problem)

**Solution:** Residual connections (skip connections)

#### Residual Block

**Mathematical Formulation:**
```
y = F(x, {W_i}) + x
```

Where:
- `F(x, {W_i})` = learned transformation (typically 2-3 conv layers)
- `x` = identity/skip connection
- `+` = element-wise addition

**Full Block:**
```
x → Conv → BN → ReLU → Conv → BN → [Add x] → ReLU → y
```

#### Architecture Variants

**ResNet-18:**
```
Input → Conv1 (7×7, 64, stride=2) → MaxPool
→ ResBlock1 (64) × 2
→ ResBlock2 (128) × 2
→ ResBlock3 (256) × 2
→ ResBlock4 (512) × 2
→ AvgPool → FC (1000)
```

**ResNet-34, ResNet-50, ResNet-101, ResNet-152:**
- Deeper variants with more blocks
- ResNet-50+ uses bottleneck blocks (1×1, 3×3, 1×1)

#### Bottleneck Block

For deeper ResNets (50+), uses bottleneck design:
```
x → Conv1×1 (reduce) → Conv3×3 → Conv1×1 (expand) → [Add x] → y
```

**Benefits:**
- Reduces parameters
- Faster computation
- Better gradient flow

#### Why ResNet Works

**1. Identity Mapping:**
- If optimal function is `H(x)`, ResNet learns `F(x) = H(x) - x`
- Easier to learn residual than full transformation

**2. Gradient Flow:**
```
∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)
```
- Gradient can flow directly through skip connection
- Prevents vanishing gradients in deep networks

**3. Ensembling Effect:**
- Multiple paths through network act like ensemble

**Parameters:** ResNet-18: ~11M, ResNet-50: ~25M, ResNet-152: ~60M

### Inception (GoogLeNet, 2014)

**Key Idea:** Use multiple filter sizes in parallel (multi-scale feature extraction)

#### Inception Module v1

**Structure:**
```
Input → [1×1 Conv] ──┐
      → [1×1 Conv → 3×3 Conv] ──┤
      → [1×1 Conv → 5×5 Conv] ──┼→ Concatenate → Output
      → [MaxPool → 1×1 Conv] ──┘
```

**Benefits:**
- **Multi-scale features:** Captures features at different scales
- **Efficient:** 1×1 convolutions reduce channels before expensive operations

#### Inception v2/v3 Improvements

**1. Factorized Convolutions:**
- Replace 5×5 with two 3×3
- Replace n×n with 1×n and n×1 (asymmetric)

**2. Batch Normalization:**
- Added after each convolution

**3. Auxiliary Classifiers:**
- Additional loss at intermediate layers
- Helps with gradient flow

#### Inception v4 / Inception-ResNet

**Combines:**
- Inception modules
- Residual connections
- Best of both architectures

**Parameters:** GoogLeNet: ~7M, Inception v3: ~24M

### DenseNet (2017) - Densely Connected Networks

**Key Idea:** Connect each layer to every other layer in a feedforward fashion

#### Dense Block

**Mathematical Formulation:**
```
xₗ = Hₗ([x₀, x₁, ..., xₗ₋₁])
```

Where:
- `[x₀, x₁, ..., xₗ₋₁]` = concatenation of all previous feature maps
- `Hₗ` = composite function (BN → ReLU → Conv)

**Structure:**
```
x₀ → H₁ → x₁
[x₀, x₁] → H₂ → x₂
[x₀, x₁, x₂] → H₃ → x₃
...
```

#### Architecture

**DenseNet-121:**
```
Input → Conv → DenseBlock1 → Transition → DenseBlock2 → Transition → DenseBlock3 → Transition → DenseBlock4 → Classifier
```

**Transition Layer:**
- 1×1 Conv (channel reduction)
- 2×2 AvgPool (spatial reduction)

#### Benefits

**1. Feature Reuse:**
- All previous features available to each layer
- Encourages feature reuse

**2. Parameter Efficiency:**
- Fewer parameters than ResNet
- Each layer adds only `k` new feature maps (growth rate)

**3. Gradient Flow:**
- Direct paths from loss to all layers
- Excellent gradient flow

**Parameters:** DenseNet-121: ~8M, DenseNet-169: ~14M

### MobileNet (2017) - Efficient Mobile Architectures

**Goal:** Efficient CNNs for mobile/edge devices

#### Depthwise Separable Convolution

**Standard Convolution:**
- Input: `[H × W × C_in]`
- Output: `[H × W × C_out]`
- Parameters: `k × k × C_in × C_out`

**Depthwise Separable:**
1. **Depthwise Convolution:**
   - `C_in` separate filters, one per channel
   - Parameters: `k × k × C_in`

2. **Pointwise Convolution (1×1):**
   - Standard 1×1 conv to combine channels
   - Parameters: `1 × 1 × C_in × C_out`

**Total Parameters:**
```
k × k × C_in + C_in × C_out = C_in × (k² + C_out)
```

**Reduction Factor:**
```
(k² + C_out) / (k² × C_out) ≈ 1 / k²  (when C_out >> k²)
```

For `k = 3`: ~8-9× fewer parameters

#### MobileNet v1 Architecture

```
Input → Conv (3×3, stride=2) → BN → ReLU
→ [Depthwise Separable] × 13 blocks
→ AvgPool → FC → Output
```

#### MobileNet v2 Improvements

**1. Inverted Residuals:**
- Expand → Depthwise → Project
- Uses residual connections

**2. Linear Bottlenecks:**
- ReLU removed from bottleneck (preserves information)

**Structure:**
```
x → 1×1 Conv (expand) → ReLU6
  → 3×3 Depthwise → ReLU6
  → 1×1 Conv (project) → [Add x] → y
```

**Parameters:** MobileNet v1: ~4.2M, MobileNet v2: ~3.4M

### EfficientNet (2019)

**Key Insight:** Balance depth, width, and resolution systematically

#### Compound Scaling

**Scaling Dimensions:**
- **Depth (d):** Number of layers
- **Width (w):** Number of channels
- **Resolution (r):** Input image size

**Scaling Formula:**
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

s.t. α · β² · γ² ≈ 2
```

Where `φ` is the compound coefficient.

#### EfficientNet-B0 Base Architecture

Uses MobileNet v2 inverted residual blocks with:
- MBConv blocks (mobile inverted bottleneck)
- Squeeze-and-Excitation (SE) attention
- Swish activation

**Variants:** EfficientNet-B0 to B7 (scaled versions)

**Parameters:** B0: ~5.3M, B7: ~66M

---

## 🎯 6. Advanced Techniques

### Attention Mechanisms

#### Squeeze-and-Excitation (SE) Block

**Operation:**
```
1. Global Average Pooling: z = (1/(H×W)) · Σᵢ Σⱼ x[i,j]
2. Excitation: s = σ(W₂ · ReLU(W₁ · z))
3. Scale: y = s ⊙ x
```

**Properties:**
- **Channel attention:** Learns channel importance
- **Lightweight:** Minimal parameters
- **Effective:** Significant performance gains

#### Spatial Attention

**Operation:**
```
1. Channel-wise statistics: s = [mean(x), max(x)]
2. Conv: a = σ(Conv(s))
3. Scale: y = a ⊙ x
```

**Properties:**
- **Spatial attention:** Learns spatial importance
- **Complementary:** Works well with channel attention

### Transfer Learning

**Strategies:**

**1. Feature Extraction:**
- Freeze all layers
- Train only classifier

**2. Fine-tuning:**
- Unfreeze some/all layers
- Train with lower learning rate

**3. Progressive Unfreezing:**
- Gradually unfreeze layers from top to bottom

**Benefits:**
- **Faster training:** Leverage pre-trained weights
- **Less data:** Works with smaller datasets
- **Better performance:** Pre-trained models are well-initialized

### Data Augmentation

**Common Techniques:**

**1. Geometric:**
- Random crop
- Random flip (horizontal/vertical)
- Rotation
- Translation
- Scaling

**2. Color:**
- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Color jittering

**3. Advanced:**
- Cutout (random erasing)
- Mixup: `x = λx₁ + (1-λ)x₂`, `y = λy₁ + (1-λ)y₂`
- CutMix: Combine parts of different images

**Benefits:**
- **Regularization:** Reduces overfitting
- **Robustness:** Improves generalization
- **Data efficiency:** Effectively increases dataset size

---

## 📊 7. Backpropagation in CNNs

### Forward Pass

**Convolution:**
```
Y[i, j, k] = Σₘ Σₙ Σ_c X[i·S + m, j·S + n, c] · W[m, n, c, k] + b[k]
```

### Backward Pass

**Gradient w.r.t. Output:**
```
∂L/∂Y[i, j, k] = δ[i, j, k]
```

**Gradient w.r.t. Weights:**
```
∂L/∂W[m, n, c, k] = Σᵢ Σⱼ X[i·S + m, j·S + n, c] · δ[i, j, k]
```

**Gradient w.r.t. Input:**
```
∂L/∂X[i, j, c] = Σₘ Σₙ Σₖ W[m, n, c, k] · δ[(i-m)/S, (j-n)/S, k]
```

(With appropriate padding and stride handling)

**Gradient w.r.t. Bias:**
```
∂L/∂b[k] = Σᵢ Σⱼ δ[i, j, k]
```

### Max Pooling Backward

**Operation:**
```
∂L/∂X[i, j] = {
    ∂L/∂Y[i', j'],  if (i, j) was max in pool
    0,              otherwise
}
```

Where `(i', j')` is the output position corresponding to the pool containing `(i, j)`.

---

## 🔧 8. Training Techniques

### Learning Rate Scheduling

**1. Step Decay:**
```
lr(t) = lr₀ · γ^⌊t/s⌋
```

**2. Exponential Decay:**
```
lr(t) = lr₀ · γ^t
```

**3. Cosine Annealing:**
```
lr(t) = lr_min + (lr_max - lr_min) · (1 + cos(πt/T)) / 2
```

**4. Warm Restarts:**
- Periodically reset learning rate
- Helps escape local minima

### Regularization

**1. Dropout:**
- Randomly set activations to zero during training
- Rate: typically 0.5 for FC layers, 0.1-0.2 for conv layers

**2. Weight Decay (L2):**
```
L_total = L + λ · ||W||²
```

**3. Early Stopping:**
- Stop when validation loss stops improving

### Optimization Algorithms

**1. SGD with Momentum:**
```
v_t = μ · v_{t-1} + α · ∇L
θ_t = θ_{t-1} - v_t
```

**2. Adam:**
```
m_t = β₁ · m_{t-1} + (1 - β₁) · ∇L
v_t = β₂ · v_{t-1} + (1 - β₂) · (∇L)²
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**3. AdamW:**
- Decoupled weight decay
- Better generalization than Adam

---

## 📈 9. Performance Comparison

| Architecture | Parameters | Top-1 Error (ImageNet) | Year |
|--------------|------------|------------------------|------|
| AlexNet | 60M | 15.3% | 2012 |
| VGG-16 | 138M | 7.3% | 2014 |
| ResNet-50 | 25M | 5.3% | 2015 |
| ResNet-152 | 60M | 4.5% | 2015 |
| DenseNet-121 | 8M | 5.0% | 2017 |
| MobileNet v2 | 3.4M | 6.9% | 2018 |
| EfficientNet-B0 | 5.3M | 5.1% | 2019 |
| EfficientNet-B7 | 66M | 2.4% | 2019 |

**Key Trends:**
- **Efficiency:** Better accuracy with fewer parameters
- **Depth:** Deeper networks with residual connections
- **Mobile:** Specialized architectures for edge devices
- **Automation:** Neural architecture search (NAS)

---

## 🎓 10. Key Takeaways

1. **Convolution exploits spatial locality** and parameter sharing for efficiency
2. **Pooling provides translation invariance** and dimensionality reduction
3. **Residual connections** enable training of very deep networks
4. **Multi-scale features** (Inception) capture patterns at different scales
5. **Depthwise separable convolution** dramatically reduces parameters
6. **Transfer learning** is essential for practical applications
7. **Data augmentation** significantly improves generalization
8. **Modern architectures** balance accuracy, efficiency, and speed

---

## 📚 11. Applications

### Computer Vision Tasks

**1. Image Classification:**
- Object recognition
- Scene understanding

**2. Object Detection:**
- R-CNN, Fast R-CNN, Faster R-CNN
- YOLO, SSD
- RetinaNet

**3. Semantic Segmentation:**
- FCN (Fully Convolutional Networks)
- U-Net
- DeepLab

**4. Instance Segmentation:**
- Mask R-CNN
- YOLACT

**5. Other Applications:**
- Face recognition
- Medical imaging
- Autonomous vehicles
- Style transfer
- Super-resolution

---

## 🔬 12. Design Principles

### Receptive Field

**Definition:** The region in input space that affects a particular output.

**Calculation:**
```
RF_l = RF_{l-1} + (k_l - 1) · S_{l-1}
```

Where:
- `RF_l` = receptive field at layer `l`
- `k_l` = kernel size at layer `l`
- `S_{l-1}` = cumulative stride up to layer `l-1`

**Goal:** Ensure receptive field covers entire object of interest.

### Feature Hierarchy

**Low-level features (early layers):**
- Edges, corners, textures
- Local patterns

**Mid-level features (middle layers):**
- Parts, shapes
- Combinations of low-level features

**High-level features (late layers):**
- Objects, scenes
- Semantic concepts

### Architecture Design Guidelines

**1. Start simple:** Begin with proven architectures (ResNet, VGG)

**2. Progressive complexity:** Add complexity only if needed

**3. Consider constraints:**
- Computational budget
- Memory limitations
- Latency requirements

**4. Use proven components:**
- Batch normalization
- Residual connections
- Attention mechanisms

---

## 📚 References & Further Reading

- **LeNet:** LeCun et al. (1998) - "Gradient-based learning applied to document recognition"
- **AlexNet:** Krizhevsky et al. (2012) - "ImageNet Classification with Deep Convolutional Neural Networks"
- **VGG:** Simonyan & Zisserman (2014) - "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- **ResNet:** He et al. (2015) - "Deep Residual Learning for Image Recognition"
- **Inception:** Szegedy et al. (2014) - "Going deeper with convolutions"
- **DenseNet:** Huang et al. (2017) - "Densely Connected Convolutional Networks"
- **MobileNet:** Howard et al. (2017) - "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
- **EfficientNet:** Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

---

*This document covers fundamental CNN architectures and techniques. For attention-based vision models (Vision Transformers), see separate documentation.*

