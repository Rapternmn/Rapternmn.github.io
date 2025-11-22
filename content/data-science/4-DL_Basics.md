+++
title = "Deep Learning Basics: Interview Q&A"
date = 2025-11-22T10:00:00+05:30
draft = false
weight = 4
description = "Essential deep learning concepts and techniques for interview preparation. Covers normalization techniques, gradient problems, transfer learning, overfitting prevention, and weight initialization."
+++


## 1. Normalization Techniques

### What is the difference between batch normalization and layer normalization?

#### Batch Normalization

- **Definition**: Normalizes activations across the batch dimension (mean and variance computed across batch samples)
- **Use Case**: Mostly used in CNNs
- **Properties**:
  - Depends on batch size — unstable with very small batches
  - Requires batch statistics during training
  - Different behavior during training vs inference

**Formula**:
```
x̂ = (x - μ_B) / √(σ_B² + ε)
```

Where $\mu_B$ and $\sigma_B$ are batch mean and variance.

#### Layer Normalization

- **Definition**: Normalizes across features for each individual sample
- **Use Case**: Common in RNNs and Transformers (e.g., GPT, BERT)
- **Properties**:
  - Independent of batch size
  - Same computation during training and inference
  - More stable for variable batch sizes

**Summary**:
- **BatchNorm** = Normalize across batch dimension
- **LayerNorm** = Normalize across features per sample

---

## 2. Gradient Problems

### How do vanishing and exploding gradients occur, and how can they be mitigated?

#### Cause

In deep networks or RNNs, gradients are multiplied across many layers → exponential shrinking (vanishing) or growth (exploding).

**Vanishing Gradients**: Gradients become too small to update weights effectively
**Exploding Gradients**: Gradients become too large, causing unstable training

#### Mitigation Strategies

**For Vanishing Gradients:**
1. **Use ReLU instead of sigmoid/tanh**: ReLU has gradient of 1 for positive inputs
2. **Use residual connections (ResNets)**: Skip connections allow gradient flow
3. **Batch Normalization**: Helps stabilize gradients
4. **Proper weight initialization**: Xavier/He initialization
5. **LSTM/GRU**: Designed to handle long-term dependencies

**For Exploding Gradients:**
1. **Gradient clipping**: Cap gradients at maximum value
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
2. **Proper weight initialization**: He/Xavier initialization
3. **Batch Normalization**: Stabilizes training
4. **Smaller learning rate**: Reduces update magnitude

---

## 3. Transfer Learning

### How does transfer learning work in deep learning?

#### Transfer Learning Steps

1. **Load a pretrained model** (e.g., ImageNet-trained ResNet, BERT)
2. **Choose strategy**:
   - **Freeze base layers, retrain last layers** on new task
   - **Fine-tune all layers** on new dataset
   - **Progressive unfreezing**: Gradually unfreeze layers

#### Why Use Transfer Learning?

- **Leverages prior learning**: Model already learned useful features
- **Helps with limited data**: Works well with small datasets
- **Faster training**: Less training time needed
- **Better performance**: Often outperforms training from scratch

#### Common Applications

- **NLP**: BERT, GPT, T5 for text tasks
- **Computer Vision**: ResNet, EfficientNet, VGG for image tasks
- **Domain Adaptation**: Transfer from source to target domain

---

## 4. Overfitting Prevention

### How do you handle overfitting in deep neural networks?

#### Techniques

1. **Regularization**
   - **L1/L2 weight decay**: Penalize large weights
   - **Dropout**: Randomly drop neurons during training
   - **Early stopping**: Stop when validation loss increases

2. **Dropout**
   - Randomly set neurons to zero during training
   - **Typical rate**: 0.2-0.5
   - **Effect**: Prevents co-adaptation of neurons

3. **Early Stopping**
   - Monitor validation loss
   - Stop training when validation loss stops improving
   - **Patience**: Number of epochs to wait before stopping

4. **Data Augmentation**
   - Create more diverse training samples
   - **Image**: Rotation, flipping, cropping, color jittering
   - **Text**: Synonym replacement, back-translation
   - **Effect**: Increases effective dataset size

5. **Simpler Architecture**
   - Reduce parameters or layers
   - **Regularization effect**: Less capacity to overfit

6. **Cross-Validation**
   - For small datasets
   - Better estimate of generalization error

7. **Batch Normalization**
   - Acts as regularization
   - Reduces internal covariate shift

---

## 5. Weight Initialization

### What is Xavier Initialization (aka Glorot Initialization)?

**Definition**: Method for initializing weights so that variance of activations and gradients remains stable across layers.

#### The Problem It Solves

If weights are initialized too:
- **Large** → Activations and gradients explode
- **Small** → Activations and gradients vanish

This causes training to:
- Become unstable, or
- Not learn at all

#### Xavier Initialization

**For Tanh/Sigmoid**:
```
W ~ N(0, 1/n_in)
```

**For ReLU (He Initialization)**:
```
W ~ N(0, 2/n_in)
```

Where $n_{in}$ = number of input neurons.

**Goal**: Keep variance of activations and gradients constant across layers.

---

## 6. Terminology

### Loss vs. Cost

- **Loss**: Error for one data point
- **Cost**: Average loss over the entire training batch or dataset

**Example**:
- Loss: $(y_i - \hat{y}_i)^2$ for sample i
- Cost: $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ for entire dataset

---

### Transfer Learning vs Fine-Tuning

- **Transfer Learning**: Using knowledge from a pretrained model (broader term)
- **Fine-Tuning**: Updating weights of the pretrained model on a new task (specific technique)

**Relationship**: Fine-tuning is a type of transfer learning.

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| **BatchNorm** | Normalize across batch dimension |
| **LayerNorm** | Normalize across features per sample |
| **Vanishing Gradients** | Use ReLU, ResNets, proper initialization |
| **Exploding Gradients** | Gradient clipping, proper initialization |
| **Overfitting** | Dropout, regularization, early stopping, data augmentation |
| **Transfer Learning** | Leverage pretrained models for new tasks |
| **Xavier Init** | Initialize weights to maintain variance |

---

*Last Updated: 2024*
