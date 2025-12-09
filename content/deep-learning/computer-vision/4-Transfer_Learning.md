+++
title = "Transfer Learning in Computer Vision"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Comprehensive guide to transfer learning in computer vision. Covers strategies (feature extraction, fine-tuning, progressive unfreezing), pretrained models, domain adaptation, and best practices for leveraging pretrained models."
+++

# 🔄 Transfer Learning in Computer Vision

Transfer learning is a technique where knowledge gained from training a model on one task is applied to a different but related task. In computer vision, this typically involves using models pretrained on large datasets (like ImageNet) and adapting them for your specific task.

**Key Benefits:**
- **Faster Training**: Leverage pretrained weights
- **Less Data**: Works well with smaller datasets
- **Better Performance**: Pretrained models are well-initialized
- **Reduced Computational Cost**: Less training time and resources

---

## 1. Why Transfer Learning Works

### Learned Features Hierarchy

CNNs learn hierarchical features:
- **Early Layers**: Low-level features (edges, textures, colors)
- **Middle Layers**: Mid-level features (shapes, patterns)
- **Late Layers**: High-level features (object parts, complex patterns)

**Key Insight:** Low and mid-level features are often **transferable** across tasks, while high-level features are task-specific.

### Example: ImageNet → Medical Imaging

- **Early layers**: Edge detection works for both natural images and medical images
- **Late layers**: Object classification (ImageNet) vs. disease detection (medical) - different

**Solution:** Reuse early layers, retrain late layers

---

## 2. Transfer Learning Strategies

### Strategy 1: Feature Extraction

**Process:**
1. Load pretrained model
2. **Freeze all layers** (set `requires_grad = False`)
3. Remove final classification layer
4. Add new classification layer for your task
5. Train only the new layer

**When to Use:**
- Small dataset (< 1000 images)
- Similar domain to pretraining
- Limited computational resources

**Advantages:**
- Fast training
- Low risk of overfitting
- Minimal computational cost

**Limitations:**
- May not achieve best performance
- Cannot adapt to domain differences

### Strategy 2: Fine-Tuning

**Process:**
1. Load pretrained model
2. **Unfreeze all layers** (or some layers)
3. Replace final classification layer
4. Train with **lower learning rate** (e.g., 1/10 of original)
5. Optionally freeze early layers

**When to Use:**
- Larger dataset (> 1000 images)
- Some domain differences
- Need best performance

**Advantages:**
- Better performance than feature extraction
- Can adapt to domain differences
- Flexible (can freeze/unfreeze layers)

**Limitations:**
- Longer training time
- Risk of overfitting with small datasets
- Requires more computational resources

### Strategy 3: Progressive Unfreezing

**Process:**
1. Start with feature extraction (freeze all)
2. Train for a few epochs
3. Unfreeze last layer, train
4. Unfreeze second-to-last layer, train
5. Continue until all layers are unfrozen

**When to Use:**
- Medium-sized dataset
- Domain differences
- Want to balance performance and training time

**Advantages:**
- Gradual adaptation
- Better than feature extraction
- More stable than full fine-tuning

**Limitations:**
- More complex training procedure
- Requires careful learning rate scheduling

---

## 3. Pretrained Models

### ImageNet Pretrained Models

**Popular Architectures:**

**1. ResNet:**
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- **Use Case**: General purpose, good balance
- **Characteristics**: Residual connections, deep networks

**2. VGG:**
- VGG-16, VGG-19
- **Use Case**: Simple architecture, good for understanding
- **Characteristics**: Simple, many parameters

**3. EfficientNet:**
- EfficientNet-B0 to B7
- **Use Case**: Efficient, good accuracy
- **Characteristics**: Compound scaling, efficient

**4. MobileNet:**
- MobileNet v1, v2, v3
- **Use Case**: Mobile/edge devices
- **Characteristics**: Lightweight, fast inference

**5. DenseNet:**
- DenseNet-121, DenseNet-169, DenseNet-201
- **Use Case**: Feature reuse
- **Characteristics**: Dense connections

**6. Vision Transformer (ViT):**
- ViT-Base, ViT-Large
- **Use Case**: Large-scale pretraining
- **Characteristics**: Transformer architecture

### Model Selection Guide

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| **ResNet-50** | 25M | Medium | High | General purpose |
| **EfficientNet-B0** | 5M | Fast | High | Efficient inference |
| **MobileNet v3** | 5M | Very Fast | Medium-High | Mobile devices |
| **ResNet-101** | 44M | Slow | Very High | High accuracy needed |
| **ViT-Base** | 86M | Medium | Very High | Large-scale tasks |

---

## 4. Implementation Examples

### Feature Extraction (PyTorch)

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 10  # Your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only train the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### Fine-Tuning (PyTorch)

```python
# Load pretrained model
model = models.resnet50(pretrained=True)

# Replace final layer
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Use lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 1/10 of typical
```

### Progressive Unfreezing (PyTorch)

```python
# Stage 1: Feature extraction
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Train for a few epochs
# ...

# Stage 2: Unfreeze last layer
for param in model.layer4.parameters():
    param.requires_grad = True

# Train for a few epochs
# ...

# Stage 3: Unfreeze more layers
for param in model.layer3.parameters():
    param.requires_grad = True

# Continue...
```

---

## 5. Domain Adaptation

### Problem

**Source Domain:** ImageNet (natural images)
**Target Domain:** Your task (e.g., medical images, satellite images)

**Challenge:** Domain shift - features learned on source may not transfer well

### Solutions

**1. Fine-Tuning:**
- Adapt pretrained model to target domain
- Use lower learning rate
- More data helps

**2. Domain-Adversarial Training:**
- Train domain discriminator
- Encourage domain-invariant features
- More complex but effective

**3. Domain-Specific Pretraining:**
- Pretrain on domain-specific data
- Then fine-tune on your task
- Best performance but requires more data

---

## 6. Learning Rate Strategies

### Differential Learning Rates

**Different learning rates for different layers:**

```python
# Early layers: very small LR (frozen or barely updated)
# Middle layers: small LR
# Final layers: larger LR

optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-3},
], lr=1e-3)
```

### Learning Rate Scheduling

**1. ReduceLROnPlateau:**
- Reduce LR when validation loss plateaus
- Good for fine-tuning

**2. Cosine Annealing:**
- Gradually reduce LR following cosine curve
- Smooth convergence

**3. Warm Restarts:**
- Periodically increase LR
- Helps escape local minima

---

## 7. Data Considerations

### Dataset Size Guidelines

| Dataset Size | Strategy | Learning Rate |
|--------------|----------|---------------|
| **< 1000 images** | Feature Extraction | Standard (0.001) |
| **1000-10000 images** | Fine-Tuning (freeze early layers) | Low (0.0001) |
| **> 10000 images** | Full Fine-Tuning | Low (0.0001) |

### Data Augmentation

**Important for transfer learning:**
- Helps prevent overfitting
- Especially important with small datasets
- Use standard augmentations: random crop, flip, color jitter

---

## 8. Best Practices

### 1. Start with Feature Extraction

**Why:**
- Fast to implement
- Low risk
- Good baseline

**Then:**
- If performance is good, stop
- If not, try fine-tuning

### 2. Use Appropriate Learning Rates

**Guidelines:**
- Feature extraction: Standard LR (0.001)
- Fine-tuning: Lower LR (0.0001 or 1/10 of standard)
- Differential LR: Smaller for early layers

### 3. Monitor Training Carefully

**Watch for:**
- Overfitting (especially with small datasets)
- Validation loss plateau
- Learning rate too high/low

### 4. Choose Right Model

**Consider:**
- Dataset size
- Computational resources
- Accuracy requirements
- Inference speed needs

### 5. Experiment with Strategies

**Try:**
- Feature extraction first
- Then fine-tuning
- Compare results
- Choose best approach

---

## 9. Common Pitfalls

### 1. Learning Rate Too High

**Problem:** Destroys pretrained weights
**Solution:** Use lower learning rate (0.0001 or less)

### 2. Overfitting

**Problem:** Model memorizes small dataset
**Solution:** 
- Use data augmentation
- Use feature extraction instead of fine-tuning
- Add regularization (dropout, weight decay)

### 3. Wrong Model Choice

**Problem:** Model too large/small for task
**Solution:** 
- Start with ResNet-50 (good balance)
- Scale up/down based on results

### 4. Not Freezing Layers Correctly

**Problem:** Accidentally training frozen layers or vice versa
**Solution:** 
- Check `requires_grad` flags
- Verify which layers are being updated

---

## 10. Applications

**1. Medical Imaging:**
- Transfer from ImageNet to medical images
- Feature extraction or fine-tuning
- Works well with limited medical data

**2. Satellite Imagery:**
- Transfer to remote sensing tasks
- Domain adaptation important
- Fine-tuning typically needed

**3. Art and Style:**
- Transfer to artistic images
- May need more fine-tuning
- Domain shift is significant

**4. Industrial Inspection:**
- Transfer to manufacturing images
- Feature extraction often sufficient
- Small datasets common

---

## 11. Key Takeaways

1. **Transfer learning** leverages pretrained models for new tasks
2. **Feature extraction** is best for small datasets and similar domains
3. **Fine-tuning** is best for larger datasets and domain differences
4. **Progressive unfreezing** balances performance and training time
5. **Lower learning rates** are crucial for fine-tuning
6. **Early layers** learn general features, **late layers** learn task-specific features
7. **Model selection** depends on dataset size, resources, and requirements
8. **Data augmentation** is especially important for transfer learning

---

## References

- "How transferable are features in deep neural networks?" (Yosinski et al., 2014)
- "Deep Residual Learning for Image Recognition" (ResNet)
- "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
- PyTorch Transfer Learning Tutorial
- TensorFlow Transfer Learning Guide

