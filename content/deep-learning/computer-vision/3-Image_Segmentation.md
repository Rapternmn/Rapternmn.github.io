+++
title = "Image Segmentation"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Comprehensive guide to image segmentation in computer vision. Covers semantic segmentation (FCN, U-Net, DeepLab), instance segmentation (Mask R-CNN), and panoptic segmentation with architectures, algorithms, and implementations."
+++

# 🎨 Image Segmentation

Image segmentation is the task of partitioning an image into multiple segments or regions, where each pixel is assigned to a specific class or object instance. Unlike object detection (which uses bounding boxes), segmentation provides pixel-level precision.

**Types of Segmentation:**
1. **Semantic Segmentation**: Classify each pixel into a category (e.g., person, car, road)
2. **Instance Segmentation**: Identify and segment individual object instances
3. **Panoptic Segmentation**: Combine semantic and instance segmentation

---

## 1. Problem Formulation

### Semantic Segmentation

**Task:** Assign a class label to every pixel in the image.

**Input:** Image `I` of size `H × W × 3`
**Output:** Segmentation mask `M` of size `H × W` where each pixel has a class label

**Example:**
- Input: Street scene image
- Output: Each pixel labeled as "road", "car", "person", "building", etc.

### Instance Segmentation

**Task:** Identify and segment each individual object instance.

**Output:**
- Class label for each pixel
- Instance ID to distinguish different objects of the same class

**Example:**
- Input: Image with multiple people
- Output: Each person segmented separately with unique instance IDs

### Panoptic Segmentation

**Task:** Combine semantic and instance segmentation.

**Output:**
- "Stuff" classes (sky, road, grass) - semantic segmentation
- "Things" classes (person, car, dog) - instance segmentation

---

## 2. Evaluation Metrics

### Intersection over Union (IoU)

**Per-Class IoU:**
```
IoU_class = (Area of Intersection) / (Area of Union)
```

**Mean IoU (mIoU):**
```
mIoU = (1/C) × Σ IoU_class
```

Where `C` is the number of classes.

### Pixel Accuracy

**Overall Pixel Accuracy:**
```
PA = (Correct Pixels) / (Total Pixels)
```

**Mean Pixel Accuracy:**
```
mPA = (1/C) × Σ (Correct Pixels in Class / Total Pixels in Class)
```

### Panoptic Quality (PQ)

For panoptic segmentation:
```
PQ = (Σ IoU) / (Number of Segments)
```

---

## 3. Semantic Segmentation

### Fully Convolutional Networks (FCN)

**Key Innovation:** Replace fully connected layers with convolutional layers to enable dense prediction.

**Architecture:**
```
Input Image → CNN (VGG/ResNet) → Feature Map
                                    ↓
                            Upsampling (Transposed Convolution)
                                    ↓
                            Segmentation Map
```

**Upsampling Methods:**
1. **Transposed Convolution (Deconvolution):**
   - Learnable upsampling
   - Parameters: kernel size, stride, padding

2. **Bilinear Interpolation:**
   - Fixed upsampling
   - No learnable parameters

**Skip Connections:**
- Combine high-level (semantic) and low-level (spatial) features
- Improves boundary accuracy

**FCN Variants:**
- **FCN-32s**: Single upsampling (32×)
- **FCN-16s**: Two-stage upsampling with skip connection
- **FCN-8s**: Three-stage upsampling with multiple skip connections

### U-Net

**Architecture:** Encoder-decoder with skip connections

```
Input → Encoder (Downsampling) → Bottleneck → Decoder (Upsampling) → Output
         ↓                                    ↑
    Skip Connections (concatenate)
```

**Key Features:**
1. **Symmetric Architecture**: Encoder and decoder are symmetric
2. **Skip Connections**: Connect encoder and decoder at same resolution
3. **Upsampling**: Transposed convolutions
4. **Concatenation**: Concatenate (not add) skip connections

**Encoder:**
- Repeated: Conv → ReLU → Conv → ReLU → MaxPool
- Downsampling: 2× at each stage

**Decoder:**
- Repeated: Upsample → Conv → ReLU → Conv → ReLU
- Upsampling: 2× at each stage
- Concatenate with corresponding encoder feature map

**Loss Function:**
- Pixel-wise cross-entropy
- Dice loss (for imbalanced classes)
- Combined loss

**Advantages:**
- Excellent for medical imaging
- Works well with small datasets
- Precise boundary detection

### DeepLab

**Key Innovations:**
1. **Atrous Convolution (Dilated Convolution)**
2. **Atrous Spatial Pyramid Pooling (ASPP)**
3. **Conditional Random Fields (CRF)**

**Atrous Convolution:**
- Increases receptive field without downsampling
- Preserves spatial resolution
- Formula: `k_eff = k + (k - 1) × (r - 1)` where `r` is dilation rate

**ASPP:**
- Multiple parallel atrous convolutions with different rates
- Captures multi-scale context
- Concatenate outputs

**CRF:**
- Post-processing to refine boundaries
- Encourages smoothness and consistency

**DeepLab Variants:**
- **DeepLab v1**: Atrous convolution + CRF
- **DeepLab v2**: ASPP + CRF
- **DeepLab v3**: Improved ASPP, no CRF
- **DeepLab v3+**: Encoder-decoder with ASPP

### PSPNet (Pyramid Scene Parsing Network)

**Key Innovation:** Pyramid Pooling Module

**Architecture:**
```
Input → CNN → Feature Map
              ↓
         Pyramid Pooling Module
         (Multiple pooling scales)
              ↓
         Concatenate → Conv → Output
```

**Pyramid Pooling:**
- Pool at multiple scales: 1×1, 2×2, 3×3, 6×6
- Upsample to original size
- Concatenate with original features

**Benefits:**
- Captures global context
- Multi-scale feature representation

---

## 4. Instance Segmentation

### Mask R-CNN

**Architecture:** Extends Faster R-CNN with mask prediction branch

```
Image → CNN → Feature Map
              ↓
         ┌─────────────────┐
         │  RPN            │ → Region Proposals
         └─────────────────┘
              ↓
         ROI Align → Three Branches:
                     1. Classification
                     2. Bounding Box Regression
                     3. Mask Prediction (Binary Segmentation)
```

**Key Differences from Faster R-CNN:**
1. **ROI Align** (instead of ROI Pooling):
   - Bilinear interpolation for precise alignment
   - No quantization errors

2. **Mask Branch:**
   - Predicts binary mask for each class
   - Small FCN (14×14 → 28×28)
   - Class-agnostic or class-specific masks

**Training:**
- Multi-task loss: `L = L_cls + L_box + L_mask`
- Mask loss: Binary cross-entropy per pixel

**Advantages:**
- High accuracy
- Unified framework (detection + segmentation)
- Good for complex scenes

### YOLACT (You Only Look At Coefficients)

**Key Innovation:** Real-time instance segmentation

**Architecture:**
```
Image → Backbone → Feature Pyramid
                   ↓
              Two Branches:
              1. Protonet (Mask Coefficients)
              2. Prediction Head (BBox + Coefficients)
                   ↓
              Combine → Instance Masks
```

**Process:**
1. Generate prototype masks
2. Predict coefficients for each instance
3. Combine: `Mask = Σ(coefficient × prototype)`

**Advantages:**
- Real-time inference
- Simple architecture
- Good speed-accuracy tradeoff

### SOLO (Segmenting Objects by Locations)

**Key Idea:** Predict instance masks directly from locations

**Architecture:**
- Divide image into grid
- Each grid cell predicts:
  - Category
  - Instance mask
  - Objectness

**Advantages:**
- Simple and efficient
- No bounding box needed
- Good accuracy

---

## 5. Panoptic Segmentation

### Unified Framework

**Task:** Combine semantic and instance segmentation

**Approach:**
1. **Semantic Segmentation Branch**: Predict "stuff" classes
2. **Instance Segmentation Branch**: Predict "things" classes
3. **Merge**: Combine predictions with conflict resolution

**Conflict Resolution:**
- If pixel belongs to both, prioritize instance segmentation
- Use confidence scores

### Panoptic FPN

**Architecture:**
- Extends Feature Pyramid Network
- Two heads:
  - Semantic segmentation head
  - Instance segmentation head

---

## 6. Key Techniques

### Atrous (Dilated) Convolution

**Purpose:** Increase receptive field without downsampling

**Formula:**
```
Output[i] = Σ Input[i + r × k] × Kernel[k]
```

Where `r` is the dilation rate.

**Benefits:**
- Larger receptive field
- Preserves spatial resolution
- Multi-scale context

### Skip Connections

**Types:**
1. **Addition**: `output = input + feature`
2. **Concatenation**: `output = [input, feature]`

**Benefits:**
- Preserves spatial information
- Helps gradient flow
- Improves boundary accuracy

### Multi-Scale Features

**Techniques:**
1. **Feature Pyramid**: Multiple resolution features
2. **ASPP**: Multiple atrous rates
3. **Pyramid Pooling**: Multiple pooling scales

**Benefits:**
- Handles objects at different scales
- Captures global context

### Loss Functions

**1. Cross-Entropy Loss:**
```
L = -Σ log(p_true_class)
```

**2. Dice Loss:**
```
Dice = (2 × |A ∩ B|) / (|A| + |B|)
L_dice = 1 - Dice
```

**3. Focal Loss:**
```
L = -α(1 - p)^γ log(p)
```

**4. Combined Loss:**
```
L = L_CE + λ × L_dice
```

---

## 7. Implementation Considerations

### Data Augmentation

**Common Techniques:**
- Random crop
- Random flip
- Color jittering
- Random scale
- Elastic deformation (for medical images)

### Training Strategies

**1. Multi-Scale Training:**
- Train on different input sizes
- Improves robustness

**2. Class Balancing:**
- Weighted loss for rare classes
- Focal loss for imbalanced data

**3. Pretrained Backbones:**
- Use ImageNet pretrained models
- Faster convergence

### Post-Processing

**1. CRF (Conditional Random Fields):**
- Refine boundaries
- Encourage smoothness

**2. Morphological Operations:**
- Remove small holes
- Smooth boundaries

**3. Connected Components:**
- Filter small regions
- Merge similar regions

---

## 8. Applications

**1. Medical Imaging:**
- Organ segmentation
- Tumor detection
- Cell segmentation

**2. Autonomous Vehicles:**
- Road segmentation
- Lane detection
- Obstacle segmentation

**3. Satellite Imagery:**
- Land use classification
- Building detection
- Vegetation mapping

**4. Augmented Reality:**
- Object segmentation
- Background removal
- Virtual object placement

**5. Quality Control:**
- Defect detection
- Product inspection
- Surface analysis

---

## 9. Comparison of Methods

| Method | Type | Accuracy | Speed | Use Case |
|--------|------|----------|-------|----------|
| **FCN** | Semantic | Medium | Fast | General purpose |
| **U-Net** | Semantic | High | Medium | Medical imaging |
| **DeepLab v3+** | Semantic | Very High | Medium | High accuracy needed |
| **PSPNet** | Semantic | Very High | Medium | Scene parsing |
| **Mask R-CNN** | Instance | Very High | Medium | Complex scenes |
| **YOLACT** | Instance | High | Very Fast | Real-time applications |
| **SOLO** | Instance | High | Fast | Simple scenes |

---

## 10. Best Practices

**1. Choose Right Architecture:**
- **High accuracy**: DeepLab v3+, Mask R-CNN
- **Fast inference**: FCN, YOLACT
- **Medical imaging**: U-Net
- **Real-time**: YOLACT, SOLO

**2. Data Quality:**
- Precise pixel-level annotations
- Diverse dataset
- Proper class balance

**3. Training:**
- Use pretrained backbones
- Multi-scale training
- Proper loss function (Dice for imbalanced classes)

**4. Evaluation:**
- Use mIoU for semantic segmentation
- Use PQ for panoptic segmentation
- Visualize predictions

---

## 11. Key Takeaways

1. **Semantic segmentation** classifies each pixel into categories
2. **Instance segmentation** identifies individual object instances
3. **U-Net** is excellent for medical imaging with skip connections
4. **DeepLab** uses atrous convolution and ASPP for multi-scale context
5. **Mask R-CNN** extends Faster R-CNN for instance segmentation
6. **Atrous convolution** increases receptive field without downsampling
7. **Skip connections** preserve spatial information and improve boundaries
8. **Multi-scale features** handle objects at different scales

---

## References

- "Fully Convolutional Networks for Semantic Segmentation" (FCN)
- "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets" (DeepLab v1)
- "Rethinking Atrous Convolution for Semantic Image Segmentation" (DeepLab v3)
- "Encoder-Decoder with Atrous Separable Convolution" (DeepLab v3+)
- "Mask R-CNN" (He et al., 2017)
- "YOLACT: Real-time Instance Segmentation"
- "SOLO: Segmenting Objects by Locations"

