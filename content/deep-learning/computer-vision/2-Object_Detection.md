+++
title = "Object Detection"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Comprehensive guide to object detection in computer vision. Covers R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN), YOLO, SSD, RetinaNet, and modern approaches with architectures, algorithms, and implementations."
+++

# 🎯 Object Detection

Object detection is a computer vision task that involves identifying and localizing multiple objects within an image. Unlike image classification (which predicts a single label for the entire image), object detection predicts both **what** objects are present and **where** they are located (bounding boxes).

**Key Challenges:**
- Detecting multiple objects of different sizes
- Handling objects at various scales
- Efficient inference for real-time applications
- Balancing accuracy and speed

---

## 1. Problem Formulation

### Task Definition

Given an input image, object detection requires:
1. **Classification**: Identify object categories (e.g., person, car, dog)
2. **Localization**: Predict bounding boxes `(x, y, w, h)` for each object
3. **Confidence**: Assign confidence scores to detections

### Evaluation Metrics

**1. Intersection over Union (IoU):**
```
IoU = (Area of Overlap) / (Area of Union)
```

**2. Average Precision (AP):**
- Precision-Recall curve area
- AP@0.5: IoU threshold = 0.5
- AP@0.5:0.95: Average over IoU thresholds 0.5 to 0.95 (mAP)

**3. Mean Average Precision (mAP):**
- Average AP across all object classes

---

## 2. Two-Stage Detectors

Two-stage detectors first generate region proposals, then classify and refine them.

### R-CNN (Region-based CNN)

**Pipeline:**
1. **Region Proposal**: Generate ~2000 candidate regions using Selective Search
2. **Feature Extraction**: Extract features using CNN for each region
3. **Classification**: Classify each region using SVM
4. **Bounding Box Regression**: Refine bounding box coordinates

**Limitations:**
- Slow: Processes each region separately
- High memory: Stores features for all regions
- Not end-to-end trainable

### Fast R-CNN

**Improvements over R-CNN:**
1. **Shared Feature Extraction**: Extract features once for entire image
2. **ROI Pooling**: Extract fixed-size features from variable-size regions
3. **End-to-End Training**: Jointly train classification and bounding box regression

**ROI Pooling:**
```
Input: Feature map + ROI coordinates
Output: Fixed-size feature map (e.g., 7×7)
```

**Architecture:**
```
Image → CNN → Feature Map
              ↓
         ROI Pooling → FC Layers → Classification + BBox Regression
```

### Faster R-CNN

**Key Innovation: Region Proposal Network (RPN)**

**Architecture:**
```
Image → CNN → Feature Map
              ↓
         ┌─────────────────┐
         │  RPN            │ → Region Proposals
         └─────────────────┘
              ↓
         ROI Pooling → Classification + BBox Regression
```

**Region Proposal Network (RPN):**
- **Input**: Feature map from CNN
- **Output**: Object proposals with objectness scores
- **Anchors**: Predefined boxes at multiple scales and aspect ratios
- **Loss**: Binary classification (object vs. background) + BBox regression

**Anchors:**
- Multiple scales: `[8², 16², 32²]` pixels
- Multiple aspect ratios: `[0.5, 1.0, 2.0]`
- Total: 9 anchors per spatial location

**Training:**
1. Train RPN to generate proposals
2. Train Fast R-CNN using RPN proposals
3. Fine-tune RPN and Fast R-CNN jointly

**Advantages:**
- End-to-end trainable
- Faster than Fast R-CNN
- Better accuracy

**Limitations:**
- Still slower than one-stage detectors
- Complex architecture

---

## 3. One-Stage Detectors

One-stage detectors predict bounding boxes and classes directly from feature maps, without region proposals.

### YOLO (You Only Look Once)

**Core Idea:** Divide image into grid cells, each cell predicts bounding boxes and class probabilities.

**YOLO v1:**
- **Grid**: 7×7 grid cells
- **Predictions per cell**: 2 bounding boxes + 1 class probability
- **Output**: `7 × 7 × (5×2 + 20) = 7 × 7 × 30` tensor

**Architecture:**
```
Image (448×448) → CNN → 7×7×30 Tensor
```

**Loss Function:**
```
L = λ_coord Σ(bbox_coord_loss) + 
    λ_obj Σ(obj_confidence_loss) + 
    λ_noobj Σ(noobj_confidence_loss) + 
    Σ(class_probability_loss)
```

**Limitations:**
- Struggles with small objects
- Limited to 2 boxes per cell
- Fixed grid size

**YOLO v2 (YOLO9000):**
- **Anchor Boxes**: Use anchor boxes like Faster R-CNN
- **Batch Normalization**: Added to all layers
- **High-Resolution Classifier**: Train on higher resolution
- **Multi-Scale Training**: Train on different input sizes

**YOLO v3:**
- **Multi-Scale Predictions**: Predict at 3 different scales
- **Feature Pyramid Network**: Use FPN for better small object detection
- **Better Backbone**: Darknet-53

**YOLO v4/v5/v8:**
- Improved architectures
- Better data augmentation
- Optimized for speed and accuracy

### SSD (Single Shot Detector)

**Key Features:**
1. **Multi-Scale Feature Maps**: Predict from multiple CNN layers
2. **Default Boxes**: Similar to anchor boxes
3. **Hard Negative Mining**: Focus on difficult examples

**Architecture:**
```
Image → VGG/ResNet Backbone
         ↓
    Multiple Feature Maps (different scales)
         ↓
    Detection Heads → Predictions
```

**Default Boxes:**
- Multiple scales per feature map
- Multiple aspect ratios
- Total: ~8732 boxes per image

**Loss Function:**
```
L = L_conf + α × L_loc
```

**Advantages:**
- Faster than two-stage detectors
- Good accuracy
- Multi-scale detection

**Limitations:**
- Struggles with very small objects
- Hard negative mining required

### RetinaNet

**Key Innovation: Focal Loss**

**Problem:** Class imbalance (many background boxes, few object boxes)

**Focal Loss:**
```
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
```

Where:
- `p_t`: Predicted probability for true class
- `α_t`: Balancing factor
- `γ`: Focusing parameter (typically 2)

**Properties:**
- Down-weights easy examples
- Focuses on hard examples
- Handles class imbalance

**Architecture:**
- **Backbone**: ResNet + FPN
- **Two Subnetworks**:
  - Classification subnet
  - Bounding box regression subnet

**Advantages:**
- State-of-the-art accuracy
- Simpler than two-stage detectors
- Good speed-accuracy tradeoff

---

## 4. Modern Approaches

### DETR (Detection Transformer)

**Key Innovation:** Use Transformer architecture for object detection

**Architecture:**
```
Image → CNN Backbone → Feature Map
                        ↓
                   Transformer Encoder
                        ↓
                   Transformer Decoder → Object Queries
                        ↓
                   Predictions (class + bbox)
```

**Object Queries:**
- Learnable embeddings
- Each query predicts one object
- Fixed number of queries (e.g., 100)

**Advantages:**
- End-to-end trainable
- No anchor boxes needed
- Simpler architecture

**Limitations:**
- Slower training
- Requires more data

### YOLO v8 / YOLO-NAS

**Modern YOLO Variants:**
- Improved architectures
- Better training strategies
- Optimized for deployment

---

## 5. Key Techniques

### Non-Maximum Suppression (NMS)

**Problem:** Multiple detections for the same object

**Algorithm:**
1. Sort detections by confidence score
2. Select highest confidence detection
3. Remove all detections with IoU > threshold
4. Repeat until no detections remain

**IoU Threshold:** Typically 0.5

### Feature Pyramid Network (FPN)

**Problem:** Objects at different scales

**Solution:** Multi-scale feature maps

**Architecture:**
```
High-level features (semantic) ← Top-down pathway
Low-level features (spatial)   ← Bottom-up pathway
         ↓
    Lateral connections
         ↓
    Multi-scale feature maps
```

**Benefits:**
- Better small object detection
- Multi-scale representation

### Anchor Boxes

**Definition:** Predefined boxes with different scales and aspect ratios

**Design:**
- **Scales**: `[8², 16², 32², 64², 128²]` pixels
- **Aspect Ratios**: `[0.5, 1.0, 2.0]`
- **Total**: 5 scales × 3 ratios = 15 anchor boxes

**Purpose:**
- Handle objects of different sizes
- Improve detection accuracy

---

## 6. Implementation Considerations

### Data Augmentation

**Common Techniques:**
- Random crop
- Random flip
- Color jittering
- Mosaic augmentation (combine multiple images)

### Training Strategies

**1. Multi-Scale Training:**
- Train on different input sizes
- Improves robustness

**2. Hard Example Mining:**
- Focus on difficult examples
- Improves accuracy

**3. Data Balancing:**
- Balance positive/negative examples
- Handle class imbalance

### Inference Optimization

**1. Model Quantization:**
- Reduce precision (FP32 → INT8)
- Faster inference

**2. Model Pruning:**
- Remove unnecessary parameters
- Smaller model size

**3. TensorRT / ONNX:**
- Optimized inference engines
- Faster deployment

---

## 7. Applications

**1. Autonomous Vehicles:**
- Pedestrian detection
- Vehicle detection
- Traffic sign recognition

**2. Surveillance:**
- Person detection
- Object tracking
- Anomaly detection

**3. Retail:**
- Product detection
- Inventory management
- Customer behavior analysis

**4. Medical Imaging:**
- Lesion detection
- Organ localization
- Disease diagnosis

**5. Sports Analytics:**
- Player detection
- Ball tracking
- Action recognition

---

## 8. Comparison of Methods

| Method | Type | Speed | Accuracy | Complexity |
|--------|------|-------|----------|------------|
| **R-CNN** | Two-stage | Slow | High | High |
| **Fast R-CNN** | Two-stage | Medium | High | Medium |
| **Faster R-CNN** | Two-stage | Medium | Very High | High |
| **YOLO v3** | One-stage | Fast | Medium-High | Medium |
| **SSD** | One-stage | Fast | Medium-High | Medium |
| **RetinaNet** | One-stage | Medium | Very High | Medium |
| **DETR** | Transformer | Medium | Very High | High |

---

## 9. Best Practices

**1. Choose Right Architecture:**
- **High accuracy**: Faster R-CNN, RetinaNet
- **Fast inference**: YOLO, SSD
- **Balanced**: RetinaNet, YOLO v5/v8

**2. Data Quality:**
- High-quality annotations
- Diverse dataset
- Proper class balance

**3. Training:**
- Use pre-trained backbones
- Multi-scale training
- Proper learning rate scheduling

**4. Evaluation:**
- Use mAP for evaluation
- Visualize detections
- Analyze failure cases

---

## 10. Key Takeaways

1. **Two-stage detectors** (Faster R-CNN) offer higher accuracy but are slower
2. **One-stage detectors** (YOLO, SSD, RetinaNet) are faster with good accuracy
3. **Focal Loss** (RetinaNet) handles class imbalance effectively
4. **Feature Pyramid Networks** improve multi-scale detection
5. **Non-Maximum Suppression** removes duplicate detections
6. **Modern approaches** (DETR, YOLO v8) combine best of both worlds

---

## References

- "Rich feature hierarchies for accurate object detection" (R-CNN)
- "Fast R-CNN" (Girshick, 2015)
- "Faster R-CNN: Towards Real-Time Object Detection" (Ren et al., 2016)
- "You Only Look Once: Unified, Real-Time Object Detection" (YOLO v1)
- "SSD: Single Shot MultiBox Detector" (Liu et al., 2016)
- "Focal Loss for Dense Object Detection" (RetinaNet)
- "End-to-End Object Detection with Transformers" (DETR)

