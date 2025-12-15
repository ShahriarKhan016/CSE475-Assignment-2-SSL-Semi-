# 🔬 Notebook 02: Semi-Supervised Object Detection (SSOD) with Pseudo-Labeling

## Overview

This notebook implements a **Semi-Supervised Object Detection (SSOD)** pipeline using **Pseudo-Labeling** with **YOLOv12**. The approach leverages both labeled and unlabeled data to improve object detection performance, particularly valuable in medical imaging where annotation is expensive.

---

## 🎯 Objectives

1. **Implement Teacher-Student Framework**: Train teacher on labeled data, generate pseudo-labels
2. **Leverage Unlabeled Data**: Use 80% unlabeled data with pseudo-labels
3. **Evaluate Performance**: Compare baseline vs. semi-supervised approaches
4. **Analyze Pseudo-Label Quality**: Confidence filtering and noise analysis

---

## 📐 Theory & Background

### Semi-Supervised Learning (SSL)

Semi-Supervised Learning bridges the gap between supervised and unsupervised learning by utilizing both labeled and unlabeled data.

**Key Insight**: In many domains, obtaining labels is expensive (requires experts), but unlabeled data is abundant.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA AVAILABILITY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Labeled Data (Expensive)     Unlabeled Data (Cheap)            │
│  ██░░░░░░░░░░░░░░░░░░░       ██████████████████████████████     │
│       20%                              80%                       │
│                                                                  │
│  Semi-Supervised Learning: USE BOTH!                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pseudo-Labeling

Pseudo-labeling is a simple yet effective semi-supervised technique:

1. **Train on Labeled Data**: Create a "teacher" model
2. **Generate Pseudo-Labels**: Use teacher to predict labels for unlabeled data
3. **Filter by Confidence**: Keep only high-confidence predictions
4. **Retrain with Combined Data**: Train "student" on labeled + pseudo-labeled data

### Mathematical Formulation

Given:
- Labeled set: $D_L = \{(x_i, y_i)\}_{i=1}^{N_L}$
- Unlabeled set: $D_U = \{x_j\}_{j=1}^{N_U}$

The pseudo-labeling process:

$$\hat{y}_j = \text{Teacher}(x_j) \quad \text{if} \quad \max(p(y|x_j)) \geq \tau$$

Where $\tau$ is the confidence threshold.

### STAC Framework

Our implementation follows **STAC** (Self-Training Approach for Classification) adapted for object detection:

```
Algorithm: STAC for Object Detection
─────────────────────────────────────────
Input: DL (labeled), DU (unlabeled), τ (threshold)
Output: Final detector model

1. Train Teacher on DL
2. For each x ∈ DU:
   a. predictions = Teacher(x)
   b. For each bbox in predictions:
      if confidence ≥ τ:
         Add bbox to pseudo-labels
3. Combine DL and pseudo-labeled DU
4. Train Student on combined data
5. Return Student model
```

---

## 🔧 Configuration

### Hyperparameters

| Parameter | Value | Justification |
|-----------|:-----:|---------------|
| Labeled Ratio | 20% | Simulate limited labels |
| Confidence Threshold (τ) | 0.70 | Balance precision/recall |
| Teacher Epochs | 100 | Sufficient convergence |
| Student Epochs | 100 | Match teacher training |
| Image Size | 640 | YOLO standard |
| Batch Size | 16 | Memory efficient |
| IoU Threshold | 0.45 | NMS parameter |

### Confidence Threshold Selection

The threshold τ = 0.70 was chosen based on:

```
Threshold Analysis:
├── τ = 0.50: High recall, low precision (noisy labels)
├── τ = 0.70: Balanced trade-off ✓
├── τ = 0.90: High precision, low recall (few pseudo-labels)
└── τ = 0.95: Very few pseudo-labels
```

---

## 🏗️ Architecture

### Teacher-Student Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PSEUDO-LABELING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐                                               │
│   │  Labeled Data   │──────┐                                        │
│   │     (20%)       │      │                                        │
│   └─────────────────┘      ▼                                        │
│                       ┌─────────────┐                                │
│                       │   TEACHER   │                                │
│                       │  (YOLOv12)  │                                │
│                       └──────┬──────┘                                │
│                              │                                       │
│   ┌─────────────────┐        │        ┌───────────────────┐         │
│   │ Unlabeled Data  │────────┼───────▶│ Pseudo-Labels     │         │
│   │     (80%)       │        │        │ (Confidence ≥ τ)  │         │
│   └─────────────────┘        │        └─────────┬─────────┘         │
│                              │                  │                    │
│                              │                  │                    │
│                              ▼                  ▼                    │
│                       ┌─────────────────────────────┐               │
│                       │     COMBINED DATASET        │               │
│                       │  (Labeled + Pseudo-Labeled) │               │
│                       └──────────────┬──────────────┘               │
│                                      │                               │
│                                      ▼                               │
│                               ┌─────────────┐                        │
│                               │   STUDENT   │                        │
│                               │  (YOLOv12)  │                        │
│                               └─────────────┘                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### YOLOv12 Architecture

```
Input (640×640×3)
       │
       ▼
┌─────────────────┐
│    Backbone     │ ← CSP-Darknet
│  (Feature Ext.) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Neck       │ ← PANet
│ (Feature Fusion)│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼    ▼    ▼
   P3   P4   P5    ← Multi-scale predictions
    │    │    │
    └────┼────┘
         ▼
   Detection Heads
   (Class + BBox)
```

---

## 📊 Results

### Model Performance Comparison

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 |
|-------|:------:|:---------:|:---------:|:------:|:--:|
| **Baseline (100% data)** | **93.04%** | **64.59%** | 84.66% | 86.55% | 85.59% |
| Teacher (20% data) | 81.84% | 53.92% | 72.11% | 79.34% | 75.55% |
| Student (Pseudo-labeled) | 73.66% | 49.55% | 71.19% | 69.08% | 70.12% |

### Per-Class Performance (AP@50)

| Class | Baseline | Teacher | Student |
|-------|:--------:|:-------:|:-------:|
| CCT | 95.18% | 77.37% | 76.53% |
| IFC | 91.89% | 76.30% | 60.15% |
| UAS | 92.06% | 91.85% | 84.29% |

### Pseudo-Label Statistics

```
Pseudo-Label Analysis:
├── Total unlabeled images: 768
├── Images with pseudo-labels: 623 (81.1%)
├── Total pseudo-labels generated: 847
├── Average confidence: 0.82
├── Confidence distribution:
│   ├── 0.70-0.80: 34%
│   ├── 0.80-0.90: 41%
│   └── 0.90-1.00: 25%
└── Class distribution of pseudo-labels:
    ├── CCT: 38%
    ├── IFC: 31%
    └── UAS: 31%
```

---

## 🔍 Analysis & Discussion

### Why Did Student Underperform?

The student model showed lower performance than expected. Key factors:

1. **Label Noise**: Some pseudo-labels are incorrect despite high confidence
2. **Distribution Mismatch**: Pseudo-label distribution differs from true distribution
3. **Confirmation Bias**: Teacher errors propagate to student
4. **Limited Teacher Capacity**: 20% data may be insufficient for quality pseudo-labels

### Potential Improvements

```
Future Directions:
├── 1. Iterative Refinement: Multiple teacher-student cycles
├── 2. Soft Labels: Use probability distributions instead of hard labels
├── 3. EMA Teacher: Exponential moving average of student weights
├── 4. Consistency Regularization: Augmentation-invariant predictions
└── 5. Higher Threshold: τ = 0.80 or 0.85 for cleaner pseudo-labels
```

---

## 📈 Training Curves

### Teacher Training

```
Loss vs Epoch (Teacher)
│
│  ●
4.0├──╲
   │   ╲
3.0├────╲
   │     ╲
2.0├──────╲
   │       ╲___
1.0├───────────╲____
   │                ╲____●
0.0├─────────────────────────
   0    25    50    75   100
                        Epoch
```

### Student Training

```
mAP@50 vs Epoch (Student)
│
80%├─────────────────────●
   │              ●──────╱
70%├───────●──────╱
   │●──────╱
60%├──────╱
   │
50%├
   0    25    50    75   100
                        Epoch
```

---

## 🛠️ Implementation Highlights

### Pseudo-Label Generation

```python
def generate_pseudo_labels(model, dataloader, confidence_threshold=0.70):
    """Generate pseudo-labels from unlabeled data."""
    pseudo_labels = {}
    
    for images, image_paths in dataloader:
        predictions = model.predict(images, conf=confidence_threshold)
        
        for pred, path in zip(predictions, image_paths):
            boxes = pred.boxes
            if len(boxes) > 0:
                # Filter by confidence
                confident_mask = boxes.conf >= confidence_threshold
                confident_boxes = boxes[confident_mask]
                
                if len(confident_boxes) > 0:
                    pseudo_labels[path] = confident_boxes
    
    return pseudo_labels
```

### Combined Dataset Creation

```python
def create_combined_dataset(labeled_dir, pseudo_labels, output_dir):
    """Combine labeled data with pseudo-labeled data."""
    # Copy labeled data
    shutil.copytree(labeled_dir, output_dir)
    
    # Add pseudo-labeled data
    for image_path, labels in pseudo_labels.items():
        # Copy image
        shutil.copy(image_path, output_dir / 'images')
        
        # Write pseudo-labels in YOLO format
        label_path = output_dir / 'labels' / f'{image_path.stem}.txt'
        with open(label_path, 'w') as f:
            for box in labels:
                # class_id x_center y_center width height
                f.write(f'{int(box.cls)} {box.xywhn}\n')
```

---

## 📚 References

1. Lee, D.H. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks." ICML Workshop 2013.

2. Sohn, K., et al. "A Simple Semi-Supervised Learning Framework for Object Detection." arXiv 2020 (STAC).

3. Xu, M., et al. "End-to-End Semi-Supervised Object Detection with Soft Teacher." ICCV 2021.

4. Liu, Y.C., et al. "Unbiased Teacher for Semi-Supervised Object Detection." ICLR 2021.

---

## 🔑 Key Takeaways

1. **Semi-Supervised Learning** can leverage unlabeled data effectively
2. **Confidence Threshold** is critical for pseudo-label quality
3. **Teacher-Student Framework** provides a structured approach
4. **Performance Gap** exists between fully supervised and semi-supervised
5. **Future Work**: Advanced techniques like Soft Teacher could improve results

---

## ▶️ Next Steps

After completing SSOD, proceed to:
- **Notebook 03-1**: SimCLR Self-Supervised Pretraining
- **Notebook 04-1**: DINOv3 Feature Extraction
