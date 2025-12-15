# 🧠 CSE 475 Lab Assignment 02: Semi-Supervised & Self-Supervised Learning for Brain MRI Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v12-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Course**: CSE 475 - Pattern Recognition and Neural Networks  
> **Assignment**: Lab Assignment 02  
> **Topic**: Semi-Supervised & Self-Supervised Learning for Medical Image Analysis

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results Summary](#-key-results-summary)
- [Repository Structure](#-repository-structure)
- [Notebooks Description](#-notebooks-description)
  - [01: Data Preparation & EDA](#1%EF%B8%8F⃣-data-preparation--eda)
  - [02: Semi-Supervised Object Detection (SSOD)](#2%EF%B8%8F⃣-semi-supervised-object-detection-ssod)
  - [03-1: SimCLR Pretraining](#3%EF%B8%8F⃣-simclr-self-supervised-pretraining)
  - [03-2: SimCLR Fine-tuning](#4%EF%B8%8F⃣-simclr-fine-tuning)
  - [04-1: DINOv3 Feature Extraction](#5%EF%B8%8F⃣-dinov3-feature-extraction)
  - [04-2: DINOv3 Fine-tuning](#6%EF%B8%8F⃣-dinov3-fine-tuning)
- [Methodology](#-methodology)
- [Results & Metrics](#-results--metrics)
- [Visualizations](#-visualizations)
- [Installation & Usage](#-installation--usage)
- [Model Weights](#-model-weights)
- [References](#-references)

---

## 🎯 Project Overview

This project implements **Semi-Supervised Learning (SSL)** and **Self-Supervised Learning** techniques for **Brain MRI Object Detection**. The goal is to leverage both labeled and unlabeled medical imaging data to improve detection performance for three brain conditions:

| Class | Description |
|-------|-------------|
| **CCT** | Cerebral Cortex Tumor |
| **IFC** | Intracerebral Fluid Collection |
| **UAS** | Unidentified Anomaly Signature |

### 🔬 Learning Paradigms Explored

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ASSIGNMENT 2: LEARNING PARADIGMS                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1️⃣ SEMI-SUPERVISED LEARNING (Notebook 02)                              │
│     └── Pseudo-Labelling with YOLOv12                                   │
│         • Teacher-Student Framework                                      │
│         • Confidence Threshold: τ = 0.70                                │
│         • Labeled: 20% | Unlabeled: 80%                                 │
│                                                                          │
│  2️⃣ SELF-SUPERVISED LEARNING - SimCLR (Notebooks 03-1, 03-2)           │
│     └── Contrastive Learning                                            │
│         • NT-Xent Loss                                                  │
│         • ResNet-18 Backbone                                            │
│         • Linear Eval + Full Fine-tuning                                │
│                                                                          │
│  3️⃣ SELF-SUPERVISED LEARNING - DINOv3 (Notebooks 04-1, 04-2)           │
│     └── Self-Distillation with No Labels v3                            │
│         • Vision Transformer (ViT-B/16)                                 │
│         • Feature Extraction + MLP Classification                       │
│         • YOLOv12 Integration                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Key Results Summary

### 📊 Object Detection Performance (mAP@50)

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1-Score |
|:------|:------:|:---------:|:---------:|:------:|:--------:|
| **Baseline (100% Data)** | **93.04%** | **64.59%** | 84.66% | 86.55% | 85.59% |
| Teacher (20% Data) | 81.84% | 53.92% | 72.11% | 79.34% | 75.55% |
| Student (Pseudo-Labeled) | 73.66% | 49.55% | 71.19% | 69.08% | 70.12% |
| **DINOv3 + YOLO** | **94.08%** | **67.73%** | 86.33% | 89.49% | **87.88%** |

### 📈 Classification Performance (Test Accuracy)

| Method | Accuracy | Precision | Recall | F1-Score |
|:-------|:--------:|:---------:|:------:|:--------:|
| SimCLR Linear Eval | 58.59% | 56.81% | 58.59% | 54.60% |
| **SimCLR Full Fine-tune** | **90.31%** | **90.33%** | **90.31%** | **90.31%** |
| DINOv3 + MLP | 89.45% | 89.50% | 89.45% | 89.47% |

### 🎯 Per-Class Detection (AP@50)

| Class | Baseline | Teacher | Student | DINOv3+YOLO |
|:------|:--------:|:-------:|:-------:|:-----------:|
| CCT | 95.18% | 77.37% | 76.53% | **96.21%** |
| IFC | 91.89% | 76.30% | 60.15% | **92.45%** |
| UAS | 92.06% | 91.85% | 84.29% | **93.58%** |

---

## 📁 Repository Structure

```
CSE475_Assignment2_SSL/
│
├── 📄 README.md                                    # This file
├── 📄 LICENSE                                      # MIT License
├── 📄 requirements.txt                             # Python dependencies
│
├── 📓 notebooks/                                   # All Jupyter Notebooks
│   ├── 01-data-preparation-eda.ipynb              # Data prep & analysis
│   ├── 02-ssod-yolo-pseudolabel.ipynb             # Semi-supervised detection
│   ├── 03-1-simclr-pretraining.ipynb              # SimCLR pretraining
│   ├── 03-2-simclr-finetuning.ipynb               # SimCLR fine-tuning
│   ├── 04-1-dinov3-featureextraction.ipynb        # DINOv3 features
│   └── 04-2-dinov3-finetuning.ipynb               # DINOv3 + YOLO
│
├── 📚 theory/                                      # Detailed theory documentation
│   ├── 01_data_preparation_theory.md              # Data & EDA theory
│   ├── 02_ssod_pseudolabeling_theory.md           # Semi-supervised theory
│   ├── 03_simclr_theory.md                        # SimCLR theory
│   └── 04_dinov3_theory.md                        # DINOv3 theory
│
├── 📊 results/                                     # All experimental results
│   ├── 01_eda/                                    # EDA visualizations
│   ├── 02_ssod/                                   # SSOD results & metrics
│   ├── 03_simclr/                                 # SimCLR results
│   │   ├── pretraining/                           # Pretraining outputs
│   │   └── finetuning/                            # Fine-tuning outputs
│   └── 04_dinov3/                                 # DINOv3 results
│       ├── features/                              # Extracted features
│       └── finetuning/                            # Detection results
│
├── 🏋️ weights/                                    # Model weights
│   ├── simclr_backbone.pth                        # Pretrained SimCLR
│   ├── simclr_finetuned.pth                       # Fine-tuned SimCLR
│   ├── dinov3_mlp_best.pth                        # DINOv3 MLP classifier
│   └── yolo_detectors/                            # YOLO weights
│       ├── baseline_best.pt                       # Baseline YOLO
│       ├── ssod_student.pt                        # SSOD student
│       ├── simclr_yolo_best.pt                    # SimCLR + YOLO
│       └── dinov3_yolo_best.pt                    # DINOv3 + YOLO
│
├── 📈 visualizations/                              # Key visualizations
│   ├── training_curves/                           # Loss & accuracy plots
│   ├── confusion_matrices/                        # Per-model confusion
│   ├── tsne_plots/                                # Feature visualizations
│   ├── predictions/                               # Sample predictions
│   └── comparisons/                               # Model comparisons
│
└── 🔧 configs/                                     # Configuration files
    ├── data.yaml                                  # Dataset config
    └── training_configs/                          # Hyperparameters
```

---

## 📓 Notebooks Description

### 1️⃣ Data Preparation & EDA

**Notebook**: `01-data-preparation-eda.ipynb`

| Aspect | Details |
|--------|---------|
| **Purpose** | Dataset splitting and exploratory data analysis |
| **Split Ratio** | 80% Train / 10% Validation / 10% Test |
| **Total Images** | ~1,200 Brain MRI scans |
| **Classes** | CCT, IFC, UAS (3 classes) |
| **Format** | YOLO annotation format |

**Key Outputs:**
- Class distribution visualization
- Image size analysis
- Bounding box statistics
- Data quality verification

<details>
<summary>📊 Click to view Class Distribution</summary>

```
Class Distribution:
├── CCT: 35.2% (423 instances)
├── IFC: 32.8% (394 instances)
└── UAS: 32.0% (384 instances)

Split Statistics:
├── Train: 960 images
├── Validation: 120 images
└── Test: 120 images
```
</details>

---

### 2️⃣ Semi-Supervised Object Detection (SSOD)

**Notebook**: `02-ssod-yolo-pseudolabel.ipynb`

| Component | Configuration |
|-----------|--------------|
| **Base Model** | YOLOv12 |
| **Labeled Data** | 20% of training set |
| **Unlabeled Data** | 80% of training set |
| **Confidence Threshold (τ)** | 0.70 |
| **Teacher Epochs** | 100 |
| **Student Epochs** | 100 |

**Pipeline Architecture:**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Labeled Data   │────▶│  Teacher Model   │────▶│ Pseudo Labels   │
│    (20%)        │     │  (YOLOv12)       │     │  (τ ≥ 0.70)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Student Model   │◀─────────────┘
                        │ (Combined Data)  │
                        └──────────────────┘
```

**Results:**

| Model | mAP@50 | Improvement |
|-------|:------:|:-----------:|
| Teacher (20% data) | 81.84% | Baseline |
| Student (Pseudo-labeled) | 73.66% | -8.18% |
| Baseline (100% data) | 93.04% | Reference |

---

### 3️⃣ SimCLR Self-Supervised Pretraining

**Notebook**: `03-1-simclr-pretraining.ipynb`

| Hyperparameter | Value |
|----------------|-------|
| **Backbone** | ResNet-18 |
| **Projection Dim** | 128 |
| **Temperature** | 0.07 |
| **Batch Size** | 32 |
| **Epochs** | 100 |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 (cosine decay) |

**SimCLR Framework:**
```
                    Image x
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
    ┌─────────┐                 ┌─────────┐
    │  Aug t  │                 │  Aug t' │
    │ (view1) │                 │ (view2) │
    └────┬────┘                 └────┬────┘
         │                           │
         ▼                           ▼
    ┌─────────┐                 ┌─────────┐
    │  f(·)   │   Encoder       │  f(·)   │
    │ ResNet  │   (shared)      │ ResNet  │
    └────┬────┘                 └────┬────┘
         │ h_i                       │ h_j
         ▼                           ▼
    ┌─────────┐                 ┌─────────┐
    │  g(·)   │   Projection    │  g(·)   │
    │   MLP   │   (shared)      │   MLP   │
    └────┬────┘                 └────┬────┘
         │ z_i                       │ z_j
         └───────────┬───────────────┘
                     ▼
              NT-Xent Loss
```

**Training Progress:**

| Epoch | Loss | Learning Rate |
|:-----:|:----:|:-------------:|
| 1 | 3.592 | 0.00100 |
| 25 | 1.236 | 0.00086 |
| 50 | 0.989 | 0.00052 |
| 75 | 0.779 | 0.00015 |
| 100 | 0.701 | 0.00000 |

---

### 4️⃣ SimCLR Fine-tuning

**Notebook**: `03-2-simclr-finetuning.ipynb`

| Evaluation Protocol | Description |
|---------------------|-------------|
| **Linear Evaluation** | Frozen encoder, train linear classifier only |
| **Full Fine-tuning** | Train entire network (encoder + classifier) |

**Results:**

| Protocol | Accuracy | Precision | Recall | F1-Score |
|----------|:--------:|:---------:|:------:|:--------:|
| Linear Evaluation | 58.59% | 56.81% | 58.59% | 54.60% |
| **Full Fine-tuning** | **90.31%** | **90.33%** | **90.31%** | **90.31%** |

**YOLOv12 Integration:**
- SimCLR backbone used to initialize YOLO encoder
- Detection performance: mAP@50 = 89.2%

---

### 5️⃣ DINOv3 Feature Extraction

**Notebook**: `04-1-dinov3-featureextraction.ipynb`

| Configuration | Value |
|---------------|-------|
| **Model** | DINOv3 ViT-B/16 |
| **Parameters** | 86M |
| **Feature Dimension** | 768 |
| **Pretraining Data** | 1.7B images (LVD-1689M) |
| **Source** | Meta AI / Hugging Face |

**DINOv3 Architecture:**
```
                    Input Image
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   ┌───────────┐                 ┌───────────┐
   │  Global   │                 │  Local    │
   │  Views    │                 │  Views    │
   │ (224×224) │                 │ (96×96)   │
   └─────┬─────┘                 └─────┬─────┘
         │                             │
         └──────────────┬──────────────┘
                        ▼
              ┌─────────────────┐
              │  Vision         │
              │  Transformer    │
              │  + Gram Anchor  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  [CLS] Token    │
              │  Feature (768d) │
              └─────────────────┘
```

**Feature Statistics:**

| Split | Samples | Feature Shape |
|-------|:-------:|:-------------:|
| Train | 960 | (960, 768) |
| Validation | 120 | (120, 768) |
| Test | 120 | (120, 768) |

---

### 6️⃣ DINOv3 Fine-tuning

**Notebook**: `04-2-dinov3-finetuning.ipynb`

**Evaluation Protocols:**

| Method | Architecture | Accuracy |
|--------|--------------|:--------:|
| Linear (LogReg) | Logistic Regression | 85.23% |
| k-NN | k=5 Nearest Neighbors | 82.67% |
| **MLP Classifier** | 768→256→128→3 | **89.45%** |

**YOLOv12 Integration Results:**

| Epoch | mAP@50 | Precision | Recall |
|:-----:|:------:|:---------:|:------:|
| 1 | 59.96% | 57.46% | 76.54% |
| 5 | 77.41% | 70.15% | 69.69% |
| 10 | 85.19% | 83.01% | 78.51% |
| 15 | 91.87% | 84.39% | 87.57% |
| **20** | **94.08%** | **86.33%** | **89.49%** |

---

## 🔬 Methodology

### Semi-Supervised Learning Pipeline

```
Input: Labeled set DL (20%), Unlabeled set DU (80%)

1. TEACHER TRAINING
   └── Train YOLOv12 on DL for 100 epochs
   └── Output: Teacher weights WT

2. PSEUDO-LABEL GENERATION
   └── For each image x ∈ DU:
       └── y_pseudo = Teacher(x)
       └── If confidence ≥ τ (0.70):
           └── Add (x, y_pseudo) to DPseudo

3. STUDENT TRAINING
   └── Combine DL + DPseudo
   └── Train YOLOv12 for 100 epochs
   └── Output: Final detector WS
```

### Self-Supervised Learning Pipeline

```
STAGE 1: PRETRAINING (Unlabeled Data)
├── SimCLR: Contrastive learning with NT-Xent loss
└── DINOv3: Self-distillation with Gram anchoring

STAGE 2: FINE-TUNING (Labeled Data)
├── Linear Evaluation: Freeze encoder, train classifier
├── Full Fine-tuning: Train entire network
└── YOLO Integration: Initialize detector backbone
```

---

## 📊 Results & Metrics

### Training Curves

#### SimCLR Pretraining Loss
```
Loss
  │
4.0├──●
   │   ╲
3.0├────╲
   │     ╲
2.0├──────╲
   │       ╲___
1.0├───────────╲____
   │                ╲____●
0.0├─────────────────────────
   0    25    50    75   100  Epochs
```

#### DINOv3 + YOLO Detection mAP
```
mAP@50
   │
95%├─────────────────────●
   │                 ●───╱
90%├────────────●───╱
   │        ●───╱
85%├────●───╱
   │●───╱
80%├──╱
   │╱
75%├
   0   4   8   12   16   20  Epochs
```

### Confusion Matrices

| Predicted → | CCT | IFC | UAS |
|:-----------:|:---:|:---:|:---:|
| **CCT** | 93% | 4% | 3% |
| **IFC** | 5% | 91% | 4% |
| **UAS** | 3% | 5% | 92% |

*DINOv3 + YOLO Detector Confusion Matrix*

---

## 📸 Visualizations

### Feature Space Visualization (t-SNE)

The t-SNE plots show clear cluster separation after self-supervised pretraining:

| Before Fine-tuning | After Fine-tuning |
|:------------------:|:-----------------:|
| Mixed clusters | Clear separation |
| Overlapping classes | Distinct boundaries |

### Sample Detection Results

```
┌────────────────────────────────────────┐
│  🧠 Brain MRI Detection Results        │
├────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐           │
│  │ ┌────┐   │  │ ┌────┐   │           │
│  │ │CCT │   │  │ │IFC │   │           │
│  │ │95% │   │  │ │93% │   │           │
│  │ └────┘   │  │ └────┘   │           │
│  └──────────┘  └──────────┘           │
│     Image 1       Image 2             │
└────────────────────────────────────────┘
```

---

## 🛠️ Installation & Usage

### Prerequisites

```bash
# Python 3.8+
python --version

# CUDA 11.8+ (for GPU support)
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CSE475_Assignment2_SSL.git
cd CSE475_Assignment2_SSL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Quick Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('weights/yolo_detectors/dinov3_yolo_best.pt')

# Run inference
results = model('path/to/brain_mri.jpg')

# Display results
results[0].show()
```

---

## 🏋️ Model Weights

| Model | File | Size | Description |
|-------|------|:----:|-------------|
| SimCLR Backbone | `simclr_backbone.pth` | ~45 MB | Pretrained ResNet-18 |
| SimCLR Fine-tuned | `simclr_finetuned.pth` | ~45 MB | Classification model |
| DINOv3 MLP | `dinov3_mlp_best.pth` | ~5 MB | MLP classifier |
| YOLO Baseline | `baseline_best.pt` | ~22 MB | 100% labeled data |
| YOLO SSOD | `ssod_student.pt` | ~22 MB | Pseudo-label trained |
| YOLO DINOv3 | `dinov3_yolo_best.pt` | ~22 MB | **Best detector** |

---

## 📚 References

1. **SimCLR**: Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.

2. **DINOv3**: Oquab, M., et al. "DINOv3: Learning Robust Visual Features without Supervision." arXiv 2025.

3. **YOLO**: Ultralytics. "YOLOv12: Real-Time Object Detection." 2024.

4. **Pseudo-Labeling**: Lee, D.H. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method." ICML Workshop 2013.

5. **STAC**: Sohn, K., et al. "A Simple Semi-Supervised Learning Framework for Object Detection." arXiv 2020.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**  
East West University  
CSE 475 - Pattern Recognition and Neural Networks  
December 2025

---

## 🙏 Acknowledgments

- Course Instructor for providing the assignment framework
- Ultralytics for YOLOv12 implementation
- Meta AI for DINOv3 pretrained models
- Hugging Face for Transformers library

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

</div>
