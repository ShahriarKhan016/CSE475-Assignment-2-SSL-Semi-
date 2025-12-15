# 🔬 Notebooks 03-1 & 03-2: SimCLR Self-Supervised Learning

## Overview

These notebooks implement **SimCLR** (A Simple Framework for Contrastive Learning of Visual Representations) for self-supervised pretraining and downstream fine-tuning. SimCLR learns visual representations by maximizing agreement between differently augmented views of the same image.

---

## 📓 Notebook Structure

| Notebook | Purpose | Output |
|----------|---------|--------|
| **03-1** | SimCLR Pretraining | `simclr_backbone.pth` |
| **03-2** | Fine-tuning & YOLO Integration | Classification & Detection models |

---

## 🎯 Objectives

### Phase 1: Pretraining (Notebook 03-1)
1. Implement SimCLR architecture (encoder + projection head)
2. Create data augmentation pipeline
3. Implement NT-Xent contrastive loss
4. Train on unlabeled images
5. Save pretrained backbone weights

### Phase 2: Fine-tuning (Notebook 03-2)
1. Load pretrained backbone
2. Perform linear evaluation (frozen encoder)
3. Perform full fine-tuning (trainable encoder)
4. Integrate with YOLOv12 for object detection
5. Compare evaluation protocols

---

## 📐 Theory & Background

### Self-Supervised Learning

Self-supervised learning learns representations from **unlabeled data** by solving pretext tasks. The learned representations transfer to downstream tasks.

```
┌─────────────────────────────────────────────────────────────────┐
│                 SELF-SUPERVISED LEARNING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PRETEXT TASK (Unlabeled)        DOWNSTREAM TASK (Labeled)     │
│   ─────────────────────────       ──────────────────────────    │
│                                                                  │
│   Learn general features    →    Transfer to specific task      │
│   No labels required             Small labeled dataset          │
│                                                                  │
│   Examples:                       Examples:                      │
│   • Contrastive Learning          • Classification               │
│   • Masked Image Modeling         • Object Detection             │
│   • Rotation Prediction           • Segmentation                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### SimCLR Framework

SimCLR uses **contrastive learning** to learn representations:

**Key Idea**: Pull together representations of augmented views of the same image, push apart representations of different images.

### SimCLR Architecture

```
                    Image x
                       │
         ┌─────────────┴─────────────┐
         │     Data Augmentation     │
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
         │ h_i (512-d)               │ h_j (512-d)
         ▼                           ▼
    ┌─────────┐                 ┌─────────┐
    │  g(·)   │   Projection    │  g(·)   │
    │   MLP   │   (shared)      │   MLP   │
    └────┬────┘                 └────┬────┘
         │ z_i (128-d)               │ z_j (128-d)
         └───────────┬───────────────┘
                     ▼
              ┌─────────────┐
              │  NT-Xent    │
              │    Loss     │
              └─────────────┘
```

### Data Augmentation Pipeline

SimCLR uses aggressive augmentation to create diverse views:

```python
augmentation_pipeline = Compose([
    RandomResizedCrop(size=224, scale=(0.2, 1.0)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Augmentation Strategy:**

| Augmentation | Parameters | Effect |
|--------------|------------|--------|
| Random Crop | scale=(0.2, 1.0) | Spatial invariance |
| Horizontal Flip | p=0.5 | Orientation invariance |
| Color Jitter | brightness=0.8 | Color invariance |
| Grayscale | p=0.2 | Color-independent features |
| Gaussian Blur | σ=(0.1, 2.0) | Texture/edge focus |

### NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy)

The contrastive loss for a positive pair (i, j):

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

Where:
- $\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}$ (cosine similarity)
- $\tau$ is the temperature parameter
- $N$ is the batch size
- $2N$ total views (two views per image)

**Temperature Effect:**

```
Low τ (0.01):  Sharp distribution → Hard negatives dominate
Medium τ (0.07): Balanced learning ✓ (SimCLR default)
High τ (1.0):   Soft distribution → All negatives contribute equally
```

---

## 🔧 Configuration

### Pretraining Configuration (Notebook 03-1)

| Parameter | Value | Justification |
|-----------|:-----:|---------------|
| Backbone | ResNet-18 | Efficient, good features |
| Projection Dim | 128 | Standard for contrastive |
| Temperature | 0.07 | SimCLR paper recommendation |
| Batch Size | 32 | Balance performance/memory |
| Epochs | 100 | Sufficient convergence |
| Optimizer | Adam | Stable training |
| Learning Rate | 0.001 | With cosine decay |
| Weight Decay | 1e-4 | Regularization |

### Fine-tuning Configuration (Notebook 03-2)

| Parameter | Value | Justification |
|-----------|:-----:|---------------|
| Linear Eval LR | 0.01 | Fast convergence for linear |
| Fine-tune LR | 0.001 | Careful update of encoder |
| Fine-tune Epochs | 50 | Sufficient adaptation |
| Frozen Layers | All (linear) / None (full) | Evaluation protocols |

---

## 🏗️ Model Architecture

### Encoder (f)

```python
class Encoder(nn.Module):
    """ResNet-18 backbone for feature extraction."""
    
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x  # (batch_size, 512)
```

### Projection Head (g)

```python
class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)  # (batch_size, 128)
```

### Complete SimCLR Model

```python
class SimCLR(nn.Module):
    """Complete SimCLR model."""
    
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection = projection_head
    
    def forward(self, x1, x2):
        # Get representations
        h1 = self.encoder(x1)  # (N, 512)
        h2 = self.encoder(x2)  # (N, 512)
        
        # Get projections
        z1 = self.projection(h1)  # (N, 128)
        z2 = self.projection(h2)  # (N, 128)
        
        return h1, h2, z1, z2
```

### NT-Xent Loss Implementation

```python
class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2N, 128)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)
        
        # Create labels (positive pairs)
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(batch_size)
        ]).to(z1.device)
        
        # Mask out self-similarity
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(z1.device)
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute loss
        loss = self.criterion(sim_matrix, labels)
        
        return loss
```

---

## 📊 Results

### Pretraining Results (Notebook 03-1)

**Training Progress:**

| Epoch | Loss | Learning Rate |
|:-----:|:----:|:-------------:|
| 1 | 3.592 | 0.00100 |
| 10 | 1.838 | 0.00098 |
| 25 | 1.236 | 0.00086 |
| 50 | 0.989 | 0.00052 |
| 75 | 0.779 | 0.00015 |
| 100 | 0.701 | ~0.00000 |

**Loss Curve:**
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
0.7├─────────────────────────●
0.0├─────────────────────────────
   0    25    50    75   100  Epochs
```

### Fine-tuning Results (Notebook 03-2)

| Evaluation Protocol | Accuracy | Precision | Recall | F1-Score |
|---------------------|:--------:|:---------:|:------:|:--------:|
| Linear Evaluation | 58.59% | 56.81% | 58.59% | 54.60% |
| **Full Fine-tuning** | **90.31%** | **90.33%** | **90.31%** | **90.31%** |

**Interpretation:**
- **Linear Evaluation** tests raw feature quality without modification
- **Full Fine-tuning** adapts features to the specific task
- The large gap (31.7%) shows features need task-specific adaptation

### Per-Class Performance (Full Fine-tuning)

| Class | Precision | Recall | F1-Score |
|-------|:---------:|:------:|:--------:|
| CCT | 91.2% | 89.8% | 90.5% |
| IFC | 88.9% | 90.1% | 89.5% |
| UAS | 90.7% | 91.0% | 90.8% |

---

## 📈 Training Curves

### Pretraining Loss

```
NT-Xent Loss vs Epoch
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
   0    25    50    75   100  Epoch
```

### Fine-tuning Accuracy

```
Accuracy (%)
   │
100├
   │
90%├───────────────────────●── Full Fine-tune
   │              ●─────────
80%├────────●─────╱
   │  ●─────╱
70%├──╱
   │╱
60%├───────────────────────●── Linear Eval
   │        ●──────────────
50%├──●─────╱
   0    10    20    30    40  Epoch
```

---

## 🔍 Feature Visualization

### t-SNE of Learned Features

**Before Fine-tuning (Linear Evaluation):**
```
          ●●●          ○○○
        ●●●●●●       ○○○○○○
       ●●●●●●●●    ○○○○○○○○
        ●●●●●●  ▲▲▲▲○○○○○
          ●●●  ▲▲▲▲▲▲ ○○○
              ▲▲▲▲▲▲▲▲
             ▲▲▲▲▲▲▲▲▲▲
              ▲▲▲▲▲▲▲▲
               ▲▲▲▲▲▲

● CCT   ○ IFC   ▲ UAS
(Some overlap between classes)
```

**After Full Fine-tuning:**
```
    ●●●●●●               ○○○○○○
   ●●●●●●●●             ○○○○○○○○
  ●●●●●●●●●●           ○○○○○○○○○○
   ●●●●●●●●             ○○○○○○○○
    ●●●●●●               ○○○○○○


              ▲▲▲▲▲▲▲▲
             ▲▲▲▲▲▲▲▲▲▲
            ▲▲▲▲▲▲▲▲▲▲▲▲
             ▲▲▲▲▲▲▲▲▲▲
              ▲▲▲▲▲▲▲▲

● CCT   ○ IFC   ▲ UAS
(Clear class separation)
```

---

## 🛠️ Implementation Highlights

### Two-View DataLoader

```python
class TwoViewDataset(Dataset):
    """Dataset that returns two augmented views of each image."""
    
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply same transform twice (with random augmentations)
        view1 = self.transform(image)
        view2 = self.transform(image)
        
        return view1, view2
    
    def __len__(self):
        return len(self.image_paths)
```

### Training Loop

```python
def train_simclr(model, dataloader, optimizer, criterion, epochs):
    """Train SimCLR model."""
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for view1, view2 in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            
            # Forward pass
            _, _, z1, z2 = model(view1, view2)
            
            # Compute loss
            loss = criterion(z1, z2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
```

---

## 📚 References

1. **SimCLR v1**: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.

2. **SimCLR v2**: Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. (2020). "Big Self-Supervised Models are Strong Semi-Supervised Learners." NeurIPS 2020.

3. **Contrastive Learning Survey**: Le-Khac, P. H., Healy, G., & Smeaton, A. F. (2020). "Contrastive Representation Learning: A Framework and Review."

4. **InfoNCE Loss**: Oord, A. v. d., Li, Y., & Vinyals, O. (2018). "Representation Learning with Contrastive Predictive Coding."

---

## 🔑 Key Takeaways

1. **Contrastive Learning** learns by comparing views of the same image
2. **Data Augmentation** is crucial for creating meaningful positive pairs
3. **Temperature Parameter** controls the concentration of the distribution
4. **Projection Head** improves feature quality (discarded after pretraining)
5. **Linear Evaluation** tests raw feature quality
6. **Full Fine-tuning** adapts features for optimal task performance
7. **90.31% Accuracy** achieved with full fine-tuning on Brain MRI

---

## ▶️ Next Steps

After completing SimCLR experiments, proceed to:
- **Notebook 04-1**: DINOv3 Feature Extraction
- **Notebook 04-2**: DINOv3 Fine-tuning + YOLO Integration
