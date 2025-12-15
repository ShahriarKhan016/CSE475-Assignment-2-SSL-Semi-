# CSE 475 Lab Assignment 02 - Semi-Supervised & Self-Supervised Learning

This repository contains the complete implementation and documentation for the CSE 475 Lab Assignment 02 on Semi-Supervised and Self-Supervised Learning techniques for Brain MRI Detection.

## 🏥 Dataset

Brain MRI dataset with 3 classes for medical imaging classification and detection:
- **CCT** (Cerebral Contusion/Trauma)
- **IFC** (Intracranial Hemorrhage/Focal Changes)
- **UAS** (Unspecified Abnormal Signal)

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [01_data_preparation_eda](notebooks/01_data_preparation_eda.ipynb) | Dataset splitting and EDA |
| 2 | [02_ssod_yolo_pseudolabel](notebooks/02_ssod_yolo_pseudolabel.ipynb) | Semi-Supervised Object Detection |
| 3 | [03_1_simclr_pretraining](notebooks/03_1_simclr_pretraining.ipynb) | SimCLR Self-Supervised Pretraining |
| 4 | [03_2_simclr_finetuning](notebooks/03_2_simclr_finetuning.ipynb) | SimCLR Fine-tuning & Evaluation |
| 5 | [04_1_dinov3_featureextraction](notebooks/04_1_dinov3_featureextraction.ipynb) | DINOv3 Feature Extraction |
| 6 | [04_2_dinov3_finetuning](notebooks/04_2_dinov3_finetuning.ipynb) | DINOv3 Fine-tuning & YOLO Integration |

## 📊 Results Summary

| Method | Key Metric | Value |
|--------|------------|:-----:|
| SSOD Baseline | mAP50 | 0.9304 |
| SimCLR Fine-tuning | Accuracy | 90.31% |
| DINOv3 + YOLO | mAP50 | **0.9408** |

## 📚 Documentation

See the [theory](theory/) folder for detailed explanations of each technique:
- [Data Preparation Theory](theory/01_data_preparation_theory.md)
- [SSOD & Pseudo-Labeling Theory](theory/02_ssod_pseudolabeling_theory.md)
- [SimCLR Theory](theory/03_simclr_theory.md)
- [DINOv3 Theory](theory/04_dinov3_theory.md)

## 🔧 Environment

- Python 3.10+
- PyTorch 2.0+
- Ultralytics YOLO
- Hugging Face Transformers

## 📄 License

This project is for educational purposes as part of CSE 475 coursework.
