# Brain MRI Tumor Classification System

A deep learning system for multi-class brain tumor classification from MRI scans using transfer learning with five CNN architectures, evaluated through an AI-agent pipeline.

## Overview

This project classifies brain MRI images into **3 tumor types**:
- **Glioma** — aggressive tumors arising from glial cells
- **Meningioma** — typically benign tumors from meninges
- **Pituitary** — tumors of the pituitary gland

> The study focuses on multi-class classification of brain tumor types using MRI images, assuming pre-identified abnormal scans.

## Project Structure

```
mri/
├── config.py                    # Centralized configuration & hyperparameters
├── preprocess_all.py            # Full pipeline runner
├── check_progress.py            # Training progress checker
├── requirements.txt             # Python dependencies
├── dataset/                     # Raw MRI images (3 classes)
├── processed_dataset/           # Train/Val/Test splits
│   ├── train/                   # 70% — 14,749 images
│   ├── val/                     # 15% — 3,160 images
│   └── test/                    # 15% — 3,163 images
├── models/                      # Saved model checkpoints (.pt)
├── results/                     # All outputs
│   ├── *_metrics.json           # Per-model evaluation metrics
│   ├── *_history.json           # Training history per model
│   ├── training_summary.json    # Combined training summary
│   ├── confusion_matrices/      # Confusion matrix plots
│   ├── performance_plots/       # Training curves & comparisons
│   └── gradcam/                 # Grad-CAM visualizations
└── src/
    ├── split_dataset.py         # Stratified train/val/test splitting
    ├── preprocessing.py         # Image preprocessing & augmentation
    ├── train_models.py          # Model training with early stopping
    ├── evaluate_models.py       # Test set evaluation & metrics
    ├── advanced_analysis.py     # Cross-validation, statistical tests, calibration
    ├── visualizations.py        # Publication-quality charts (8 chart types)
    ├── comprehensive_report.py  # Full analysis report generator
    ├── gradcam.py               # Grad-CAM explainability
    ├── inference.py             # Single-image & batch inference
    ├── agents.py                # AI agent pipeline (model selection, consensus)
    ├── manager.py               # Pipeline orchestration
    └── resource_manager.py      # GPU/CPU resource management
```

## Models & Architecture

Five CNN architectures trained with **transfer learning** (ImageNet-pretrained backbones, fine-tuned final layers):

| Model | Parameters | Trainable | Size | Frozen Strategy |
|-------|-----------|-----------|------|-----------------|
| ResNet-50 | 23.5M | Fine-tuned | 224.7 MB | Last 2 blocks unfrozen |
| ResNet-101 | 42.5M | Fine-tuned | 297.5 MB | Last 2 blocks unfrozen |
| DenseNet-121 | 6.9M | Fine-tuned | 49.7 MB | Last dense block unfrozen |
| EfficientNet-B4 | 17.5M | Fine-tuned | 185.5 MB | Last 3 blocks unfrozen |
| MobileNetV2 | 2.88M | 1.86M | 25.5 MB | Last 3 blocks unfrozen |

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 224 x 224 |
| Batch Size | 32 |
| Max Epochs | 50 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early Stopping | Patience=7, min_delta=0.001 |
| Data Split | 70% train / 15% val / 15% test |

### Data Augmentation
- Random rotation (±20°)
- Horizontal flip (p=0.5)
- Brightness/contrast jitter (±0.2)
- Random zoom (±15%)
- ImageNet normalization

## Results

### Training Summary

| Model | Best Val Acc | Epochs | Training Time | Early Stopped? |
|-------|-------------|--------|--------------|----------------|
| **EfficientNet-B4** | **99.34%** | 34 | 179.0 min | Yes (epoch 34) |
| ResNet-101 | 99.11% | 49 | 230.9 min | Yes (epoch 49) |
| ResNet-50 | 98.73% | 16 | 56.4 min | Yes (epoch 16) |
| DenseNet-121 | 98.70% | 39 | 130.5 min | Yes (epoch 39) |
| MobileNetV2 | 97.34% | 50 | 73.7 min | No (all 50 epochs) |

### Test Set Evaluation (3,163 images)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **ResNet-101** | **99.30%** | **99.28%** | **99.34%** | **99.30%** |
| EfficientNet-B4 | 98.99% | 99.00% | 98.97% | 98.98% |
| DenseNet-121 | 98.93% | 98.91% | 98.94% | 98.93% |
| ResNet-50 | 98.48% | 98.49% | 98.46% | 98.48% |
| MobileNetV2 | 97.47% | 97.46% | 97.51% | 97.48% |

### Per-Class F1 Scores

| Model | Glioma | Meningioma | Pituitary |
|-------|--------|------------|-----------|
| **ResNet-101** | **99.30%** | **99.00%** | **99.61%** |
| EfficientNet-B4 | 99.09% | 98.65% | 99.21% |
| DenseNet-121 | 98.86% | 98.50% | 99.41% |
| ResNet-50 | 98.52% | 97.74% | 99.16% |
| MobileNetV2 | 97.11% | 96.22% | 99.12% |

### AI Agent Recommendation

The AI agent pipeline evaluates models on a weighted reliability score:

| Model | Reliability Score | Tier | Strengths |
|-------|------------------|------|-----------|
| **EfficientNet-B4** | **0.925** | TOP | Excellent F1, high recall, stable, well-calibrated |
| ResNet-101 | 0.899 | HIGH | Excellent F1, high recall, well-calibrated |
| DenseNet-121 | 0.880 | HIGH | High recall |
| ResNet-50 | 0.878 | HIGH | Fast inference |
| MobileNetV2 | 0.839 | HIGH | Fastest inference |

> **Best Overall: EfficientNet-B4** — Highest reliability score (0.925), best validation accuracy (99.34%), and no weaknesses identified by the agent pipeline.
>
> **Best Test Accuracy: ResNet-101** — Highest test set performance (99.30% accuracy, 99.30% F1) with 100% meningioma recall.
>
> **Best for Deployment: MobileNetV2** — Lightest model (25.5 MB), fastest inference, 97.47% accuracy. Ideal for edge/mobile deployment.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# Train all models
python src/train_models.py

# Train a specific model
python src/train_models.py --models resnet101

# Custom training
python src/train_models.py --models resnet50 densenet121 --epochs 30 --batch-size 16
```

### Evaluation
```bash
# Evaluate all models
python src/evaluate_models.py

# Evaluate specific models
python src/evaluate_models.py --models resnet101 efficientnet_b4
```

### Inference
```python
from src.inference import BrainMRIClassifier

classifier = BrainMRIClassifier()
result = classifier.predict("path/to/mri_scan.jpg")
print(result)
```

### Comprehensive Report
```bash
python src/comprehensive_report.py
python src/comprehensive_report.py --cv  # With cross-validation
```

## Technology Stack

- **Framework**: PyTorch with torchvision
- **Models**: ResNet-50/101, DenseNet-121, EfficientNet-B4, MobileNetV2
- **Training**: Transfer learning, early stopping, LR scheduling
- **Evaluation**: scikit-learn metrics, seaborn/matplotlib visualizations
- **Explainability**: Grad-CAM attention maps
- **Agent System**: Multi-model consensus with weighted reliability voting

## Hardware

- **GPU**: NVIDIA CUDA-enabled GPU
- **Training Time**: ~11 hours total (all 5 models)

## License

This project is for academic and research purposes.
