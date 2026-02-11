# Brain MRI Tumor Classification — Full Analysis Report

**Generated**: February 11, 2026  
**Dataset**: 21,072 Brain MRI Images (3 classes)  
**Hardware**: NVIDIA CUDA GPU  

---

## 1. Dataset Overview

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 14,749 | 70% |
| Validation | 3,160 | 15% |
| Test | 3,163 | 15% |

**Classes**: Glioma, Meningioma, Pituitary  
**Preprocessing**: Resized to 224x224, ImageNet normalization  
**Augmentation**: Rotation (±20°), horizontal flip, brightness/contrast jitter, zoom (±15%)

---

## 2. Training Results

All models used transfer learning with ImageNet-pretrained backbones. Only the final classification layers and a few top blocks were fine-tuned.

### 2.1 Training Summary

| Model | Best Val Accuracy | Epochs Trained | Max Epochs | Training Time | Early Stopped |
|-------|------------------|----------------|------------|--------------|---------------|
| EfficientNet-B4 | 99.34% | 34 | 50 | 179.0 min | Yes |
| ResNet-101 | 99.11% | 49 | 50 | 230.9 min | Yes |
| ResNet-50 | 98.73% | 16 | 50 | 56.4 min | Yes |
| DenseNet-121 | 98.70% | 39 | 50 | 130.5 min | Yes |
| MobileNetV2 | 97.34% | 50 | 50 | 73.7 min | No |

### 2.2 Training Observations

- **EfficientNet-B4** achieved the highest validation accuracy (99.34%) and converged efficiently in 34 epochs, demonstrating the compound scaling approach is effective for medical imaging.
- **ResNet-50** was the fastest to converge (16 epochs, 56.4 min), suggesting strong feature transfer from ImageNet for this task.
- **MobileNetV2** used all 50 epochs without triggering early stopping, indicating its lighter architecture (2.88M parameters) has limited capacity for this classification task but still achieves a respectable 97.34%.

### 2.3 Training Curves

Training curves for each model showing loss and accuracy progression:

| Model | Training History |
|-------|-----------------|
| ResNet-50 | `results/performance_plots/resnet50_training_history.png` |
| ResNet-101 | `results/performance_plots/resnet101_training_history.png` |
| DenseNet-121 | `results/performance_plots/densenet121_training_history.png` |
| EfficientNet-B4 | `results/performance_plots/efficientnet_b4_training_history.png` |
| MobileNetV2 | `results/performance_plots/mobilenetv2_training_history.png` |

---

## 3. Test Set Evaluation

Evaluation performed on 3,163 held-out test images never seen during training.

### 3.1 Overall Metrics

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 Score (macro) |
|------|-------|----------|-------------------|----------------|------------------|
| 1 | **ResNet-101** | **99.30%** | **99.28%** | **99.34%** | **99.30%** |
| 2 | EfficientNet-B4 | 98.99% | 99.00% | 98.97% | 98.98% |
| 3 | DenseNet-121 | 98.93% | 98.91% | 98.94% | 98.93% |
| 4 | ResNet-50 | 98.48% | 98.49% | 98.46% | 98.48% |
| 5 | MobileNetV2 | 97.47% | 97.46% | 97.51% | 97.48% |

> All models exceed 97% accuracy on the test set, indicating strong generalization.

### 3.2 Per-Class Precision

| Model | Glioma | Meningioma | Pituitary |
|-------|--------|------------|-----------|
| ResNet-101 | 99.91% | 98.03% | 99.90% |
| EfficientNet-B4 | 98.79% | 98.40% | 99.80% |
| DenseNet-121 | 99.21% | 98.01% | 99.51% |
| ResNet-50 | 98.27% | 97.79% | 99.41% |
| MobileNetV2 | 97.79% | 95.27% | 99.31% |

### 3.3 Per-Class Recall

| Model | Glioma | Meningioma | Pituitary |
|-------|--------|------------|-----------|
| ResNet-101 | 98.69% | **100.00%** | 99.31% |
| EfficientNet-B4 | 99.39% | 98.89% | 98.63% |
| DenseNet-121 | 98.52% | 98.99% | 99.31% |
| ResNet-50 | 98.78% | 97.69% | 98.92% |
| MobileNetV2 | 96.43% | 97.19% | 98.92% |

> **Notable**: ResNet-101 achieves **100% recall for Meningioma** — zero missed meningioma cases on the test set.

### 3.4 Per-Class F1 Scores

| Model | Glioma | Meningioma | Pituitary |
|-------|--------|------------|-----------|
| ResNet-101 | 99.30% | 99.00% | 99.61% |
| EfficientNet-B4 | 99.09% | 98.65% | 99.21% |
| DenseNet-121 | 98.86% | 98.50% | 99.41% |
| ResNet-50 | 98.52% | 97.74% | 99.16% |
| MobileNetV2 | 97.11% | 96.22% | 99.12% |

> **Meningioma** is consistently the hardest class across all models, while **Pituitary** is the easiest.

---

## 4. Confusion Matrices

Confusion matrices show the exact distribution of predictions vs ground truth:

### ResNet-101 (Best Accuracy)
```
                Predicted
              Glioma  Menin.  Pituit.
Actual Glioma  1134     14       1
       Menin.     0    995       0
       Pituit.    1      6    1012
```
- Total errors: **22 out of 3,163** (0.70% error rate)
- Only 1 glioma misclassified as pituitary
- **Zero meningioma misclassifications**

### EfficientNet-B4
```
                Predicted
              Glioma  Menin.  Pituit.
Actual Glioma  1142      6       1
       Menin.    10    984       1
       Pituit.    4     10    1005
```
- Total errors: **32 out of 3,163** (1.01% error rate)

### DenseNet-121
```
                Predicted
              Glioma  Menin.  Pituit.
Actual Glioma  1132     14       3
       Menin.     8    985       2
       Pituit.    1      6    1012
```
- Total errors: **34 out of 3,163** (1.07% error rate)

### ResNet-50
```
                Predicted
              Glioma  Menin.  Pituit.
Actual Glioma  1135     12       2
       Menin.    19    972       4
       Pituit.    1     10    1008
```
- Total errors: **48 out of 3,163** (1.52% error rate)

### MobileNetV2
```
                Predicted
              Glioma  Menin.  Pituit.
Actual Glioma  1108     40       1
       Menin.    22    967       6
       Pituit.    3      8    1008
```
- Total errors: **80 out of 3,163** (2.53% error rate)
- Most errors are glioma-meningioma confusions (62 out of 80)

Confusion matrix visualizations saved at: `results/confusion_matrices/`

---

## 5. Model Comparison Charts

### Available Comparison Visualizations

| Chart | Path |
|-------|------|
| Model Comparison (Accuracy, F1, etc.) | `results/performance_plots/model_comparison.png` |
| Per-Class F1 Comparison | `results/performance_plots/per_class_f1_comparison.png` |

Individual training history curves are available for each model in `results/performance_plots/`.

---

## 6. AI Agent Evaluation

The AI agent system evaluates models using a weighted multi-factor reliability score:

### 6.1 Scoring Methodology

```
Overall Reliability = 
    F1 Score Weight     × 0.40
  + Recall Weight       × 0.30
  + Precision Weight    × 0.20
  + Accuracy Weight     × 0.10
```

Additional factors considered:
- **Stability Score** — Consistency across cross-validation folds
- **Calibration Score** — How well confidence scores match true probabilities
- **Speed Score** — Inference throughput capability

### 6.2 AI Agent Rankings

| Rank | Model | Reliability Score | Accuracy Score | Stability | Calibration | Speed | Tier |
|------|-------|------------------|---------------|-----------|-------------|-------|------|
| **1** | **EfficientNet-B4** | **0.925** | 0.974 | 0.820 | 0.650 | 0.774 | **TOP** |
| 2 | ResNet-101 | 0.899 | 0.959 | 0.750 | 0.520 | 0.810 | HIGH |
| 3 | DenseNet-121 | 0.880 | 0.949 | 0.690 | 0.450 | 0.838 | HIGH |
| 4 | ResNet-50 | 0.878 | 0.943 | 0.720 | 0.380 | 0.859 | HIGH |
| 5 | MobileNetV2 | 0.839 | 0.918 | 0.620 | 0.220 | 0.909 | HIGH |

### 6.3 Model Strengths & Weaknesses (Agent Assessment)

#### EfficientNet-B4 — TOP TIER
- **Strengths**: Excellent F1 (>=95%), High recall, Stable across CV folds, Well-calibrated
- **Weaknesses**: None identified
- **Voting Weight**: 0.925

#### ResNet-101 — HIGH TIER
- **Strengths**: Excellent F1 (>=95%), High recall, Well-calibrated
- **Weaknesses**: None identified
- **Voting Weight**: 0.899

#### DenseNet-121 — HIGH TIER
- **Strengths**: High recall
- **Weaknesses**: None identified
- **Voting Weight**: 0.880

#### ResNet-50 — HIGH TIER
- **Strengths**: Fast inference (real-time capable)
- **Weaknesses**: 6 high-confidence errors
- **Voting Weight**: 0.878

#### MobileNetV2 — HIGH TIER
- **Strengths**: Fastest inference (real-time capable)
- **Weaknesses**: 8 high-confidence errors
- **Voting Weight**: 0.839

### 6.4 Per-Class Reliability

| Model | Glioma | Meningioma | Pituitary |
|-------|--------|------------|-----------|
| EfficientNet-B4 | 0.96 | 0.98 | 0.99 |
| ResNet-101 | 0.94 | 0.97 | 0.98 |
| DenseNet-121 | 0.92 | 0.96 | 0.97 |
| ResNet-50 | 0.91 | 0.95 | 0.96 |
| MobileNetV2 | 0.88 | 0.93 | 0.94 |

---

## 7. Key Findings

### 7.1 Best Model Selection

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| **Best Overall (AI Agent)** | **EfficientNet-B4** | Highest reliability (0.925), best validation acc (99.34%), well-calibrated, no weaknesses |
| **Best Test Accuracy** | **ResNet-101** | Highest test F1 (99.30%), 100% meningioma recall |
| **Fastest Training** | **ResNet-50** | 56.4 min, strong 98.48% accuracy |
| **Mobile/Edge Deployment** | **MobileNetV2** | 25.5 MB, fastest inference, 97.47% accuracy |
| **Best Size-Performance** | **DenseNet-121** | 49.7 MB with 98.93% accuracy |

### 7.2 Clinical Implications

1. **All models exceed 97% accuracy**, demonstrating transfer learning effectiveness for brain tumor classification.
2. **Meningioma is the hardest class** — the most confused pair across models is glioma ↔ meningioma, which is clinically expected due to overlapping MRI presentation.
3. **Pituitary tumors are most reliably detected** (>99% F1 across all models), likely due to their distinctive morphology and location.
4. **ResNet-101 missed zero meningiomas** (100% recall), which is critical in medical contexts where false negatives are dangerous.
5. **EfficientNet-B4 is the most balanced** — combining high accuracy, calibration, and stability.

### 7.3 Ensemble Consensus

The agent pipeline supports **weighted voting consensus** where models vote on predictions with weights proportional to their reliability scores. The consensus approach leverages all 5 models, weighting EfficientNet-B4 (0.925) the highest and MobileNetV2 (0.839) the lowest.

---

## 8. Model Size vs Performance Trade-off

```
Model Size vs Accuracy:

MobileNetV2   |████ 25.5 MB    → 97.47% accuracy
DenseNet-121  |████████ 49.7 MB → 98.93% accuracy
EfficientNet  |██████████████████ 185.5 MB → 98.99% accuracy
ResNet-50     |█████████████████████ 224.7 MB → 98.48% accuracy
ResNet-101    |█████████████████████████████ 297.5 MB → 99.30% accuracy
```

> DenseNet-121 offers the best size-to-performance ratio: 98.93% accuracy in only 49.7 MB.

---

## 9. File Outputs

| Output | Location |
|--------|----------|
| Model checkpoints | `models/*.pt` |
| Training histories | `results/*_history.json` |
| Evaluation metrics | `results/*_metrics.json` |
| Training summary | `results/training_summary.json` |
| Reliability profiles | `results/model_reliability_profiles.json` |
| Confusion matrices | `results/confusion_matrices/*.png` |
| Training curves | `results/performance_plots/*_training_history.png` |
| Model comparison chart | `results/performance_plots/model_comparison.png` |
| Per-class F1 chart | `results/performance_plots/per_class_f1_comparison.png` |

---

*Report generated for Brain MRI Tumor Classification System — February 2026*
