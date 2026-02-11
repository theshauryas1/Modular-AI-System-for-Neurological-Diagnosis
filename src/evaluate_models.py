"""
Model Evaluation Module for Brain MRI Tumor Classification.
Computes metrics and generates visualizations for all trained models.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEST_DIR, MODELS_DIR, RESULTS_DIR, CONFUSION_MATRICES_DIR,
    PERFORMANCE_PLOTS_DIR, IMAGE_SIZE, BATCH_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, CLASSES, DEVICE,
    MODELS, get_model_path
)
from train_models import get_model, get_val_transforms


# ==================== DATA LOADING ====================

def create_test_loader(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, datasets.ImageFolder]:
    """
    Create test data loader.
    
    Returns:
        Tuple of (test_loader, test_dataset)
    """
    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=get_val_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, test_dataset


# ==================== MODEL LOADING ====================

def load_trained_model(model_name: str, device: torch.device = DEVICE) -> Optional[nn.Module]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on
        
    Returns:
        Loaded model or None if not found
    """
    model_path = get_model_path(model_name)
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    try:
        # Build model architecture
        model = get_model(model_name)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"[OK] Loaded {model_name} (Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
        return model
    
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None


# ==================== PREDICTION ====================

def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions for all samples in the data loader.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run inference on
        
    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Predicting", leave=False):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ==================== METRICS COMPUTATION ====================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASSES
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        class_names: List of class names
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics["per_class"] = {}
    for i, class_name in enumerate(class_names):
        metrics["per_class"][class_name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i])
        }
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


# ==================== VISUALIZATION ====================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: Optional[Path] = None,
    normalize: bool = True
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        model_name: Name of the model
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'{model_name} - Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = f'{model_name} - Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_metrics_comparison(
    all_metrics: Dict[str, Dict],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot comparison of metrics across all models.
    
    Args:
        all_metrics: Dictionary of metrics per model
        save_path: Path to save the plot
    """
    model_names = list(all_metrics.keys())
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(model_names)))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [all_metrics[m][metric] * 100 for m in model_names]
        
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_per_class_metrics(
    all_metrics: Dict[str, Dict],
    class_names: List[str] = CLASSES,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot per-class F1 scores for all models.
    
    Args:
        all_metrics: Dictionary of metrics per model
        class_names: List of class names
        save_path: Path to save the plot
    """
    model_names = list(all_metrics.keys())
    n_models = len(model_names)
    n_classes = len(class_names)
    
    x = np.arange(n_classes)
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    for i, model_name in enumerate(model_names):
        f1_values = [
            all_metrics[model_name]["per_class"][c]["f1"] * 100
            for c in class_names
        ]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_values, width, label=model_name, color=colors[i])
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('Per-Class F1 Scores by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_history(
    model_name: str,
    history: Dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        model_name: Name of the model
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Training Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ==================== EVALUATION PIPELINE ====================

def evaluate_model(
    model_name: str,
    test_loader: DataLoader,
    device: torch.device = DEVICE
) -> Optional[Dict]:
    """
    Evaluate a single model.
    
    Args:
        model_name: Name of the model
        test_loader: Test data loader
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation results or None if model not found
    """
    print(f"\nEvaluating: {model_name}")
    print("-" * 40)
    
    # Load model
    model = load_trained_model(model_name, device)
    if model is None:
        return None
    
    # Get predictions
    y_pred, y_true, y_prob = get_predictions(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Print summary
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Precision (macro): {metrics['precision_macro']*100:.2f}%")
    print(f"  Recall (macro): {metrics['recall_macro']*100:.2f}%")
    print(f"  F1 Score (macro): {metrics['f1_macro']*100:.2f}%")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = CONFUSION_MATRICES_DIR / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, CLASSES, model_name, cm_path)
    
    # Plot training history if available
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        if 'history' in history_data:
            history_plot_path = PERFORMANCE_PLOTS_DIR / f"{model_name}_training_history.png"
            plot_training_history(model_name, history_data['history'], history_plot_path)
    
    # Save metrics
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")
    
    return metrics


def evaluate_all_models(
    model_names: Optional[List[str]] = None,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Evaluate all trained models.
    
    Args:
        model_names: List of model names to evaluate. If None, evaluates all.
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with results for all models
    """
    if model_names is None:
        model_names = list(MODELS.keys())
    
    print("=" * 60)
    print("BRAIN MRI TUMOR CLASSIFICATION - MODEL EVALUATION")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Models to evaluate: {model_names}")
    
    # Create test loader
    test_loader, test_dataset = create_test_loader(batch_size)
    print(f"Test set size: {len(test_dataset)} images")
    print(f"Classes: {test_dataset.classes}")
    
    all_metrics = {}
    
    for model_name in model_names:
        metrics = evaluate_model(model_name, test_loader)
        if metrics is not None:
            all_metrics[model_name] = metrics
    
    if not all_metrics:
        print("\nNo models were evaluated. Make sure models are trained first.")
        return {}
    
    # Generate comparison plots
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)
    
    # Metrics comparison
    comparison_path = PERFORMANCE_PLOTS_DIR / "model_comparison.png"
    plot_metrics_comparison(all_metrics, comparison_path)
    
    # Per-class comparison
    per_class_path = PERFORMANCE_PLOTS_DIR / "per_class_f1_comparison.png"
    plot_per_class_metrics(all_metrics, CLASSES, per_class_path)
    
    # Create summary table
    summary_data = []
    for model_name, metrics in all_metrics.items():
        summary_data.append({
            "Model": model_name,
            "Accuracy (%)": f"{metrics['accuracy']*100:.2f}",
            "Precision (%)": f"{metrics['precision_macro']*100:.2f}",
            "Recall (%)": f"{metrics['recall_macro']*100:.2f}",
            "F1 Score (%)": f"{metrics['f1_macro']*100:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("F1 Score (%)", ascending=False)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = RESULTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    
    # Save all metrics
    all_metrics_path = RESULTS_DIR / "all_metrics.json"
    with open(all_metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Brain MRI Classification Models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    
    args = parser.parse_args()
    
    evaluate_all_models(
        model_names=args.models,
        batch_size=args.batch_size
    )
