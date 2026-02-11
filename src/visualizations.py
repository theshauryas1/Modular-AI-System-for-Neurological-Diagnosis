"""
Advanced Visualizations Module for Brain MRI Tumor Classification.

Generates publication-quality charts and graphs for comprehensive
model comparison, per-class analysis, and agent pipeline outputs.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, CLASSES, NUM_CLASSES


# ==================== STYLE CONFIGURATION ====================

# Professional color palette
MODEL_COLORS = {
    "resnet50": "#2196F3",
    "resnet101": "#4CAF50",
    "densenet121": "#FF9800",
    "efficientnet_b4": "#9C27B0",
    "mobilenetv2": "#F44336",
}

CLASS_COLORS = {
    "glioma": "#E53935",
    "meningioma": "#1E88E5",
    "pituitary": "#43A047",
}

def _get_model_color(model_name: str) -> str:
    """Get color for a model, with fallback."""
    return MODEL_COLORS.get(model_name, "#607D8B")

def _setup_style():
    """Set up consistent plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ==============================================================================
# 1. RADAR / SPIDER CHART — Multi-metric model comparison
# ==============================================================================

def plot_radar_comparison(
    model_reports: Dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Create radar/spider chart comparing all models across multiple metrics.

    Args:
        model_reports: Dict of model_name -> ModelReport (dataclass with metrics)
        save_path: Path to save the figure
    """
    _setup_style()

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Calibration", "Speed"]
    n_metrics = len(metrics)

    # Compute angles for radar
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for model_name, report in model_reports.items():
        # Normalize metrics to 0-1 scale
        cal_score = max(0, 1 - report.ece * 10)
        speed_score = max(0, 1 - report.inference_time_ms / 200)

        values = [
            report.accuracy,
            report.precision,
            report.recall,
            report.f1_score,
            cal_score,
            speed_score,
        ]
        values += values[:1]  # Close the polygon

        color = _get_model_color(model_name)
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9, color="grey")
    ax.set_title("Multi-Metric Model Comparison\n", fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11,
              frameon=True, fancybox=True, shadow=True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# 2. ROC CURVES — Per-class with AUC
# ==============================================================================

def plot_roc_curves(
    predictions_dict: Dict[str, Dict],
    class_names: List[str] = CLASSES,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot One-vs-Rest ROC curves for each class, overlaying all models.

    Args:
        predictions_dict: Dict of model_name -> {y_true, y_prob}
        class_names: List of class names
        save_path: Path to save the figure
    """
    _setup_style()
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6))
    if n_classes == 1:
        axes = [axes]

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]

        for model_name, preds in predictions_dict.items():
            y_true = preds["y_true"]
            y_prob = preds["y_prob"]

            # Binarize for OvR
            y_true_bin = (np.array(y_true) == class_idx).astype(int)
            y_score = np.array(y_prob)[:, class_idx]

            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)

            color = _get_model_color(model_name)
            ax.plot(fpr, tpr, linewidth=2, color=color,
                    label=f"{model_name} (AUC={roc_auc:.3f})")

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {class_name.capitalize()}", fontweight='bold')
        ax.legend(fontsize=9, loc='lower right', frameon=True)
        ax.grid(alpha=0.3)

    plt.suptitle("Receiver Operating Characteristic (One-vs-Rest)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# 3. PRECISION-RECALL CURVES — Per-class
# ==============================================================================

def plot_precision_recall_curves(
    predictions_dict: Dict[str, Dict],
    class_names: List[str] = CLASSES,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot precision-recall curves for each class, overlaying all models.

    Args:
        predictions_dict: Dict of model_name -> {y_true, y_prob}
        class_names: List of class names
        save_path: Path to save the figure
    """
    _setup_style()
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6))
    if n_classes == 1:
        axes = [axes]

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]

        for model_name, preds in predictions_dict.items():
            y_true = preds["y_true"]
            y_prob = preds["y_prob"]

            # Binarize for OvR
            y_true_bin = (np.array(y_true) == class_idx).astype(int)
            y_score = np.array(y_prob)[:, class_idx]

            precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
            ap = average_precision_score(y_true_bin, y_score)

            color = _get_model_color(model_name)
            ax.plot(recall, precision, linewidth=2, color=color,
                    label=f"{model_name} (AP={ap:.3f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {class_name.capitalize()}", fontweight='bold')
        ax.legend(fontsize=9, loc='lower left', frameon=True)
        ax.grid(alpha=0.3)

    plt.suptitle("Precision-Recall Curves (One-vs-Rest)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# 4. MODEL RANKING HEATMAP — All metrics at a glance
# ==============================================================================

def plot_metrics_heatmap(
    model_reports: Dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Create heatmap with rows=models, columns=metrics, cells colored by value.

    Args:
        model_reports: Dict of model_name -> ModelReport
        save_path: Path to save the figure
    """
    _setup_style()

    # Build data matrix
    metrics_names = [
        "Accuracy", "Precision", "Recall", "F1 Score",
        "ECE ↓", "Inference (ms) ↓", "Overall Score"
    ]

    data = []
    model_names = []
    for model_name, report in model_reports.items():
        model_names.append(model_name)
        data.append([
            report.accuracy * 100,
            report.precision * 100,
            report.recall * 100,
            report.f1_score * 100,
            report.ece * 100,         # ×100 for visibility
            report.inference_time_ms,
            report.overall_score * 100,
        ])

    df = pd.DataFrame(data, index=model_names, columns=metrics_names)

    # Create annotation matrix (formatted strings)
    annot = df.copy()
    for col in metrics_names:
        if "ms" in col:
            annot[col] = annot[col].apply(lambda x: f"{x:.1f}")
        elif "ECE" in col:
            annot[col] = annot[col].apply(lambda x: f"{x:.2f}")
        else:
            annot[col] = annot[col].apply(lambda x: f"{x:.1f}%")

    # For heatmap coloring, invert ECE and inference time (lower is better)
    df_colored = df.copy()
    df_colored["ECE ↓"] = df_colored["ECE ↓"].max() - df_colored["ECE ↓"] + df_colored["ECE ↓"].min()
    df_colored["Inference (ms) ↓"] = (df_colored["Inference (ms) ↓"].max() -
                                       df_colored["Inference (ms) ↓"] +
                                       df_colored["Inference (ms) ↓"].min())

    fig, ax = plt.subplots(figsize=(14, max(4, len(model_names) * 1.2)))

    sns.heatmap(
        df_colored,
        annot=annot.values,
        fmt="",
        cmap="YlGnBu",
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Performance (higher = better)", "shrink": 0.8},
        ax=ax,
        square=False,
    )

    ax.set_title("Model Performance Heatmap\n", fontsize=16, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# 5. ERROR DISTRIBUTION — Donut charts per model
# ==============================================================================

def plot_error_distribution(
    predictions_dict: Dict[str, Dict],
    class_names: List[str] = CLASSES,
    save_path: Optional[Path] = None
) -> None:
    """
    Donut charts showing how errors are distributed across classes per model.

    Args:
        predictions_dict: Dict of model_name -> {y_true, y_pred}
        class_names: List of class names
        save_path: Path to save the figure
    """
    _setup_style()
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    class_colors = [CLASS_COLORS.get(c, "#607D8B") for c in class_names]

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        preds = predictions_dict[model_name]
        y_true = np.array(preds["y_true"])
        y_pred = np.array(preds["y_pred"])

        # Find errors
        errors = y_true != y_pred
        error_true_classes = y_true[errors]

        if len(error_true_classes) == 0:
            ax.text(0.5, 0.5, "No Errors!", ha='center', va='center',
                    fontsize=16, fontweight='bold', color='green')
            ax.set_title(model_name, fontsize=13, fontweight='bold')
            ax.axis('off')
            continue

        # Count errors per true class
        error_counts = []
        labels = []
        colors = []
        for ci, class_name in enumerate(class_names):
            count = np.sum(error_true_classes == ci)
            if count > 0:
                error_counts.append(count)
                labels.append(f"{class_name}\n({count})")
                colors.append(class_colors[ci])

        # Donut chart
        wedges, texts, autotexts = ax.pie(
            error_counts, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.78,
            textprops={'fontsize': 10}
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight('bold')

        # Inner circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.55, fc='white')
        ax.add_artist(centre_circle)

        total_errors = sum(error_counts)
        total_samples = len(y_true)
        err_rate = total_errors / total_samples * 100
        ax.text(0, 0, f"{total_errors}\nerrors\n({err_rate:.1f}%)",
                ha='center', va='center', fontsize=11, fontweight='bold')

        ax.set_title(model_name, fontsize=13, fontweight='bold')

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle("Error Distribution by True Class",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# 6. CONFIDENCE DISTRIBUTION — Correct vs Incorrect
# ==============================================================================

def plot_confidence_distributions(
    predictions_dict: Dict[str, Dict],
    save_path: Optional[Path] = None
) -> None:
    """
    Overlapping histograms showing confidence for correct vs incorrect predictions.

    Args:
        predictions_dict: Dict of model_name -> {y_true, y_pred, y_prob}
        save_path: Path to save the figure
    """
    _setup_style()
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        preds = predictions_dict[model_name]
        y_true = np.array(preds["y_true"])
        y_pred = np.array(preds["y_pred"])
        y_prob = np.array(preds["y_prob"])

        # Get max confidence for each prediction
        max_conf = y_prob.max(axis=1)

        correct_mask = y_true == y_pred
        conf_correct = max_conf[correct_mask]
        conf_incorrect = max_conf[~correct_mask]

        # Plot overlapping histograms
        bins = np.linspace(0, 1, 25)
        ax.hist(conf_correct, bins=bins, alpha=0.7, color="#4CAF50",
                label=f"Correct ({len(conf_correct)})", density=True, edgecolor='white')
        if len(conf_incorrect) > 0:
            ax.hist(conf_incorrect, bins=bins, alpha=0.7, color="#F44336",
                    label=f"Incorrect ({len(conf_incorrect)})", density=True, edgecolor='white')

        ax.set_xlabel("Prediction Confidence")
        ax.set_ylabel("Density")
        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, frameon=True)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis='y')

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle("Confidence Distribution: Correct vs Incorrect Predictions",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# 7. TRAINING CURVES OVERLAY — All models on one plot
# ==============================================================================

def plot_training_curves_overlay(
    histories: Dict[str, Dict],
    save_path: Optional[Path] = None
) -> None:
    """
    Overlay all models' training/validation curves on a single plot.

    Args:
        histories: Dict of model_name -> {train_loss, val_loss, train_acc, val_acc}
        save_path: Path to save the figure
    """
    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Validation Accuracy ---
    ax = axes[0]
    for model_name, history in histories.items():
        if "val_acc" in history:
            epochs = range(1, len(history["val_acc"]) + 1)
            color = _get_model_color(model_name)
            ax.plot(epochs, history["val_acc"], linewidth=2.5, color=color,
                    label=model_name, marker='o', markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy Across Models", fontweight='bold')
    ax.legend(fontsize=10, frameon=True, fancybox=True)
    ax.grid(alpha=0.3)

    # --- Validation Loss ---
    ax = axes[1]
    for model_name, history in histories.items():
        if "val_loss" in history:
            epochs = range(1, len(history["val_loss"]) + 1)
            color = _get_model_color(model_name)
            ax.plot(epochs, history["val_loss"], linewidth=2.5, color=color,
                    label=model_name, marker='o', markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Across Models", fontweight='bold')
    ax.legend(fontsize=10, frameon=True, fancybox=True)
    ax.grid(alpha=0.3)

    plt.suptitle("Training Curves Overlay — All Models",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# 8. AGENT PIPELINE SUMMARY DASHBOARD
# ==============================================================================

def plot_agent_dashboard(
    model_reports: Dict,
    save_path: Optional[Path] = None
) -> None:
    """
    Combined dashboard: model ranking, score comparison, and composition analysis.

    Args:
        model_reports: Dict of model_name -> ModelReport
        save_path: Path to save the figure
    """
    _setup_style()

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    sorted_reports = sorted(model_reports.items(),
                            key=lambda x: x[1].overall_score, reverse=True)
    model_names = [r[0] for r in sorted_reports]
    reports = [r[1] for r in sorted_reports]

    # --- Panel 1: Model Ranking Horizontal Bar ---
    ax1 = fig.add_subplot(gs[0, 0])
    scores = [r.overall_score * 100 for r in reports]
    colors = [_get_model_color(m) for m in model_names]
    bars = ax1.barh(model_names[::-1], scores[::-1], color=colors[::-1],
                    edgecolor='white', linewidth=1.5, height=0.6)
    for bar, score in zip(bars, scores[::-1]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{score:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel("Overall Score (%)")
    ax1.set_title("Model Ranking", fontweight='bold')
    ax1.set_xlim(0, max(scores) * 1.15)
    ax1.grid(alpha=0.3, axis='x')

    # --- Panel 2: Accuracy vs F1 Scatter ---
    ax2 = fig.add_subplot(gs[0, 1])
    for m, r in sorted_reports:
        color = _get_model_color(m)
        ax2.scatter(r.accuracy * 100, r.f1_score * 100, s=200, color=color,
                    edgecolors='black', linewidth=1.5, zorder=5)
        ax2.annotate(m, (r.accuracy * 100, r.f1_score * 100),
                     textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_ylabel("F1 Score (%)")
    ax2.set_title("Accuracy vs F1 Score", fontweight='bold')
    ax2.grid(alpha=0.3)
    # Diagonal reference
    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

    # --- Panel 3: Speed vs Accuracy Trade-off ---
    ax3 = fig.add_subplot(gs[0, 2])
    for m, r in sorted_reports:
        color = _get_model_color(m)
        size = r.overall_score * 400 + 50  # Bubble size = overall score
        ax3.scatter(r.inference_time_ms, r.accuracy * 100, s=size, color=color,
                    alpha=0.7, edgecolors='black', linewidth=1.5, zorder=5)
        ax3.annotate(m, (r.inference_time_ms, r.accuracy * 100),
                     textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax3.set_xlabel("Inference Time (ms)")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Speed vs Accuracy\n(bubble size = overall score)", fontweight='bold')
    ax3.grid(alpha=0.3)

    # --- Panel 4: Per-Class Best/Worst F1 ---
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(model_names))
    width = 0.35
    best_f1 = [r.best_class_f1 * 100 for r in reports]
    worst_f1 = [r.worst_class_f1 * 100 for r in reports]
    ax4.bar(x - width/2, best_f1, width, label='Best Class F1', color='#4CAF50',
            edgecolor='white')
    ax4.bar(x + width/2, worst_f1, width, label='Worst Class F1', color='#F44336',
            edgecolor='white')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=30, ha='right')
    ax4.set_ylabel("F1 Score (%)")
    ax4.set_title("Best vs Worst Class F1", fontweight='bold')
    ax4.legend(fontsize=10, frameon=True)
    ax4.grid(alpha=0.3, axis='y')

    # --- Panel 5: Calibration (ECE) Comparison ---
    ax5 = fig.add_subplot(gs[1, 1])
    ece_values = [r.ece for r in reports]
    bar_colors = ['#4CAF50' if e < 0.05 else '#FF9800' if e < 0.1 else '#F44336'
                  for e in ece_values]
    bars = ax5.bar(model_names, ece_values, color=bar_colors, edgecolor='white',
                   linewidth=1.5)
    for bar, val in zip(bars, ece_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax5.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Good threshold')
    ax5.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Warning threshold')
    ax5.set_ylabel("Expected Calibration Error")
    ax5.set_title("Calibration Quality", fontweight='bold')
    ax5.legend(fontsize=9, frameon=True)
    ax5.set_xticklabels(model_names, rotation=30, ha='right')
    ax5.grid(alpha=0.3, axis='y')

    # --- Panel 6: Error Count Comparison ---
    ax6 = fig.add_subplot(gs[1, 2])
    total_errors = [r.total_errors for r in reports]
    hc_errors = [r.high_confidence_errors for r in reports]
    x = np.arange(len(model_names))
    ax6.bar(x, total_errors, width=0.6, label='Total Errors', color='#90CAF9',
            edgecolor='white')
    ax6.bar(x, hc_errors, width=0.6, label='High-Confidence Errors', color='#E53935',
            edgecolor='white')
    ax6.set_xticks(x)
    ax6.set_xticklabels(model_names, rotation=30, ha='right')
    ax6.set_ylabel("Number of Errors")
    ax6.set_title("Error Analysis", fontweight='bold')
    ax6.legend(fontsize=10, frameon=True)
    ax6.grid(alpha=0.3, axis='y')

    plt.suptitle("Agent Pipeline Summary Dashboard",
                 fontsize=18, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close()


# ==============================================================================
# CONVENIENCE: Generate all visualizations
# ==============================================================================

def generate_all_charts(
    model_reports: Dict,
    predictions_dict: Dict[str, Dict],
    histories: Dict[str, Dict],
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Generate all 8 visualization charts.

    Args:
        model_reports: Dict of model_name -> ModelReport
        predictions_dict: Dict of model_name -> {y_true, y_pred, y_prob}
        histories: Dict of model_name -> training history
        output_dir: Directory to save charts

    Returns:
        Dict of chart_name -> save_path
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "comprehensive_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    print("\n📊 Generating advanced visualizations...")

    # 1. Radar chart
    path = output_dir / "radar_comparison.png"
    plot_radar_comparison(model_reports, path)
    saved["radar_comparison"] = path

    # 2. ROC curves
    path = output_dir / "roc_curves.png"
    plot_roc_curves(predictions_dict, CLASSES, path)
    saved["roc_curves"] = path

    # 3. Precision-recall curves
    path = output_dir / "precision_recall_curves.png"
    plot_precision_recall_curves(predictions_dict, CLASSES, path)
    saved["precision_recall_curves"] = path

    # 4. Metrics heatmap
    path = output_dir / "metrics_heatmap.png"
    plot_metrics_heatmap(model_reports, path)
    saved["metrics_heatmap"] = path

    # 5. Error distribution
    path = output_dir / "error_distribution.png"
    plot_error_distribution(predictions_dict, CLASSES, path)
    saved["error_distribution"] = path

    # 6. Confidence distributions
    path = output_dir / "confidence_distributions.png"
    plot_confidence_distributions(predictions_dict, path)
    saved["confidence_distributions"] = path

    # 7. Training curves overlay
    if histories:
        path = output_dir / "training_curves_overlay.png"
        plot_training_curves_overlay(histories, path)
        saved["training_curves_overlay"] = path

    # 8. Agent dashboard
    path = output_dir / "agent_dashboard.png"
    plot_agent_dashboard(model_reports, path)
    saved["agent_dashboard"] = path

    print(f"  ✅ Generated {len(saved)} charts in {output_dir}")
    return saved
