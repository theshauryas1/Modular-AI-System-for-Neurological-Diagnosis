"""
Advanced Analysis Module for Brain MRI Tumor Classification.
Includes k-fold cross-validation, statistical tests, error analysis,
inference time benchmarking, and confidence calibration.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    MODELS_DIR, RESULTS_DIR, PERFORMANCE_PLOTS_DIR,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CLASSES, NUM_CLASSES, DEVICE, MODELS, get_model_path
)
from train_models import (
    get_model, get_train_transforms, get_val_transforms,
    train_epoch, validate
)


# ==================== 1. K-FOLD CROSS-VALIDATION ====================

class CrossValidator:
    """
    K-Fold Cross-Validation for robust model evaluation.
    
    Ensures results are not due to lucky data splits.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_seed: int = 42,
        device: torch.device = DEVICE
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
            device: Device for training
        """
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.device = device
        self.results: Dict[str, Dict] = {}
    
    def create_folds(
        self,
        dataset: datasets.ImageFolder
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified k-fold splits.
        
        Args:
            dataset: Full dataset
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        labels = [label for _, label in dataset.samples]
        
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_seed
        )
        
        folds = []
        indices = np.arange(len(dataset))
        
        for train_idx, val_idx in skf.split(indices, labels):
            folds.append((train_idx, val_idx))
        
        return folds
    
    def cross_validate_model(
        self,
        model_name: str,
        dataset: datasets.ImageFolder,
        epochs: int = 10,  # Reduced epochs for CV
        batch_size: int = BATCH_SIZE
    ) -> Dict:
        """
        Perform k-fold cross-validation for a single model.
        
        Args:
            model_name: Name of the model
            dataset: Full dataset for CV
            epochs: Epochs per fold
            batch_size: Batch size
            
        Returns:
            Dictionary with CV results
        """
        print(f"\n{'='*60}")
        print(f"Cross-Validation: {model_name.upper()} ({self.n_folds}-Fold)")
        print(f"{'='*60}")
        
        folds = self.create_folds(dataset)
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{self.n_folds} ---")
            
            # Create fold datasets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset, batch_size=batch_size,
                shuffle=True, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size,
                shuffle=False, num_workers=4, pin_memory=True
            )
            
            # Build and train model
            model = get_model(model_name).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LEARNING_RATE
            )
            
            best_val_acc = 0
            for epoch in range(epochs):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, self.device
                )
                val_loss, val_acc = validate(model, val_loader, criterion, self.device)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Get final predictions
            model.eval()
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            fold_metrics.append({
                "fold": fold_idx + 1,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            
            print(f"  Accuracy: {accuracy*100:.2f}% | F1: {f1*100:.2f}%")
            
            # Clear memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate statistics
        metrics_df = pd.DataFrame(fold_metrics)
        
        results = {
            "model": model_name,
            "n_folds": self.n_folds,
            "fold_results": fold_metrics,
            "statistics": {
                "accuracy": {
                    "mean": metrics_df["accuracy"].mean(),
                    "std": metrics_df["accuracy"].std(),
                    "min": metrics_df["accuracy"].min(),
                    "max": metrics_df["accuracy"].max()
                },
                "precision": {
                    "mean": metrics_df["precision"].mean(),
                    "std": metrics_df["precision"].std()
                },
                "recall": {
                    "mean": metrics_df["recall"].mean(),
                    "std": metrics_df["recall"].std()
                },
                "f1": {
                    "mean": metrics_df["f1"].mean(),
                    "std": metrics_df["f1"].std()
                }
            }
        }
        
        self.results[model_name] = results
        
        print(f"\n📊 {model_name} CV Results:")
        print(f"   Accuracy: {results['statistics']['accuracy']['mean']*100:.2f}% ± {results['statistics']['accuracy']['std']*100:.2f}%")
        print(f"   F1 Score: {results['statistics']['f1']['mean']*100:.2f}% ± {results['statistics']['f1']['std']*100:.2f}%")
        
        return results
    
    def cross_validate_all(
        self,
        model_names: Optional[List[str]] = None,
        epochs: int = 10
    ) -> Dict:
        """
        Perform cross-validation for all models.
        
        Args:
            model_names: List of models (default: all)
            epochs: Epochs per fold
            
        Returns:
            Dictionary with all CV results
        """
        if model_names is None:
            model_names = list(MODELS.keys())
        
        # Load full dataset
        dataset = datasets.ImageFolder(
            root=str(TRAIN_DIR.parent),  # Use combined dataset
            transform=get_train_transforms()
        )
        
        for model_name in model_names:
            self.cross_validate_model(model_name, dataset, epochs)
        
        return self.results
    
    def generate_cv_report(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """Generate cross-validation summary report."""
        data = []
        
        for model_name, results in self.results.items():
            stats = results["statistics"]
            data.append({
                "Model": model_name,
                "Accuracy": f"{stats['accuracy']['mean']*100:.2f} ± {stats['accuracy']['std']*100:.2f}",
                "Precision": f"{stats['precision']['mean']*100:.2f} ± {stats['precision']['std']*100:.2f}",
                "Recall": f"{stats['recall']['mean']*100:.2f} ± {stats['recall']['std']*100:.2f}",
                "F1 Score": f"{stats['f1']['mean']*100:.2f} ± {stats['f1']['std']*100:.2f}"
            })
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved CV report: {save_path}")
        
        return df


# ==================== 2. STATISTICAL COMPARISON ====================

class StatisticalComparator:
    """
    Statistical comparison between models using paired tests.
    
    Tests whether differences between models are statistically significant.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize comparator.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        self.comparison_results: Dict = {}
    
    def paired_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a: str,
        model_b: str
    ) -> Dict:
        """
        Perform paired t-test between two models.
        
        Args:
            scores_a: Scores from model A (e.g., fold accuracies)
            scores_b: Scores from model B
            model_a: Name of model A
            model_b: Name of model B
            
        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        
        significant = p_value < self.alpha
        
        return {
            "model_a": model_a,
            "model_b": model_b,
            "test": "paired_t_test",
            "t_statistic": t_stat,
            "p_value": p_value,
            "alpha": self.alpha,
            "significant": significant,
            "interpretation": (
                f"{model_a} vs {model_b}: "
                f"{'Significant difference' if significant else 'No significant difference'} "
                f"(p={p_value:.4f})"
            )
        }
    
    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a: str,
        model_b: str
    ) -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative).
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            model_a: Name of model A
            model_b: Name of model B
            
        Returns:
            Dictionary with test results
        """
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
        except ValueError:
            # All differences are zero
            stat, p_value = 0, 1.0
        
        significant = p_value < self.alpha
        
        return {
            "model_a": model_a,
            "model_b": model_b,
            "test": "wilcoxon",
            "statistic": stat,
            "p_value": p_value,
            "alpha": self.alpha,
            "significant": significant,
            "interpretation": (
                f"{model_a} vs {model_b}: "
                f"{'Significant difference' if significant else 'No significant difference'} "
                f"(p={p_value:.4f})"
            )
        }
    
    def compare_all_models(
        self,
        cv_results: Dict[str, Dict],
        metric: str = "accuracy"
    ) -> pd.DataFrame:
        """
        Compare all model pairs statistically.
        
        Args:
            cv_results: Cross-validation results
            metric: Metric to compare (accuracy, f1, etc.)
            
        Returns:
            DataFrame with comparison results
        """
        model_names = list(cv_results.keys())
        comparisons = []
        
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                # Extract fold scores
                scores_a = [fold[metric] for fold in cv_results[model_a]["fold_results"]]
                scores_b = [fold[metric] for fold in cv_results[model_b]["fold_results"]]
                
                # Perform tests
                t_test = self.paired_t_test(scores_a, scores_b, model_a, model_b)
                wilcoxon = self.wilcoxon_test(scores_a, scores_b, model_a, model_b)
                
                comparisons.append({
                    "Model A": model_a,
                    "Model B": model_b,
                    "Mean A": np.mean(scores_a),
                    "Mean B": np.mean(scores_b),
                    "Difference": np.mean(scores_a) - np.mean(scores_b),
                    "t-test p-value": t_test["p_value"],
                    "Wilcoxon p-value": wilcoxon["p_value"],
                    "Significant (t)": t_test["significant"],
                    "Significant (W)": wilcoxon["significant"]
                })
        
        df = pd.DataFrame(comparisons)
        self.comparison_results = df
        
        return df
    
    def generate_significance_matrix(
        self,
        cv_results: Dict[str, Dict],
        metric: str = "accuracy",
        save_path: Optional[Path] = None
    ) -> None:
        """
        Generate and plot significance matrix.
        
        Args:
            cv_results: Cross-validation results
            metric: Metric to compare
            save_path: Path to save plot
        """
        model_names = list(cv_results.keys())
        n_models = len(model_names)
        
        # Create significance matrix
        sig_matrix = np.zeros((n_models, n_models))
        
        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i == j:
                    sig_matrix[i, j] = np.nan
                else:
                    scores_a = [fold[metric] for fold in cv_results[model_a]["fold_results"]]
                    scores_b = [fold[metric] for fold in cv_results[model_b]["fold_results"]]
                    _, p_value = stats.ttest_rel(scores_a, scores_b)
                    sig_matrix[i, j] = p_value
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.eye(n_models, dtype=bool)
        sns.heatmap(
            sig_matrix,
            annot=True,
            fmt=".3f",
            mask=mask,
            cmap="RdYlGn_r",
            xticklabels=model_names,
            yticklabels=model_names,
            vmin=0,
            vmax=0.1,
            ax=ax
        )
        
        ax.set_title("Statistical Significance Matrix (p-values)\nGreen = Significant (p < 0.05)", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()


# ==================== 3. ERROR ANALYSIS ====================

class ErrorAnalyzer:
    """
    Analyzes misclassified samples to identify patterns.
    
    Helps understand model limitations and failure modes.
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.misclassifications: Dict = {}
    
    def analyze_errors(
        self,
        model: nn.Module,
        model_name: str,
        test_loader: DataLoader,
        dataset: datasets.ImageFolder
    ) -> Dict:
        """
        Analyze misclassified samples.
        
        Args:
            model: Trained model
            model_name: Name of the model
            test_loader: Test data loader
            dataset: Test dataset
            
        Returns:
            Dictionary with error analysis
        """
        model.eval()
        
        misclassified = []
        correct = []
        all_confidences = []
        
        sample_idx = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Analyzing errors"):
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    confidence = probs[i, pred_label].item()
                    
                    sample_info = {
                        "index": sample_idx,
                        "path": dataset.samples[sample_idx][0],
                        "true_class": CLASSES[true_label],
                        "predicted_class": CLASSES[pred_label],
                        "confidence": confidence,
                        "true_class_prob": probs[i, true_label].item()
                    }
                    
                    all_confidences.append(confidence)
                    
                    if pred_label != true_label:
                        misclassified.append(sample_info)
                    else:
                        correct.append(sample_info)
                    
                    sample_idx += 1
        
        # Analyze patterns
        error_patterns = self._analyze_patterns(misclassified)
        
        analysis = {
            "model_name": model_name,
            "total_samples": len(correct) + len(misclassified),
            "correct": len(correct),
            "misclassified": len(misclassified),
            "error_rate": len(misclassified) / (len(correct) + len(misclassified)),
            "misclassified_samples": misclassified,
            "error_patterns": error_patterns,
            "confidence_stats": {
                "mean": np.mean(all_confidences),
                "std": np.std(all_confidences),
                "min": np.min(all_confidences),
                "max": np.max(all_confidences)
            }
        }
        
        self.misclassifications[model_name] = analysis
        
        return analysis
    
    def _analyze_patterns(self, misclassified: List[Dict]) -> Dict:
        """Analyze patterns in misclassifications."""
        if not misclassified:
            return {"message": "No misclassifications found"}
        
        # Confusion pairs
        confusion_pairs = defaultdict(int)
        for sample in misclassified:
            pair = (sample["true_class"], sample["predicted_class"])
            confusion_pairs[pair] += 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        # Confidence analysis for errors
        error_confidences = [s["confidence"] for s in misclassified]
        
        # Per-class error rates
        class_errors = defaultdict(int)
        class_totals = defaultdict(int)
        for sample in misclassified:
            class_errors[sample["true_class"]] += 1
        
        return {
            "most_common_confusions": [
                {"true": pair[0], "predicted": pair[1], "count": count}
                for pair, count in sorted_pairs[:5]
            ],
            "error_confidence": {
                "mean": np.mean(error_confidences),
                "std": np.std(error_confidences),
                "high_confidence_errors": sum(1 for c in error_confidences if c > 0.8)
            },
            "per_class_errors": dict(class_errors)
        }
    
    def generate_error_report(
        self,
        model_name: str,
        save_path: Optional[Path] = None
    ) -> str:
        """Generate human-readable error analysis report."""
        if model_name not in self.misclassifications:
            return "No error analysis available for this model"
        
        analysis = self.misclassifications[model_name]
        patterns = analysis["error_patterns"]
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"ERROR ANALYSIS REPORT: {model_name.upper()}")
        lines.append("=" * 60)
        
        lines.append(f"\n📊 Overview")
        lines.append(f"   Total Samples: {analysis['total_samples']}")
        lines.append(f"   Correct: {analysis['correct']}")
        lines.append(f"   Misclassified: {analysis['misclassified']}")
        lines.append(f"   Error Rate: {analysis['error_rate']*100:.2f}%")
        
        lines.append(f"\n🔍 Most Common Confusions")
        for conf in patterns.get("most_common_confusions", []):
            lines.append(f"   {conf['true']} → {conf['predicted']}: {conf['count']} cases")
        
        lines.append(f"\n⚠️ High-Confidence Errors")
        ec = patterns.get("error_confidence", {})
        lines.append(f"   Errors with confidence > 80%: {ec.get('high_confidence_errors', 0)}")
        lines.append(f"   Mean error confidence: {ec.get('mean', 0)*100:.1f}%")
        
        lines.append(f"\n📈 Per-Class Error Distribution")
        for cls, count in patterns.get("per_class_errors", {}).items():
            lines.append(f"   {cls}: {count} errors")
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Saved: {save_path}")
        
        return report


# ==================== 4. INFERENCE TIME ANALYSIS ====================

class InferenceTimeBenchmark:
    """
    Measures prediction time per image for each model.
    
    Helps evaluate real-world deployment feasibility.
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.benchmarks: Dict = {}
    
    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        n_iterations: int = 100,
        batch_size: int = 1,
        warmup_iterations: int = 10
    ) -> Dict:
        """
        Benchmark inference time for a model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            n_iterations: Number of iterations for timing
            batch_size: Batch size for inference
            warmup_iterations: Warmup iterations (not timed)
            
        Returns:
            Dictionary with timing results
        """
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Synchronize CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(n_iterations):
                start = time.perf_counter()
                
                _ = model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        results = {
            "model_name": model_name,
            "batch_size": batch_size,
            "device": str(self.device),
            "n_iterations": n_iterations,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
            "fps": 1000 / np.mean(times) * batch_size
        }
        
        self.benchmarks[model_name] = results
        
        print(f"{model_name}: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms ({results['fps']:.1f} FPS)")
        
        return results
    
    def benchmark_all_models(
        self,
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Benchmark all trained models."""
        if model_names is None:
            model_names = list(MODELS.keys())
        
        print("\n" + "=" * 60)
        print("INFERENCE TIME BENCHMARK")
        print("=" * 60)
        print(f"Device: {self.device}\n")
        
        for model_name in model_names:
            model_path = get_model_path(model_name)
            if not model_path.exists():
                print(f"Skipping {model_name} (not trained)")
                continue
            
            # Load model
            model = get_model(model_name)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.benchmark_model(model, model_name)
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self.generate_benchmark_table()
    
    def generate_benchmark_table(
        self,
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Generate benchmark comparison table."""
        data = []
        
        for model_name, results in self.benchmarks.items():
            data.append({
                "Model": model_name,
                "Mean (ms)": f"{results['mean_ms']:.2f}",
                "Std (ms)": f"{results['std_ms']:.2f}",
                "Min (ms)": f"{results['min_ms']:.2f}",
                "Max (ms)": f"{results['max_ms']:.2f}",
                "FPS": f"{results['fps']:.1f}"
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("Mean (ms)")
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nSaved: {save_path}")
        
        return df


# ==================== 5. CONFIDENCE CALIBRATION ====================

class CalibrationAnalyzer:
    """
    Analyzes model confidence calibration.
    
    Checks if predicted probabilities match actual correctness rates.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibration_data: Dict = {}
    
    def compute_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Compute calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary with calibration metrics
        """
        # Get predicted class and confidence
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=self.n_bins, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_of_positives - mean_predicted_value)) if len(fraction_of_positives) > 0 else 0
        
        results = {
            "model_name": model_name,
            "ece": ece,
            "mce": mce,
            "mean_confidence": confidences.mean(),
            "mean_accuracy": accuracies.mean(),
            "calibration_curve": {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist()
            }
        }
        
        self.calibration_data[model_name] = results
        
        return results
    
    def plot_reliability_diagram(
        self,
        model_name: str,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot reliability diagram for a model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save plot
        """
        if model_name not in self.calibration_data:
            print(f"No calibration data for {model_name}")
            return
        
        data = self.calibration_data[model_name]
        curve = data["calibration_curve"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax1.plot(
            curve["mean_predicted_value"],
            curve["fraction_of_positives"],
            'o-', label=model_name, linewidth=2, markersize=8
        )
        ax1.set_xlabel('Mean Predicted Confidence', fontsize=11)
        ax1.set_ylabel('Fraction of Positives (Accuracy)', fontsize=11)
        ax1.set_title(f'Reliability Diagram\nECE: {data["ece"]:.4f}', fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Confidence histogram
        # (Would need actual confidences - omitted for simplicity)
        ax2.bar(
            range(len(curve["mean_predicted_value"])),
            curve["fraction_of_positives"],
            alpha=0.7, label='Accuracy'
        )
        ax2.bar(
            range(len(curve["mean_predicted_value"])),
            curve["mean_predicted_value"],
            alpha=0.5, label='Confidence'
        )
        ax2.set_xlabel('Confidence Bin', fontsize=11)
        ax2.set_ylabel('Value', fontsize=11)
        ax2.set_title('Confidence vs Accuracy by Bin', fontsize=12)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_all_models(
        self,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot reliability diagrams for all models."""
        n_models = len(self.calibration_data)
        
        if n_models == 0:
            print("No calibration data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        for i, (model_name, data) in enumerate(self.calibration_data.items()):
            curve = data["calibration_curve"]
            ax.plot(
                curve["mean_predicted_value"],
                curve["fraction_of_positives"],
                'o-', label=f"{model_name} (ECE: {data['ece']:.3f})",
                color=colors[i], linewidth=2, markersize=6
            )
        
        ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
        ax.set_ylabel('Fraction Correct', fontsize=12)
        ax.set_title('Reliability Diagrams - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()


# ==================== 6. CLASS-WISE PERFORMANCE ====================

class ClassWiseAnalyzer:
    """
    Detailed per-class performance analysis.
    
    Identifies which tumor types are easier/harder to classify.
    """
    
    def __init__(self):
        self.class_metrics: Dict = {}
    
    def analyze_per_class(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Compute per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary with per-class metrics
        """
        metrics = {}
        
        for i, class_name in enumerate(CLASSES):
            # Binary classification for this class
            true_binary = (y_true == i).astype(int)
            pred_binary = (y_pred == i).astype(int)
            
            # Metrics
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            tn = np.sum((true_binary == 0) & (pred_binary == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Confidence for this class
            class_probs = y_prob[:, i]
            true_class_probs = class_probs[y_true == i]
            
            metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(np.sum(true_binary)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "mean_confidence": float(true_class_probs.mean()) if len(true_class_probs) > 0 else 0,
                "std_confidence": float(true_class_probs.std()) if len(true_class_probs) > 0 else 0
            }
        
        self.class_metrics[model_name] = metrics
        
        return metrics
    
    def plot_class_comparison(
        self,
        metric: str = "f1",
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot per-class metric comparison across models.
        
        Args:
            metric: Metric to plot (precision, recall, f1)
            save_path: Path to save plot
        """
        n_models = len(self.class_metrics)
        n_classes = len(CLASSES)
        
        x = np.arange(n_classes)
        width = 0.8 / n_models
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        for i, (model_name, class_data) in enumerate(self.class_metrics.items()):
            values = [class_data[cls][metric] for cls in CLASSES]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name, color=colors[i])
        
        ax.set_xlabel('Tumor Class', fontsize=12)
        ax.set_ylabel(f'{metric.title()} Score', fontsize=12)
        ax.set_title(f'Per-Class {metric.title()} by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def generate_class_report(self, save_path: Optional[Path] = None) -> str:
        """Generate per-class analysis report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CLASS-WISE PERFORMANCE ANALYSIS")
        lines.append("=" * 70)
        
        for model_name, class_data in self.class_metrics.items():
            lines.append(f"\n### {model_name.upper()}")
            lines.append("-" * 50)
            
            # Create table
            lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
            lines.append("-" * 55)
            
            for cls in CLASSES:
                m = class_data[cls]
                lines.append(
                    f"{cls:<15} {m['precision']*100:>8.1f}%   "
                    f"{m['recall']*100:>8.1f}%   {m['f1']*100:>8.1f}%   {m['support']:>5}"
                )
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\nSaved: {save_path}")
        
        return report


# ==================== COMPREHENSIVE ANALYSIS RUNNER ====================

class AdvancedAnalysisPipeline:
    """
    Runs all advanced analyses in one pipeline.
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        
        # Initialize analyzers
        self.cross_validator = CrossValidator(n_folds=5, device=device)
        self.stat_comparator = StatisticalComparator()
        self.error_analyzer = ErrorAnalyzer(device=device)
        self.inference_benchmark = InferenceTimeBenchmark(device=device)
        self.calibration_analyzer = CalibrationAnalyzer()
        self.class_analyzer = ClassWiseAnalyzer()
    
    def run_full_analysis(
        self,
        model_names: Optional[List[str]] = None,
        run_cv: bool = True,
        cv_epochs: int = 10
    ) -> Dict:
        """
        Run comprehensive analysis pipeline.
        
        Args:
            model_names: Models to analyze
            run_cv: Whether to run cross-validation
            cv_epochs: Epochs for CV
            
        Returns:
            Dictionary with all results
        """
        if model_names is None:
            model_names = list(MODELS.keys())
        
        results = {}
        output_dir = RESULTS_DIR / "advanced_analysis"
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print("ADVANCED ANALYSIS PIPELINE")
        print("=" * 70)
        
        # 1. Cross-validation (optional - time intensive)
        if run_cv:
            print("\n📊 1. Cross-Validation Analysis")
            cv_results = self.cross_validator.cross_validate_all(
                model_names, epochs=cv_epochs
            )
            results["cross_validation"] = cv_results
            
            cv_report = self.cross_validator.generate_cv_report(
                output_dir / "cv_results.csv"
            )
            print(cv_report)
            
            # 2. Statistical comparison
            print("\n📈 2. Statistical Comparison")
            stat_df = self.stat_comparator.compare_all_models(cv_results)
            stat_df.to_csv(output_dir / "statistical_comparison.csv", index=False)
            print(stat_df.to_string(index=False))
            
            self.stat_comparator.generate_significance_matrix(
                cv_results, save_path=output_dir / "significance_matrix.png"
            )
        
        # 3. Inference benchmarking
        print("\n⏱️ 3. Inference Time Benchmark")
        benchmark_df = self.inference_benchmark.benchmark_all_models(model_names)
        benchmark_df.to_csv(output_dir / "inference_benchmark.csv", index=False)
        results["inference_benchmark"] = self.inference_benchmark.benchmarks
        
        print("\n✅ Analysis complete!")
        print(f"Results saved to: {output_dir}")
        
        return results


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Analysis Tools")
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["cv", "benchmark", "all"],
                        help="Analysis to run")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to analyze")
    parser.add_argument("--cv-epochs", type=int, default=10,
                        help="Epochs per CV fold")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds")
    
    args = parser.parse_args()
    
    pipeline = AdvancedAnalysisPipeline()
    
    if args.analysis == "cv":
        pipeline.cross_validator.cross_validate_all(
            args.models, epochs=args.cv_epochs
        )
    elif args.analysis == "benchmark":
        pipeline.inference_benchmark.benchmark_all_models(args.models)
    else:
        pipeline.run_full_analysis(
            args.models, run_cv=True, cv_epochs=args.cv_epochs
        )
