"""
Comprehensive Report Generator.
Integrates all analysis modules with the Model Performance Evaluation Agent.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RESULTS_DIR, MODELS_DIR, TEST_DIR, CLASSES, DEVICE, MODELS, get_model_path
)
from train_models import get_model, get_val_transforms
from advanced_analysis import (
    CrossValidator, StatisticalComparator, ErrorAnalyzer,
    InferenceTimeBenchmark, CalibrationAnalyzer, ClassWiseAnalyzer
)
from visualizations import generate_all_charts


# ==================== COMPREHENSIVE MODEL REPORT ====================

@dataclass
class ModelReport:
    """Complete report for a single model."""
    model_name: str
    
    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Cross-validation
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0
    
    # Inference performance
    inference_time_ms: float = 0.0
    inference_fps: float = 0.0
    
    # Calibration
    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Maximum Calibration Error
    
    # Error analysis
    total_errors: int = 0
    high_confidence_errors: int = 0
    most_confused_pair: str = ""
    
    # Class-wise (best/worst)
    best_class: str = ""
    best_class_f1: float = 0.0
    worst_class: str = ""
    worst_class_f1: float = 0.0
    
    # Ranking
    overall_score: float = 0.0
    rank: int = 0
    tier: str = ""


class ComprehensiveReportAgent:
    """
    Master agent that combines all analysis modules.
    
    Generates complete performance reports for all models including:
    - Standard metrics (accuracy, precision, recall, F1)
    - Cross-validation results with confidence intervals
    - Statistical significance comparisons
    - Error analysis and failure patterns
    - Inference time benchmarks
    - Confidence calibration
    - Per-class performance breakdown
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        
        # Initialize all analyzers
        self.cross_validator = CrossValidator(n_folds=5, device=device)
        self.stat_comparator = StatisticalComparator()
        self.error_analyzer = ErrorAnalyzer(device=device)
        self.inference_benchmark = InferenceTimeBenchmark(device=device)
        self.calibration_analyzer = CalibrationAnalyzer()
        self.class_analyzer = ClassWiseAnalyzer()
        
        # Results storage
        self.model_reports: Dict[str, ModelReport] = {}
        self.predictions_dict: Dict[str, Dict] = {}  # model -> {y_true, y_pred, y_prob}
        self.training_histories: Dict[str, Dict] = {}  # model -> history
        self.statistical_comparisons: Optional[pd.DataFrame] = None
        self.output_dir = RESULTS_DIR / "comprehensive_report"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== DATA LOADING ====================
    
    def _load_test_data(self, batch_size: int = 32) -> Tuple:
        """Load test dataset."""
        transform = get_val_transforms()
        test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )
        return test_loader, test_dataset
    
    def _load_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Load a trained model."""
        model_path = get_model_path(model_name)
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return None
        
        model = get_model(model_name)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_predictions(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions from a model."""
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    # ==================== ANALYSIS PIPELINE ====================
    
    def analyze_model(
        self,
        model_name: str,
        run_cv: bool = False,
        cv_epochs: int = 5
    ) -> ModelReport:
        """
        Run complete analysis for a single model.
        
        Args:
            model_name: Name of the model
            run_cv: Whether to run cross-validation
            cv_epochs: Epochs for cross-validation
            
        Returns:
            Complete ModelReport
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING: {model_name.upper()}")
        print(f"{'='*60}")
        
        report = ModelReport(model_name=model_name)
        
        # Load model
        model = self._load_model(model_name)
        if model is None:
            return report
        
        # Load test data
        test_loader, test_dataset = self._load_test_data()
        
        # Get predictions
        y_true, y_pred, y_prob = self._get_predictions(model, test_loader)
        
        # Store predictions for visualization charts
        self.predictions_dict[model_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        
        # 1. Basic metrics
        print("\n[METRICS] Computing basic metrics...")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        report.accuracy = accuracy_score(y_true, y_pred)
        report.precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        report.recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        report.f1_score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"   Accuracy: {report.accuracy*100:.2f}%")
        print(f"   F1 Score: {report.f1_score*100:.2f}%")
        
        # 2. Error analysis
        print("\n[ERRORS] Analyzing errors...")
        error_results = self.error_analyzer.analyze_errors(
            model, model_name, test_loader, test_dataset
        )
        
        report.total_errors = error_results["misclassified"]
        patterns = error_results["error_patterns"]
        report.high_confidence_errors = patterns.get("error_confidence", {}).get("high_confidence_errors", 0)
        
        confusions = patterns.get("most_common_confusions", [])
        if confusions:
            top = confusions[0]
            report.most_confused_pair = f"{top['true']} → {top['predicted']}"
        
        print(f"   Total errors: {report.total_errors}")
        print(f"   High-confidence errors: {report.high_confidence_errors}")
        
        # 3. Inference benchmark
        print("\n[BENCH] Benchmarking inference time...")
        bench_results = self.inference_benchmark.benchmark_model(model, model_name)
        report.inference_time_ms = bench_results["mean_ms"]
        report.inference_fps = bench_results["fps"]
        
        # 4. Calibration analysis
        print("\n[CAL] Analyzing calibration...")
        cal_results = self.calibration_analyzer.compute_calibration(y_true, y_prob, model_name)
        report.ece = cal_results["ece"]
        report.mce = cal_results["mce"]
        print(f"   ECE: {report.ece:.4f}")
        
        # 5. Per-class analysis
        print("\n[CLASS] Per-class analysis...")
        class_results = self.class_analyzer.analyze_per_class(y_true, y_pred, y_prob, model_name)
        
        # Find best/worst classes
        f1_scores = {cls: data["f1"] for cls, data in class_results.items()}
        report.best_class = max(f1_scores, key=f1_scores.get)
        report.best_class_f1 = f1_scores[report.best_class]
        report.worst_class = min(f1_scores, key=f1_scores.get)
        report.worst_class_f1 = f1_scores[report.worst_class]
        
        print(f"   Best class: {report.best_class} (F1: {report.best_class_f1*100:.1f}%)")
        print(f"   Worst class: {report.worst_class} (F1: {report.worst_class_f1*100:.1f}%)")
        
        # 6. Cross-validation (optional)
        if run_cv:
            print("\n[CV] Running cross-validation...")
            cv_dataset = datasets.ImageFolder(
                str(TEST_DIR.parent),
                transform=get_val_transforms()
            )
            cv_results = self.cross_validator.cross_validate_model(
                model_name, cv_dataset, epochs=cv_epochs
            )
            
            stats = cv_results["statistics"]
            report.cv_accuracy_mean = stats["accuracy"]["mean"]
            report.cv_accuracy_std = stats["accuracy"]["std"]
            report.cv_f1_mean = stats["f1"]["mean"]
            report.cv_f1_std = stats["f1"]["std"]
        
        # Calculate overall score
        report.overall_score = self._calculate_overall_score(report)
        
        # Store report
        self.model_reports[model_name] = report
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return report
    
    def _calculate_overall_score(self, report: ModelReport) -> float:
        """Calculate weighted overall score."""
        weights = {
            "accuracy": 0.15,
            "precision": 0.15,
            "recall": 0.25,  # Higher for medical
            "f1_score": 0.30,
            "calibration": 0.10,
            "speed": 0.05
        }
        
        # Normalize calibration (lower is better)
        cal_score = max(0, 1 - report.ece * 10)
        
        # Normalize speed (faster is better, assume 200ms max)
        speed_score = max(0, 1 - report.inference_time_ms / 200)
        
        score = (
            weights["accuracy"] * report.accuracy +
            weights["precision"] * report.precision +
            weights["recall"] * report.recall +
            weights["f1_score"] * report.f1_score +
            weights["calibration"] * cal_score +
            weights["speed"] * speed_score
        )
        
        return score
    
    def analyze_all_models(
        self,
        model_names: Optional[List[str]] = None,
        run_cv: bool = False,
        cv_epochs: int = 5
    ) -> Dict[str, ModelReport]:
        """
        Analyze all models and generate comprehensive report.
        
        Args:
            model_names: List of models to analyze
            run_cv: Whether to run cross-validation
            cv_epochs: Epochs for CV
            
        Returns:
            Dictionary of ModelReports
        """
        if model_names is None:
            model_names = list(MODELS.keys())
        
        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL ANALYSIS")
        print("=" * 70)
        print(f"Models: {', '.join(model_names)}")
        print(f"Device: {self.device}")
        print(f"Cross-validation: {'Yes' if run_cv else 'No'}")
        
        # Analyze each model
        for model_name in model_names:
            self.analyze_model(model_name, run_cv=run_cv, cv_epochs=cv_epochs)
        
        # Load training histories for overlay charts
        self._load_training_histories(model_names)
        
        # Rank models
        self._rank_models()
        
        # Statistical comparison if CV was run
        if run_cv and len(self.model_reports) > 1:
            print("\n[STATS] Running statistical comparisons...")
            self.statistical_comparisons = self.stat_comparator.compare_all_models(
                self.cross_validator.results
            )
        
        return self.model_reports
    
    def _load_training_histories(self, model_names: List[str]) -> None:
        """Load training history JSON files for overlay charts."""
        for model_name in model_names:
            history_path = RESULTS_DIR / f"{model_name}_history.json"
            if history_path.exists():
                try:
                    with open(history_path, 'r') as f:
                        data = json.load(f)
                    if 'history' in data:
                        self.training_histories[model_name] = data['history']
                except Exception:
                    pass
    
    def _rank_models(self) -> None:
        """Rank models by overall score."""
        sorted_models = sorted(
            self.model_reports.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        for rank, (name, report) in enumerate(sorted_models, 1):
            report.rank = rank
            
            if rank == 1:
                report.tier = "[1st] TOP PERFORMER"
            elif rank <= 2:
                report.tier = "[2nd] HIGH PERFORMER"
            elif report.overall_score >= 0.85:
                report.tier = "[OK] RELIABLE"
            else:
                report.tier = "[!!] NEEDS REVIEW"
    
    # ==================== REPORT GENERATION ====================
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all models."""
        data = []
        
        for name, report in self.model_reports.items():
            data.append({
                "Rank": report.rank,
                "Model": name,
                "Accuracy": f"{report.accuracy*100:.2f}%",
                "F1 Score": f"{report.f1_score*100:.2f}%",
                "Inference (ms)": f"{report.inference_time_ms:.1f}",
                "ECE": f"{report.ece:.4f}",
                "Score": f"{report.overall_score:.3f}",
                "Tier": report.tier
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("Rank")
        
        return df
    
    def generate_full_report(self) -> str:
        """Generate comprehensive text report."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE MODEL PERFORMANCE REPORT")
        lines.append("Brain MRI Tumor Classification System")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        # Executive Summary
        lines.append("\n## EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        
        if self.model_reports:
            best = max(self.model_reports.values(), key=lambda x: x.overall_score)
            lines.append(f"Best Model: {best.model_name.upper()}")
            lines.append(f"  Accuracy: {best.accuracy*100:.2f}%")
            lines.append(f"  F1 Score: {best.f1_score*100:.2f}%")
            lines.append(f"  Inference: {best.inference_time_ms:.1f} ms")
        
        # Summary Table
        lines.append("\n## MODEL COMPARISON")
        lines.append("-" * 40)
        summary_df = self.generate_summary_table()
        lines.append(summary_df.to_string(index=False))
        
        # Detailed Reports
        lines.append("\n" + "=" * 80)
        lines.append("DETAILED MODEL REPORTS")
        lines.append("=" * 80)
        
        for name, report in sorted(self.model_reports.items(), key=lambda x: x[1].rank):
            lines.append(f"\n### {name.upper()} {report.tier}")
            lines.append("-" * 50)
            
            lines.append(f"\n📊 Core Metrics")
            lines.append(f"   Accuracy:  {report.accuracy*100:.2f}%")
            lines.append(f"   Precision: {report.precision*100:.2f}%")
            lines.append(f"   Recall:    {report.recall*100:.2f}%")
            lines.append(f"   F1 Score:  {report.f1_score*100:.2f}%")
            
            if report.cv_accuracy_mean > 0:
                lines.append(f"\n🔄 Cross-Validation (5-Fold)")
                lines.append(f"   Accuracy: {report.cv_accuracy_mean*100:.2f}% ± {report.cv_accuracy_std*100:.2f}%")
                lines.append(f"   F1 Score: {report.cv_f1_mean*100:.2f}% ± {report.cv_f1_std*100:.2f}%")
            
            lines.append(f"\n⏱️ Inference Performance")
            lines.append(f"   Time: {report.inference_time_ms:.2f} ms")
            lines.append(f"   Throughput: {report.inference_fps:.1f} FPS")
            
            lines.append(f"\n📐 Calibration")
            lines.append(f"   ECE: {report.ece:.4f}")
            lines.append(f"   MCE: {report.mce:.4f}")
            
            lines.append(f"\n🔍 Error Analysis")
            lines.append(f"   Total Errors: {report.total_errors}")
            lines.append(f"   High-Confidence Errors: {report.high_confidence_errors}")
            if report.most_confused_pair:
                lines.append(f"   Most Common Confusion: {report.most_confused_pair}")
            
            lines.append(f"\n📈 Per-Class Performance")
            lines.append(f"   Best:  {report.best_class} (F1: {report.best_class_f1*100:.1f}%)")
            lines.append(f"   Worst: {report.worst_class} (F1: {report.worst_class_f1*100:.1f}%)")
            
            lines.append(f"\n   Overall Score: {report.overall_score:.3f}")
        
        # Statistical Comparisons
        if self.statistical_comparisons is not None:
            lines.append("\n" + "=" * 80)
            lines.append("STATISTICAL SIGNIFICANCE")
            lines.append("=" * 80)
            lines.append("\nPaired t-test comparisons (α = 0.05):")
            lines.append(self.statistical_comparisons.to_string(index=False))
        
        # Recommendations
        lines.append("\n" + "=" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 80)
        
        if self.model_reports:
            best = max(self.model_reports.values(), key=lambda x: x.overall_score)
            fastest = min(self.model_reports.values(), key=lambda x: x.inference_time_ms)
            
            lines.append(f"\n🏆 For Best Accuracy: {best.model_name}")
            lines.append(f"   → Use when accuracy is the priority")
            
            lines.append(f"\n⚡ For Real-time Deployment: {fastest.model_name}")
            lines.append(f"   → {fastest.inference_time_ms:.1f}ms inference time")
            
            # Check for calibration issues
            poorly_calibrated = [
                r for r in self.model_reports.values() if r.ece > 0.1
            ]
            if poorly_calibrated:
                lines.append(f"\n⚠️ Calibration Warning:")
                for r in poorly_calibrated:
                    lines.append(f"   {r.model_name} has ECE={r.ece:.3f} - consider temperature scaling")
        
        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_report(self) -> Dict[str, Path]:
        """Save all reports and visualizations."""
        saved_files = {}
        
        # Text report
        report_text = self.generate_full_report()
        text_path = self.output_dir / "comprehensive_report.txt"
        with open(text_path, 'w') as f:
            f.write(report_text)
        saved_files["text_report"] = text_path
        print(f"Saved: {text_path}")
        
        # Summary CSV
        summary_df = self.generate_summary_table()
        csv_path = self.output_dir / "model_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        saved_files["summary_csv"] = csv_path
        
        # JSON data
        json_data = {
            name: asdict(report)
            for name, report in self.model_reports.items()
        }
        json_path = self.output_dir / "model_data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        saved_files["json_data"] = json_path
        
        # Visualizations
        self._generate_visualizations()
        saved_files["visualizations"] = self.output_dir
        
        print(f"\n[DONE] All reports saved to: {self.output_dir}")
        
        return saved_files
    
    def _generate_visualizations(self) -> None:
        """Generate comparison visualizations."""
        if not self.model_reports:
            return
        
        # --- Original charts ---
        
        # 1. Overall comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(self.model_reports.keys())
        
        # Accuracy comparison
        ax = axes[0, 0]
        accuracies = [self.model_reports[m].accuracy * 100 for m in models]
        bars = ax.bar(models, accuracies, color='steelblue')
        ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # F1 Score comparison
        ax = axes[0, 1]
        f1_scores = [self.model_reports[m].f1_score * 100 for m in models]
        bars = ax.bar(models, f1_scores, color='forestgreen')
        ax.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score (%)')
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Inference time comparison
        ax = axes[1, 0]
        times = [self.model_reports[m].inference_time_ms for m in models]
        bars = ax.bar(models, times, color='coral')
        ax.set_title('Inference Time Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)')
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)
        
        # Overall score comparison
        ax = axes[1, 1]
        scores = [self.model_reports[m].overall_score for m in models]
        colors = ['gold' if s == max(scores) else 'steelblue' for s in scores]
        bars = ax.bar(models, scores, color=colors)
        ax.set_title('Overall Score Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Calibration diagrams
        self.calibration_analyzer.plot_all_models(
            save_path=self.output_dir / "calibration_diagrams.png"
        )
        
        # 3. Per-class F1 comparison
        self.class_analyzer.plot_class_comparison(
            metric="f1",
            save_path=self.output_dir / "class_f1_comparison.png"
        )
        
        print("Generated original visualizations")
        
        # --- New advanced charts (8 additional) ---
        try:
            generate_all_charts(
                model_reports=self.model_reports,
                predictions_dict=self.predictions_dict,
                histories=self.training_histories,
                output_dir=self.output_dir,
            )
        except Exception as e:
            print(f"[WARN] Some advanced charts could not be generated: {e}")


# ==================== MAIN ====================




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Model Analysis")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to analyze")
    parser.add_argument("--cv", action="store_true",
                        help="Run cross-validation")
    parser.add_argument("--cv-epochs", type=int, default=5,
                        help="CV epochs per fold")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ComprehensiveReportAgent()
    
    # Run analysis
    agent.analyze_all_models(
        model_names=args.models,
        run_cv=args.cv,
        cv_epochs=args.cv_epochs
    )
    
    # Generate report
    report = agent.generate_full_report()
    print("\n" + report)
    
    # Save everything
    agent.save_report()
