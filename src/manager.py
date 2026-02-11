"""
Training Manager for Brain MRI Tumor Classification.
Orchestrates the complete training pipeline with resource-aware optimization.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, RESULTS_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MODELS, CLASSES, DEVICE,
    get_model_path
)
from resource_manager import ResourceManager, MemoryMonitor


# ==================== ENUMS AND DATA CLASSES ====================

class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingPriority(Enum):
    """Training priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3


@dataclass
class TrainingJob:
    """Training job configuration."""
    model_name: str
    epochs: int = EPOCHS
    batch_size: Optional[int] = None  # Auto-determined if None
    learning_rate: float = LEARNING_RATE
    priority: TrainingPriority = TrainingPriority.NORMAL
    
    # Runtime info
    status: TrainingStatus = TrainingStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_epoch: int = 0
    best_val_acc: float = 0.0
    error_message: Optional[str] = None
    
    # Callbacks
    on_epoch_end: Optional[Callable] = None
    on_complete: Optional[Callable] = None


@dataclass
class TrainingConfig:
    """Global training configuration."""
    models_to_train: List[str] = field(default_factory=lambda: list(MODELS.keys()))
    epochs: int = EPOCHS
    batch_size: Optional[int] = None  # Auto-determined per model
    learning_rate: float = LEARNING_RATE
    auto_optimize_resources: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 5  # Save every N epochs
    early_stopping: bool = True
    early_stopping_patience: int = 7
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_concurrent_jobs: int = 1  # Sequential by default


# ==================== TRAINING MANAGER ====================

class TrainingManager:
    """
    Orchestrates the complete training pipeline.
    
    Features:
    - Resource-aware training
    - Automatic batch size optimization
    - Multi-model training orchestration
    - Checkpoint management
    - Progress tracking and logging
    - Error recovery
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        """
        Initialize the training manager.
        
        Args:
            config: Training configuration
            resource_manager: Resource manager instance
        """
        self.config = config or TrainingConfig()
        self.rm = resource_manager or ResourceManager()
        
        # Job queue
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_order: List[str] = []
        
        # State
        self.is_running = False
        self.current_job: Optional[TrainingJob] = None
        
        # Logging
        self._setup_logging()
        
        # Memory monitor
        self.memory_monitor = MemoryMonitor(self.rm)
    
    def _setup_logging(self) -> None:
        """Set up logging."""
        log_dir = RESULTS_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    # ==================== JOB MANAGEMENT ====================
    
    def add_job(
        self,
        model_name: str,
        priority: TrainingPriority = TrainingPriority.NORMAL,
        **kwargs
    ) -> TrainingJob:
        """
        Add a training job to the queue.
        
        Args:
            model_name: Name of the model to train
            priority: Job priority
            **kwargs: Additional job configuration
            
        Returns:
            Created TrainingJob
        """
        job = TrainingJob(
            model_name=model_name,
            priority=priority,
            epochs=kwargs.get('epochs', self.config.epochs),
            batch_size=kwargs.get('batch_size', self.config.batch_size),
            learning_rate=kwargs.get('learning_rate', self.config.learning_rate),
        )
        
        self.jobs[model_name] = job
        self.job_order.append(model_name)
        
        # Sort by priority
        self.job_order.sort(
            key=lambda x: self.jobs[x].priority.value,
            reverse=True
        )
        
        self.logger.info(f"Added training job: {model_name} (priority: {priority.name})")
        
        return job
    
    def add_all_models(self) -> List[TrainingJob]:
        """Add all configured models to the queue."""
        jobs = []
        for model_name in self.config.models_to_train:
            job = self.add_job(model_name)
            jobs.append(job)
        return jobs
    
    def remove_job(self, model_name: str) -> bool:
        """Remove a job from the queue."""
        if model_name in self.jobs:
            del self.jobs[model_name]
            self.job_order.remove(model_name)
            self.logger.info(f"Removed training job: {model_name}")
            return True
        return False
    
    def get_job_status(self, model_name: str) -> Optional[TrainingJob]:
        """Get the status of a specific job."""
        return self.jobs.get(model_name)
    
    def get_queue_status(self) -> Dict:
        """Get the status of all jobs in the queue."""
        return {
            "total_jobs": len(self.jobs),
            "pending": sum(1 for j in self.jobs.values() if j.status == TrainingStatus.PENDING),
            "running": sum(1 for j in self.jobs.values() if j.status == TrainingStatus.RUNNING),
            "completed": sum(1 for j in self.jobs.values() if j.status == TrainingStatus.COMPLETED),
            "failed": sum(1 for j in self.jobs.values() if j.status == TrainingStatus.FAILED),
            "jobs": {name: job.status.value for name, job in self.jobs.items()}
        }
    
    # ==================== RESOURCE OPTIMIZATION ====================
    
    def optimize_job(self, job: TrainingJob) -> TrainingJob:
        """
        Optimize job parameters based on available resources.
        
        Args:
            job: Training job to optimize
            
        Returns:
            Optimized job
        """
        if not self.config.auto_optimize_resources:
            return job
        
        # Auto-determine batch size
        if job.batch_size is None:
            job.batch_size = self.rm.recommend_batch_size(job.model_name)
            self.logger.info(f"Auto-selected batch size: {job.batch_size} for {job.model_name}")
        
        return job
    
    def check_resources(self) -> Tuple[bool, str]:
        """
        Check if resources are sufficient for training.
        
        Returns:
            Tuple of (is_sufficient, message)
        """
        # Check GPU memory
        if self.rm.device.type == 'cuda':
            available_mb = self.rm.get_available_memory()
            if available_mb < 2000:  # Less than 2GB
                return False, f"Insufficient GPU memory: {available_mb:.0f} MB available"
        
        # Check RAM
        ram = self.rm.get_memory_usage()
        if ram['ram_percent'] > 90:
            return False, f"System RAM usage too high: {ram['ram_percent']:.1f}%"
        
        return True, "Resources OK"
    
    # ==================== TRAINING EXECUTION ====================
    
    def _create_data_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training."""
        from train_models import create_dataloaders
        train_loader, val_loader, _ = create_dataloaders(batch_size)
        return train_loader, val_loader
    
    def _train_single_model(self, job: TrainingJob) -> bool:
        """
        Train a single model.
        
        Args:
            job: Training job
            
        Returns:
            True if successful, False otherwise
        """
        from train_models import train_model
        
        try:
            # Optimize job
            job = self.optimize_job(job)
            
            # Update status
            job.status = TrainingStatus.RUNNING
            job.start_time = time.time()
            self.current_job = job
            
            self.logger.info(f"Starting training: {job.model_name}")
            self.logger.info(f"  Batch size: {job.batch_size}")
            self.logger.info(f"  Epochs: {job.epochs}")
            self.logger.info(f"  Learning rate: {job.learning_rate}")
            
            # Start memory monitor
            self.memory_monitor.start()
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(job.batch_size)
            
            # Train model
            with self.rm.training_context():
                results = train_model(
                    model_name=job.model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=job.epochs,
                    learning_rate=job.learning_rate,
                    device=self.rm.device
                )
            
            # Update job with results
            job.best_val_acc = results.get('best_val_acc', 0)
            job.current_epoch = results.get('epochs_trained', job.epochs)
            job.status = TrainingStatus.COMPLETED
            job.end_time = time.time()
            
            # Log memory summary
            self.memory_monitor.print_summary()
            
            # Call completion callback
            if job.on_complete:
                job.on_complete(job, results)
            
            self.logger.info(f"Completed training: {job.model_name} (Best Val Acc: {job.best_val_acc:.2f}%)")
            
            return True
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.end_time = time.time()
            
            self.logger.error(f"Training failed for {job.model_name}: {e}")
            
            # Clear memory on failure
            self.rm.clear_memory(aggressive=True)
            
            return False
        
        finally:
            self.current_job = None
    
    def run(self, models: Optional[List[str]] = None) -> Dict:
        """
        Run the training pipeline.
        
        Args:
            models: Optional list of specific models to train
            
        Returns:
            Dictionary with training results
        """
        self.is_running = True
        
        # Add specified models or all models
        if models:
            for model in models:
                if model not in self.jobs:
                    self.add_job(model)
        elif not self.jobs:
            self.add_all_models()
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING MANAGER - STARTING PIPELINE")
        self.logger.info("=" * 60)
        
        # Print system info
        self.rm.print_system_info()
        
        # Check resources
        is_ok, message = self.check_resources()
        if not is_ok:
            self.logger.error(f"Resource check failed: {message}")
            self.is_running = False
            return {"error": message}
        
        results = {
            "start_time": datetime.now().isoformat(),
            "device": str(self.rm.device),
            "models": {}
        }
        
        # Process jobs in order
        for model_name in self.job_order:
            job = self.jobs[model_name]
            
            if job.status != TrainingStatus.PENDING:
                continue
            
            # Check resources before each job
            is_ok, message = self.check_resources()
            if not is_ok:
                self.logger.warning(f"Resource issue: {message}")
                self.rm.clear_memory(aggressive=True)
                time.sleep(5)  # Wait for cleanup
            
            # Train model
            success = self._train_single_model(job)
            
            results["models"][model_name] = {
                "status": job.status.value,
                "best_val_acc": job.best_val_acc,
                "epochs_trained": job.current_epoch,
                "training_time": job.end_time - job.start_time if job.end_time else 0,
                "error": job.error_message
            }
            
            # Clean up between models
            self.rm.clear_memory()
        
        results["end_time"] = datetime.now().isoformat()
        results["summary"] = self.get_queue_status()
        
        # Save results
        results_path = RESULTS_DIR / "training_manager_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.is_running = False
        self.logger.info("=" * 60)
        self.logger.info("TRAINING PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        
        return results
    
    def stop(self) -> None:
        """Stop the training pipeline."""
        self.logger.info("Stopping training pipeline...")
        self.is_running = False
        
        if self.current_job:
            self.current_job.status = TrainingStatus.STOPPED
    
    # ==================== REPORTING ====================
    
    def print_status(self) -> None:
        """Print current pipeline status."""
        status = self.get_queue_status()
        
        print("\n" + "=" * 60)
        print("TRAINING MANAGER STATUS")
        print("=" * 60)
        
        print(f"\n📊 Queue Overview")
        print(f"   Total Jobs: {status['total_jobs']}")
        print(f"   Pending: {status['pending']}")
        print(f"   Running: {status['running']}")
        print(f"   Completed: {status['completed']}")
        print(f"   Failed: {status['failed']}")
        
        print(f"\n📋 Job Details")
        for name, job in self.jobs.items():
            status_emoji = {
                TrainingStatus.PENDING: "⏳",
                TrainingStatus.RUNNING: "🔄",
                TrainingStatus.COMPLETED: "✅",
                TrainingStatus.FAILED: "❌",
                TrainingStatus.STOPPED: "⏹️"
            }
            emoji = status_emoji.get(job.status, "❓")
            
            print(f"   {emoji} {name}: {job.status.value}")
            if job.status == TrainingStatus.COMPLETED:
                print(f"      Best Val Acc: {job.best_val_acc:.2f}%")
            elif job.status == TrainingStatus.FAILED:
                print(f"      Error: {job.error_message}")
        
        print("=" * 60)
    
    def generate_report(self) -> str:
        """Generate a summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append("TRAINING MANAGER REPORT")
        lines.append("=" * 60)
        lines.append(f"\nDevice: {self.rm.device}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        
        lines.append("\n\n## Model Training Results\n")
        
        for name, job in self.jobs.items():
            lines.append(f"### {name}")
            lines.append(f"- Status: {job.status.value}")
            lines.append(f"- Epochs: {job.current_epoch}/{job.epochs}")
            
            if job.best_val_acc > 0:
                lines.append(f"- Best Val Accuracy: {job.best_val_acc:.2f}%")
            
            if job.start_time and job.end_time:
                duration = (job.end_time - job.start_time) / 60
                lines.append(f"- Training Time: {duration:.1f} minutes")
            
            if job.error_message:
                lines.append(f"- Error: {job.error_message}")
            
            lines.append("")
        
        return "\n".join(lines)


# ==================== EVALUATION MANAGER ====================

class EvaluationManager:
    """Manages model evaluation after training."""
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.rm = resource_manager or ResourceManager()
        self.logger = logging.getLogger(__name__)
    
    def evaluate_all_models(self, model_names: Optional[List[str]] = None) -> Dict:
        """
        Evaluate all trained models.
        
        Args:
            model_names: Optional list of models to evaluate
            
        Returns:
            Evaluation results
        """
        from evaluate_models import evaluate_all_models
        
        self.logger.info("Starting model evaluation...")
        
        with self.rm.inference_context():
            results = evaluate_all_models(model_names)
        
        self.logger.info("Evaluation complete")
        return results


# ==================== PIPELINE MANAGER ====================

class PipelineManager:
    """
    High-level manager for the complete ML pipeline.
    
    Orchestrates:
    - Data preprocessing
    - Dataset splitting
    - Model training
    - Model evaluation
    - Inference preparation
    """
    
    def __init__(self):
        self.rm = ResourceManager()
        self.training_manager = TrainingManager(resource_manager=self.rm)
        self.evaluation_manager = EvaluationManager(resource_manager=self.rm)
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(
        self,
        skip_preprocessing: bool = False,
        skip_splitting: bool = False,
        models: Optional[List[str]] = None
    ) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            skip_preprocessing: Skip data preprocessing
            skip_splitting: Skip dataset splitting
            models: Optional list of models to train
            
        Returns:
            Pipeline results
        """
        results = {
            "start_time": datetime.now().isoformat(),
            "steps": {}
        }
        
        self.logger.info("=" * 60)
        self.logger.info("FULL PIPELINE EXECUTION")
        self.logger.info("=" * 60)
        
        # Step 1: Preprocessing
        if not skip_preprocessing:
            self.logger.info("\n📷 Step 1: Preprocessing...")
            try:
                from preprocessing import preprocess_dataset
                preprocess_stats = preprocess_dataset()
                results["steps"]["preprocessing"] = {
                    "status": "completed",
                    "stats": preprocess_stats
                }
            except Exception as e:
                results["steps"]["preprocessing"] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.logger.error(f"Preprocessing failed: {e}")
        else:
            results["steps"]["preprocessing"] = {"status": "skipped"}
        
        # Step 2: Dataset Splitting
        if not skip_splitting:
            self.logger.info("\n📂 Step 2: Dataset Splitting...")
            try:
                from split_dataset import split_dataset
                split_stats = split_dataset()
                results["steps"]["splitting"] = {
                    "status": "completed",
                    "stats": split_stats
                }
            except Exception as e:
                results["steps"]["splitting"] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.logger.error(f"Splitting failed: {e}")
        else:
            results["steps"]["splitting"] = {"status": "skipped"}
        
        # Step 3: Training
        self.logger.info("\n🏋️ Step 3: Model Training...")
        training_results = self.training_manager.run(models)
        results["steps"]["training"] = training_results
        
        # Step 4: Evaluation
        self.logger.info("\n📊 Step 4: Model Evaluation...")
        eval_results = self.evaluation_manager.evaluate_all_models(models)
        results["steps"]["evaluation"] = {
            "status": "completed" if eval_results else "failed",
            "models": list(eval_results.keys()) if eval_results else []
        }
        
        results["end_time"] = datetime.now().isoformat()
        
        # Save pipeline results
        results_path = RESULTS_DIR / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        
        return results
    
    def get_status(self) -> Dict:
        """Get pipeline status."""
        return {
            "resource_manager": {
                "device": str(self.rm.device),
                "memory": self.rm.get_memory_usage()
            },
            "training_manager": self.training_manager.get_queue_status()
        }


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Manager CLI")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (auto if not specified)")
    parser.add_argument("--status", action="store_true",
                        help="Show system status only")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run full pipeline including preprocessing")
    
    args = parser.parse_args()
    
    if args.status:
        rm = ResourceManager()
        rm.print_system_info()
    elif args.full_pipeline:
        pm = PipelineManager()
        pm.run_full_pipeline(models=args.models)
    else:
        # Training only
        config = TrainingConfig(
            models_to_train=args.models or list(MODELS.keys()),
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        tm = TrainingManager(config)
        tm.run()
        tm.print_status()
