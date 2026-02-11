"""
Configuration module for Brain MRI Tumor Classification System.
Centralized settings for paths, hyperparameters, and model configurations.
"""

import os
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent.resolve()

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DATASET_DIR = BASE_DIR / "processed_dataset"

# Split directories
TRAIN_DIR = PROCESSED_DATASET_DIR / "train"
VAL_DIR = PROCESSED_DATASET_DIR / "val"
TEST_DIR = PROCESSED_DATASET_DIR / "test"

# Output paths
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
PERFORMANCE_PLOTS_DIR = RESULTS_DIR / "performance_plots"
GRADCAM_DIR = RESULTS_DIR / "gradcam"

# ==================== CLASSES ====================
# 3-class tumor-type classification (no normal class)
# "The study focuses on multi-class classification of brain tumor types
#  using MRI images, assuming pre-identified abnormal scans."
CLASSES = ["glioma", "meningioma", "pituitary"]
NUM_CLASSES = len(CLASSES)  # 3 output neurons

# ==================== IMAGE SETTINGS ====================
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# ==================== DATA SPLIT ====================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ==================== TRAINING HYPERPARAMETERS ====================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Early stopping
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 0.001

# Learning rate scheduler
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# ==================== MODEL CONFIGURATIONS ====================
MODELS = {
    "resnet50": {
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_layers": 2,  # Last N blocks to unfreeze
    },
    "resnet101": {
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_layers": 2,
    },
    "densenet121": {
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_layers": 1,  # Last dense block
    },
    "efficientnet_b4": {
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_layers": 3,
    },
    "mobilenetv2": {
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_layers": 3,
    },
}

# ==================== NORMALIZATION ====================
# ImageNet mean and std for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==================== DATA AUGMENTATION ====================
AUGMENTATION_CONFIG = {
    "rotation_limit": 20,
    "horizontal_flip_prob": 0.5,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "zoom_limit": 0.15,
}

# ==================== AGENT CONFIGURATIONS ====================
AGENT_CONFIG = {
    "performance_weights": {
        "f1_score": 0.40,
        "recall": 0.30,
        "precision": 0.20,
        "accuracy": 0.10,
    },
    "risk_thresholds": {
        "high_confidence": 0.85,
        "moderate_confidence": 0.65,
        "low_confidence": 0.45,
    },
    "consensus_min_agreement": 3,  # Minimum models agreeing
}

# ==================== GRAD-CAM SETTINGS ====================
GRADCAM_TARGET_LAYERS = {
    "resnet50": "layer4",
    "resnet101": "layer4",
    "densenet121": "features.denseblock4",
    "efficientnet_b4": "features",
    "mobilenetv2": "features",
}

# ==================== DEVICE CONFIGURATION ====================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== UTILITY FUNCTIONS ====================
def create_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATASET_DIR,
        PROCESSED_DATASET_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        CONFUSION_MATRICES_DIR,
        PERFORMANCE_PLOTS_DIR,
        GRADCAM_DIR,
    ]
    for class_name in CLASSES:
        directories.extend([
            TRAIN_DIR / class_name,
            VAL_DIR / class_name,
            TEST_DIR / class_name,
        ])
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_path(model_name: str) -> Path:
    """Get the path for saving/loading a model."""
    return MODELS_DIR / f"{model_name}_brain_mri.pt"

def get_metrics_path(model_name: str) -> Path:
    """Get the path for saving model metrics."""
    return RESULTS_DIR / f"{model_name}_metrics.json"


if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Classes: {CLASSES}")
    create_directories()
    print("All directories created successfully!")
