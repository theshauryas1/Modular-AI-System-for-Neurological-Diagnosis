"""
Model Training Module for Brain MRI Tumor Classification.
Trains multiple CNN architectures with transfer learning on the same dataset.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR, RESULTS_DIR,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR,
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, CLASSES,
    AUGMENTATION_CONFIG, MODELS, DEVICE, get_model_path
)


# ==================== DATA TRANSFORMS ====================

def get_train_transforms():
    """Get training data transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(AUGMENTATION_CONFIG["rotation_limit"]),
        transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG["horizontal_flip_prob"]),
        transforms.ColorJitter(
            brightness=AUGMENTATION_CONFIG["brightness_limit"],
            contrast=AUGMENTATION_CONFIG["contrast_limit"]
        ),
        transforms.RandomAffine(
            degrees=0,
            scale=(1 - AUGMENTATION_CONFIG["zoom_limit"], 1 + AUGMENTATION_CONFIG["zoom_limit"])
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """Get validation/test data transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# ==================== DATA LOADERS ====================

def create_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=get_train_transforms()
    )
    
    val_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=get_val_transforms()
    )
    
    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader


# ==================== MODEL BUILDERS ====================

def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build ResNet50 model with custom classifier."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last 2 layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.layer3[-1].parameters():
        param.requires_grad = True
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def build_resnet101(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build ResNet101 model with custom classifier."""
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last 2 layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.layer3[-1].parameters():
        param.requires_grad = True
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def build_densenet121(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build DenseNet121 model with custom classifier."""
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last dense block
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    
    # Replace classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def build_efficientnet_b4(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B4 model with custom classifier."""
    model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
    
    # Freeze backbone except last 3 blocks
    for name, param in model.named_parameters():
        if 'blocks.6' not in name and 'blocks.5' not in name and 'blocks.4' not in name:
            if 'classifier' not in name:
                param.requires_grad = False
    
    return model


def build_mobilenetv2(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV2 model with custom classifier."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last 3 blocks
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def get_model(model_name: str) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model to build
        
    Returns:
        PyTorch model
    """
    model_builders = {
        "resnet50": build_resnet50,
        "resnet101": build_resnet101,
        "densenet121": build_densenet121,
        "efficientnet_b4": build_efficientnet_b4,
        "mobilenetv2": build_mobilenetv2,
    }
    
    if model_name not in model_builders:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_builders.keys())}")
    
    return model_builders[model_name]()


# ==================== EARLY STOPPING ====================

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_score: Current validation score (higher is better)
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop
    
    def restore(self, model: nn.Module):
        """Restore best weights to model."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: torch.device = DEVICE
) -> Dict:
    """
    Train a single model.
    
    Args:
        model_name: Name of the model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Dictionary with training history and final metrics
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Build model
    model = get_model(model_name)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    
    # Early stopping
    early_stopping = EarlyStopping()
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = get_model_path(model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': CLASSES,
            }, model_path)
            print(f"[SAVED] Best model (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        if early_stopping(val_acc, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            early_stopping.restore(model)
            break
    
    training_time = time.time() - start_time
    
    # Save final results
    results = {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "epochs_trained": len(history["train_loss"]),
        "training_time_seconds": training_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": history
    }
    
    # Save history
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[DONE] Training complete in {training_time/60:.1f} minutes")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    
    return results


def train_all_models(
    model_names: Optional[List[str]] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Train all specified models.
    
    Args:
        model_names: List of model names to train. If None, trains all models.
        epochs: Number of epochs per model
        batch_size: Batch size for training
        
    Returns:
        Dictionary with results for all models
    """
    if model_names is None:
        model_names = list(MODELS.keys())
    
    print("=" * 60)
    print("BRAIN MRI TUMOR CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Models to train: {model_names}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size)
    
    all_results = {}
    
    for model_name in model_names:
        try:
            results = train_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"\n[ERROR] Error training {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"  {model_name}: FAILED - {results['error']}")
        else:
            print(f"  {model_name}: {results['best_val_acc']:.2f}% (Val Acc)")
    
    # Save summary
    summary_path = RESULTS_DIR / "training_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "models": {
            name: {
                "best_val_acc": r.get("best_val_acc"),
                "epochs_trained": r.get("epochs_trained"),
                "training_time": r.get("training_time_seconds")
            } if "error" not in r else {"error": r["error"]}
            for name, r in all_results.items()
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Brain MRI Classification Models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train (default: all)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    
    args = parser.parse_args()
    
    train_all_models(
        model_names=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
