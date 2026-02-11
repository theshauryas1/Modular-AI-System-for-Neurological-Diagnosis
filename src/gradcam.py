"""
Grad-CAM Explainability Module for Brain MRI Tumor Classification.
Visualizes which regions of MRI images influence model predictions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, RESULTS_DIR, GRADCAM_DIR, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, CLASSES, DEVICE,
    GRADCAM_TARGET_LAYERS, get_model_path
)
from train_models import get_model, get_val_transforms


# ==================== GRAD-CAM IMPLEMENTATION ====================

class GradCAM:
    """
    Grad-CAM implementation for visualizing CNN activations.
    
    Grad-CAM uses the gradients of any target concept flowing into the
    final convolutional layer to produce a coarse localization map
    highlighting important regions in the image.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The CNN model
            target_layer: The target convolutional layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Preprocessed input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bilinear',
            align_corners=False
        )
        
        heatmap = cam.squeeze().cpu().numpy()
        
        return heatmap, target_class, confidence


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation for improved visualization.
    
    Grad-CAM++ provides better localization of multiple occurrences
    of objects in an image compared to Grad-CAM.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Grad-CAM++ weighting
        grads = self.gradients
        activations = self.activations
        
        # Second and third order derivatives (approximated)
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + 1e-8)
        
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bilinear',
            align_corners=False
        )
        
        heatmap = cam.squeeze().cpu().numpy()
        
        return heatmap, target_class, confidence


# ==================== VISUALIZATION HELPERS ====================

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM visualization.
    
    Args:
        model: The CNN model
        model_name: Name of the model
        
    Returns:
        Target convolutional layer
    """
    if model_name in ["resnet50", "resnet101"]:
        return model.layer4[-1]
    elif model_name == "densenet121":
        return model.features.denseblock4
    elif model_name == "efficientnet_b4":
        # For timm models
        return model.conv_head
    elif model_name == "mobilenetv2":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_and_preprocess_image(
    image_path: str,
    device: torch.device = DEVICE
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess an image for Grad-CAM.
    
    Args:
        image_path: Path to the image
        device: Device to load tensor on
        
    Returns:
        Tuple of (preprocessed_tensor, original_image)
    """
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    
    # Preprocess
    transform = get_val_transforms()
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    return input_tensor, original_array


def apply_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Apply heatmap overlay on the original image.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Overlay transparency
        colormap: OpenCV colormap to use
        
    Returns:
        Image with heatmap overlay
    """
    # Convert heatmap to colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = np.float32(heatmap_colored) * alpha + np.float32(image) * (1 - alpha)
    overlaid = np.uint8(np.clip(overlaid, 0, 255))
    
    return overlaid


# ==================== VISUALIZATION FUNCTIONS ====================

def visualize_gradcam(
    image_path: str,
    model: nn.Module,
    model_name: str,
    target_class: Optional[int] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    use_plus_plus: bool = False,
    device: torch.device = DEVICE
) -> Dict:
    """
    Generate and visualize Grad-CAM for a single image.
    
    Args:
        image_path: Path to the input image
        model: Trained model
        model_name: Name of the model
        target_class: Target class (None = use predicted)
        save_path: Path to save visualization
        show: Whether to display the plot
        use_plus_plus: Whether to use Grad-CAM++
        device: Device for inference
        
    Returns:
        Dictionary with visualization results
    """
    # Load and preprocess image
    input_tensor, original_image = load_and_preprocess_image(image_path, device)
    
    # Get target layer
    target_layer = get_target_layer(model, model_name)
    
    # Create Grad-CAM
    if use_plus_plus:
        gradcam = GradCAMPlusPlus(model, target_layer)
    else:
        gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap, pred_class, confidence = gradcam(input_tensor, target_class)
    
    # Apply heatmap overlay
    overlaid = apply_heatmap(original_image, heatmap)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlaid)
    axes[2].set_title(
        f'Prediction: {CLASSES[pred_class]} ({confidence*100:.1f}%)',
        fontsize=12
    )
    axes[2].axis('off')
    
    plt.suptitle(
        f'{model_name} - Grad-CAM{"++*" if use_plus_plus else ""} Visualization',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        "image_path": image_path,
        "model_name": model_name,
        "predicted_class": CLASSES[pred_class],
        "predicted_class_idx": pred_class,
        "confidence": confidence,
        "heatmap": heatmap,
        "overlaid_image": overlaid
    }


def visualize_all_models(
    image_path: str,
    model_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    device: torch.device = DEVICE
) -> Dict:
    """
    Generate Grad-CAM visualizations for all models on a single image.
    
    Args:
        image_path: Path to the input image
        model_names: List of model names (None = all)
        save_path: Path to save combined visualization
        device: Device for inference
        
    Returns:
        Dictionary with results from all models
    """
    from train_models import get_model
    
    if model_names is None:
        model_names = list(GRADCAM_TARGET_LAYERS.keys())
    
    results = {}
    heatmaps = []
    overlays = []
    predictions = []
    
    # Load original image
    input_tensor, original_image = load_and_preprocess_image(image_path, device)
    
    for model_name in model_names:
        print(f"Processing {model_name}...")
        
        # Load model
        model_path = get_model_path(model_name)
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            continue
        
        model = get_model(model_name)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Get target layer and generate Grad-CAM
        target_layer = get_target_layer(model, model_name)
        gradcam = GradCAM(model, target_layer)
        heatmap, pred_class, confidence = gradcam(input_tensor)
        
        # Store results
        overlaid = apply_heatmap(original_image, heatmap)
        heatmaps.append(heatmap)
        overlays.append(overlaid)
        predictions.append((model_name, CLASSES[pred_class], confidence))
        
        results[model_name] = {
            "predicted_class": CLASSES[pred_class],
            "confidence": confidence
        }
    
    # Create combined visualization
    n_models = len(overlays)
    if n_models > 0:
        fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))
        
        # Original image (span both rows in first column)
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original', fontsize=12)
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Model visualizations
        for i, (model_name, pred_class, conf) in enumerate(predictions):
            # Heatmap
            axes[0, i + 1].imshow(heatmaps[i], cmap='jet')
            axes[0, i + 1].set_title(f'{model_name}\nHeatmap', fontsize=10)
            axes[0, i + 1].axis('off')
            
            # Overlay
            axes[1, i + 1].imshow(overlays[i])
            axes[1, i + 1].set_title(f'{pred_class}\n({conf*100:.1f}%)', fontsize=10)
            axes[1, i + 1].axis('off')
        
        plt.suptitle('Grad-CAM Comparison Across Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    return results


def generate_gradcam_samples(
    sample_dir: Path,
    n_samples: int = 3,
    output_dir: Path = GRADCAM_DIR,
    device: torch.device = DEVICE
) -> None:
    """
    Generate Grad-CAM visualizations for sample images from each class.
    
    Args:
        sample_dir: Directory containing class subfolders with images
        n_samples: Number of samples per class
        output_dir: Directory to save visualizations
        device: Device for inference
    """
    from train_models import get_model
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING GRAD-CAM SAMPLE VISUALIZATIONS")
    print("=" * 60)
    
    for class_name in CLASSES:
        class_dir = sample_dir / class_name
        if not class_dir.exists():
            print(f"Class directory not found: {class_dir}")
            continue
        
        # Get sample images
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        image_files = image_files[:n_samples]
        
        for i, image_path in enumerate(image_files):
            print(f"\nProcessing {class_name} sample {i+1}/{len(image_files)}")
            
            save_path = output_dir / f"{class_name}_sample_{i+1}_gradcam.png"
            visualize_all_models(str(image_path), save_path=save_path, device=device)
    
    print(f"\n[OK] Grad-CAM visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image to visualize")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: all models)")
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="Directory with sample images for batch processing")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of samples per class (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for visualization")
    
    args = parser.parse_args()
    
    if args.image:
        # Single image visualization
        output_path = args.output or GRADCAM_DIR / "gradcam_output.png"
        
        if args.model:
            # Single model
            from train_models import get_model
            model_path = get_model_path(args.model)
            if model_path.exists():
                model = get_model(args.model)
                checkpoint = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(DEVICE)
                model.eval()
                
                visualize_gradcam(
                    args.image, model, args.model,
                    save_path=Path(output_path), show=True
                )
            else:
                print(f"Model not found: {model_path}")
        else:
            # All models
            visualize_all_models(args.image, save_path=Path(output_path))
    
    elif args.samples_dir:
        # Batch processing
        from config import TEST_DIR
        samples_dir = Path(args.samples_dir) if args.samples_dir else TEST_DIR
        generate_gradcam_samples(samples_dir, n_samples=args.n_samples)
    
    else:
        print("Please specify --image or --samples-dir")
        print("Run with --help for more options")
