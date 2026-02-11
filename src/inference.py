"""
Inference Pipeline for Brain MRI Tumor Classification.
End-to-end pipeline integrating all models and AI agents.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, RESULTS_DIR, GRADCAM_DIR,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CLASSES, DEVICE, MODELS, get_model_path
)
from train_models import get_model, get_val_transforms
from agents import (
    ModelPrediction, AgentPipeline,
    ModelPerformanceAgent, PredictionConsensusAgent, RiskAssessmentAgent
)
from gradcam import visualize_gradcam, visualize_all_models, get_target_layer


# ==================== INFERENCE ENGINE ====================

class BrainMRIClassifier:
    """
    End-to-end Brain MRI Tumor Classifier.
    
    Integrates multiple CNN models with AI agents for:
    - Multi-model inference
    - Consensus prediction
    - Risk assessment
    - Explainability (Grad-CAM)
    """
    
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        device: torch.device = DEVICE,
        load_models: bool = True
    ):
        """
        Initialize the classifier.
        
        Args:
            model_names: List of model names to use (default: all)
            device: Device for inference
            load_models: Whether to load models on initialization
        """
        self.device = device
        self.model_names = model_names or list(MODELS.keys())
        self.models: Dict[str, torch.nn.Module] = {}
        
        # Initialize transform
        self.transform = get_val_transforms()
        
        # Initialize AI agents pipeline
        self.agent_pipeline = AgentPipeline()
        
        # Load models if requested
        if load_models:
            self.load_all_models()
    
    def load_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """
        Load a single trained model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model or None if not found
        """
        model_path = get_model_path(model_name)
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return None
        
        try:
            model = get_model(model_name)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None
    
    def load_all_models(self) -> None:
        """Load all specified models."""
        print("Loading models...")
        for model_name in self.model_names:
            model = self.load_model(model_name)
            if model is not None:
                self.models[model_name] = model
                print(f"  [OK] Loaded {model_name}")
        
        print(f"\nLoaded {len(self.models)}/{len(self.model_names)} models")
        
        # Initialize agent rankings
        self.agent_pipeline.initialize_rankings()
    
    def preprocess_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image_input: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor ready for inference
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def predict_single_model(
        self,
        image_tensor: torch.Tensor,
        model_name: str
    ) -> ModelPrediction:
        """
        Get prediction from a single model.
        
        Args:
            image_tensor: Preprocessed image tensor
            model_name: Name of the model
            
        Returns:
            ModelPrediction object
        """
        model = self.models[model_name]
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        
        return ModelPrediction(
            model_name=model_name,
            predicted_class=CLASSES[pred_idx],
            predicted_class_idx=pred_idx,
            confidence=float(probs[pred_idx]),
            class_probabilities={
                cls: float(probs[i]) for i, cls in enumerate(CLASSES)
            }
        )
    
    def predict_all_models(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> List[ModelPrediction]:
        """
        Get predictions from all loaded models.
        
        Args:
            image_input: Image to classify
            
        Returns:
            List of ModelPrediction objects
        """
        if not self.models:
            raise RuntimeError("No models loaded. Call load_all_models() first.")
        
        image_tensor = self.preprocess_image(image_input)
        
        predictions = []
        for model_name in self.models:
            pred = self.predict_single_model(image_tensor, model_name)
            predictions.append(pred)
        
        return predictions
    
    def classify(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        generate_gradcam: bool = False,
        gradcam_output_path: Optional[Path] = None
    ) -> Dict:
        """
        Perform full classification with all models and agents.
        
        Args:
            image_input: Image to classify
            generate_gradcam: Whether to generate Grad-CAM visualization
            gradcam_output_path: Path to save Grad-CAM output
            
        Returns:
            Dictionary with comprehensive results
        """
        # Get predictions from all models
        predictions = self.predict_all_models(image_input)
        
        # Process through agent pipeline
        result = self.agent_pipeline.process_predictions(predictions)
        
        # Add individual model predictions
        result["model_predictions"] = {
            p.model_name: {
                "class": p.predicted_class,
                "confidence": p.confidence,
                "probabilities": p.class_probabilities
            }
            for p in predictions
        }
        
        # Add metadata
        result["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "models_used": list(self.models.keys()),
            "device": str(self.device)
        }
        
        # Generate Grad-CAM if requested
        if generate_gradcam:
            image_path = str(image_input) if isinstance(image_input, (str, Path)) else None
            if image_path:
                output_path = gradcam_output_path or GRADCAM_DIR / "inference_gradcam.png"
                visualize_all_models(image_path, save_path=output_path, device=self.device)
                result["gradcam_path"] = str(output_path)
        
        return result
    
    def classify_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]]
    ) -> List[Dict]:
        """
        Classify multiple images.
        
        Args:
            image_inputs: List of images to classify
            
        Returns:
            List of classification results
        """
        results = []
        for i, image in enumerate(image_inputs):
            print(f"Processing image {i+1}/{len(image_inputs)}...")
            result = self.classify(image)
            results.append(result)
        return results
    
    def generate_report(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> str:
        """
        Generate a full diagnostic report for an image.
        
        Args:
            image_input: Image to classify
            
        Returns:
            Formatted report string
        """
        predictions = self.predict_all_models(image_input)
        return self.agent_pipeline.generate_full_report(predictions)
    
    def save_result(
        self,
        result: Dict,
        output_path: Path
    ) -> None:
        """
        Save classification result to JSON file.
        
        Args:
            result: Classification result dictionary
            output_path: Path to save the result
        """
        # Make result JSON serializable
        serializable_result = {
            k: v for k, v in result.items()
            if not isinstance(v, (np.ndarray, torch.Tensor))
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        print(f"Result saved to: {output_path}")


# ==================== CLI INTERFACE ====================

def run_inference(
    image_path: str,
    output_dir: Optional[str] = None,
    generate_gradcam: bool = True,
    generate_report: bool = True
) -> Dict:
    """
    Run inference on a single image.
    
    Args:
        image_path: Path to the MRI image
        output_dir: Directory to save outputs
        generate_gradcam: Whether to generate Grad-CAM
        generate_report: Whether to generate text report
        
    Returns:
        Classification result dictionary
    """
    output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier
    print("Initializing Brain MRI Classifier...")
    classifier = BrainMRIClassifier()
    
    # Get image name for outputs
    image_name = Path(image_path).stem
    
    # Run classification
    print(f"\nClassifying: {image_path}")
    gradcam_path = output_dir / f"{image_name}_gradcam.png" if generate_gradcam else None
    
    result = classifier.classify(
        image_path,
        generate_gradcam=generate_gradcam,
        gradcam_output_path=gradcam_path
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"Prediction: {result['final_prediction'].upper()}")
    print(f"Confidence: {result['consensus_confidence']*100:.1f}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Agreement: {result['agreement_ratio']*100:.0f}%")
    
    # Save result
    result_path = output_dir / f"{image_name}_result.json"
    classifier.save_result(result, result_path)
    
    # Generate report
    if generate_report:
        report = classifier.generate_report(image_path)
        report_path = output_dir / f"{image_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        print("\n" + report)
    
    return result


def run_batch_inference(
    input_dir: str,
    output_dir: Optional[str] = None,
    extensions: List[str] = None
) -> List[Dict]:
    """
    Run inference on all images in a directory.
    
    Args:
        input_dir: Directory containing MRI images
        output_dir: Directory to save outputs
        extensions: Image file extensions to process
        
    Returns:
        List of classification results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else RESULTS_DIR / "batch_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} images")
    
    # Initialize classifier
    classifier = BrainMRIClassifier()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path.name}")
        
        result = classifier.classify(str(image_path))
        result["input_file"] = str(image_path)
        results.append(result)
        
        print(f"  → {result['final_prediction']} ({result['consensus_confidence']*100:.1f}%)")
    
    # Save batch results
    batch_results_path = output_dir / "batch_results.json"
    with open(batch_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH INFERENCE SUMMARY")
    print("=" * 60)
    
    class_counts = {}
    for r in results:
        cls = r['final_prediction']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"Total images: {len(results)}")
    print("Classification breakdown:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} ({count/len(results)*100:.1f}%)")
    
    print(f"\nResults saved to: {batch_results_path}")
    
    return results


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Brain MRI Tumor Classification Inference Pipeline"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single MRI image"
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Directory containing multiple MRI images"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-gradcam", action="store_true",
        help="Skip Grad-CAM visualization"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip text report generation"
    )
    
    args = parser.parse_args()
    
    if args.image:
        run_inference(
            args.image,
            output_dir=args.output,
            generate_gradcam=not args.no_gradcam,
            generate_report=not args.no_report
        )
    elif args.batch:
        run_batch_inference(
            args.batch,
            output_dir=args.output
        )
    else:
        print("Please specify --image or --batch")
        print("Run with --help for more options")
        
        # Demo mode
        print("\n" + "=" * 60)
        print("DEMO MODE - Classifier Initialization")
        print("=" * 60)
        
        try:
            classifier = BrainMRIClassifier(load_models=False)
            print("[OK] Classifier initialized successfully")
            print(f"  Models configured: {classifier.model_names}")
            print(f"  Device: {classifier.device}")
            print("\nTo run inference, use:")
            print("  python inference.py --image /path/to/mri.jpg")
            print("  python inference.py --batch /path/to/folder/")
        except Exception as e:
            print(f"Error initializing classifier: {e}")
