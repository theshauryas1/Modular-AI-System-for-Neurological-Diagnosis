"""
Image Preprocessing Module for Brain MRI Tumor Classification.
Handles image loading, conversion, resizing, and validation.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR, PROCESSED_DATASET_DIR, IMAGE_SIZE, 
    CLASSES, create_directories
)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from the given path using OpenCV.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array, or None if loading fails
    """
    try:
        # Read image using OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            # Try with PIL as fallback
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        RGB image as numpy array
    """
    if len(image.shape) == 2:
        # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        # BGRA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image


def resize_image(image: np.ndarray, size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """
    Resize image to the specified dimensions.
    
    Args:
        image: Input image
        size: Target size (width, height)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


def preprocess_single_image(image_path: str, output_path: str) -> bool:
    """
    Preprocess a single image: load, convert to RGB, resize, and save.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        
    Returns:
        True if successful, False otherwise
    """
    # Load image
    image = load_image(image_path)
    if image is None:
        return False
    
    # Convert to RGB
    image = convert_to_rgb(image)
    
    # Resize
    image = resize_image(image)
    
    # Save (convert back to BGR for OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)
    
    return True


def validate_images(directory: Path) -> Tuple[List[str], List[str]]:
    """
    Validate all images in the directory structure.
    Checks for corrupted and invalid images.
    
    Args:
        directory: Base directory containing class folders
        
    Returns:
        Tuple of (valid_images, invalid_images) lists
    """
    valid_images = []
    invalid_images = []
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for class_name in CLASSES:
        class_dir = directory / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
            
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in supported_extensions:
                continue
                
            try:
                # Try to load and verify image
                image = load_image(str(image_path))
                if image is None:
                    invalid_images.append(str(image_path))
                    continue
                    
                # Check minimum dimensions
                if image.shape[0] < 10 or image.shape[1] < 10:
                    invalid_images.append(str(image_path))
                    continue
                    
                valid_images.append(str(image_path))
                
            except Exception as e:
                print(f"Validation error for {image_path}: {e}")
                invalid_images.append(str(image_path))
    
    return valid_images, invalid_images


def preprocess_dataset(
    input_dir: Path = DATASET_DIR, 
    output_dir: Path = PROCESSED_DATASET_DIR
) -> dict:
    """
    Preprocess the entire dataset: load, convert, resize, and save all images.
    
    Args:
        input_dir: Directory containing raw images (with class subfolders)
        output_dir: Directory to save processed images
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total": 0,
        "processed": 0,
        "failed": 0,
        "per_class": {}
    }
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for class_name in CLASSES:
        input_class_dir = input_dir / class_name
        output_class_dir = output_dir / class_name
        
        if not input_class_dir.exists():
            print(f"Warning: Input class directory not found: {input_class_dir}")
            stats["per_class"][class_name] = {"processed": 0, "failed": 0}
            continue
        
        # Create output directory
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        class_processed = 0
        class_failed = 0
        
        # Get all image files
        image_files = [
            f for f in input_class_dir.iterdir() 
            if f.suffix.lower() in supported_extensions
        ]
        
        print(f"\nProcessing class: {class_name} ({len(image_files)} images)")
        
        for image_path in tqdm(image_files, desc=f"  {class_name}"):
            stats["total"] += 1
            output_path = output_class_dir / image_path.name
            
            if preprocess_single_image(str(image_path), str(output_path)):
                stats["processed"] += 1
                class_processed += 1
            else:
                stats["failed"] += 1
                class_failed += 1
        
        stats["per_class"][class_name] = {
            "processed": class_processed,
            "failed": class_failed
        }
    
    return stats


def get_dataset_statistics(directory: Path) -> dict:
    """
    Get statistics about the dataset.
    
    Args:
        directory: Base directory containing class folders
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_images": 0,
        "per_class": {},
        "image_sizes": [],
    }
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for class_name in CLASSES:
        class_dir = directory / class_name
        if not class_dir.exists():
            stats["per_class"][class_name] = 0
            continue
            
        count = 0
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in supported_extensions:
                count += 1
                
                # Sample image sizes (first 10 per class)
                if len(stats["image_sizes"]) < 40:
                    image = load_image(str(image_path))
                    if image is not None:
                        stats["image_sizes"].append({
                            "path": str(image_path),
                            "shape": image.shape
                        })
        
        stats["per_class"][class_name] = count
        stats["total_images"] += count
    
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("Brain MRI Image Preprocessing Module")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"\nError: Dataset directory not found: {DATASET_DIR}")
        print("Please place your MRI images in the dataset folder with class subfolders:")
        for class_name in CLASSES:
            print(f"  - {DATASET_DIR / class_name}/")
        sys.exit(1)
    
    # Get dataset statistics
    print("\nAnalyzing dataset...")
    stats = get_dataset_statistics(DATASET_DIR)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Images: {stats['total_images']}")
    print(f"  Per Class:")
    for class_name, count in stats["per_class"].items():
        print(f"    - {class_name}: {count}")
    
    if stats["total_images"] == 0:
        print("\nNo images found in dataset. Please add images first.")
        sys.exit(1)
    
    # Validate images
    print("\nValidating images...")
    valid, invalid = validate_images(DATASET_DIR)
    print(f"  Valid: {len(valid)}")
    print(f"  Invalid: {len(invalid)}")
    
    if invalid:
        print("\n  Invalid images:")
        for img in invalid[:10]:  # Show first 10
            print(f"    - {img}")
        if len(invalid) > 10:
            print(f"    ... and {len(invalid) - 10} more")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    process_stats = preprocess_dataset()
    
    print(f"\nPreprocessing Complete!")
    print(f"  Total: {process_stats['total']}")
    print(f"  Processed: {process_stats['processed']}")
    print(f"  Failed: {process_stats['failed']}")
    
    print(f"\nProcessed images saved to: {PROCESSED_DATASET_DIR}")
