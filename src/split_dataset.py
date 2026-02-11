"""
Dataset Splitting Module for Brain MRI Tumor Classification.
Creates stratified train/validation/test splits with no data leakage.
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, CLASSES,
    create_directories
)


def get_image_files(directory: Path, class_name: str) -> List[Path]:
    """
    Get all image files from a class directory.
    
    Args:
        directory: Base directory containing class folders
        class_name: Name of the class folder
        
    Returns:
        List of image file paths
    """
    class_dir = directory / class_name
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if not class_dir.exists():
        return []
    
    return [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]


def stratified_split(
    files: List[Path],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split files into train/val/test sets.
    
    Args:
        files: List of file paths
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle files
    files = files.copy()
    random.shuffle(files)
    
    # Calculate split indices
    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def copy_files_to_split(
    files: List[Path],
    destination_dir: Path,
    class_name: str,
    move: bool = False
) -> int:
    """
    Copy or move files to the destination split directory.
    
    Args:
        files: List of file paths to copy
        destination_dir: Base destination directory (train/val/test)
        class_name: Name of the class folder
        move: If True, move files instead of copying
        
    Returns:
        Number of files copied/moved
    """
    dest_class_dir = destination_dir / class_name
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for file_path in files:
        dest_path = dest_class_dir / file_path.name
        
        if move:
            shutil.move(str(file_path), str(dest_path))
        else:
            shutil.copy2(str(file_path), str(dest_path))
        count += 1
    
    return count


def split_dataset(
    source_dir: Path = PROCESSED_DATASET_DIR,
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR,
    move_files: bool = False
) -> Dict:
    """
    Split the dataset into train/val/test sets for all classes.
    
    Args:
        source_dir: Directory containing processed images with class subfolders
        train_dir: Directory for training set
        val_dir: Directory for validation set
        test_dir: Directory for test set
        move_files: If True, move files instead of copying
        
    Returns:
        Dictionary with split statistics
    """
    stats = {
        "total": 0,
        "train": 0,
        "val": 0,
        "test": 0,
        "per_class": {}
    }
    
    print("\nSplitting dataset...")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%")
    print(f"  Validation: {VAL_RATIO*100:.0f}%")
    print(f"  Test: {TEST_RATIO*100:.0f}%")
    print(f"  Random Seed: {RANDOM_SEED}")
    
    for class_name in CLASSES:
        print(f"\n  Processing class: {class_name}")
        
        # Get all image files for this class
        files = get_image_files(source_dir, class_name)
        
        if not files:
            print(f"    Warning: No files found for class {class_name}")
            stats["per_class"][class_name] = {"train": 0, "val": 0, "test": 0}
            continue
        
        # Split files
        train_files, val_files, test_files = stratified_split(files)
        
        # Copy/move to destination directories
        train_count = copy_files_to_split(train_files, train_dir, class_name, move_files)
        val_count = copy_files_to_split(val_files, val_dir, class_name, move_files)
        test_count = copy_files_to_split(test_files, test_dir, class_name, move_files)
        
        # Update statistics
        total = train_count + val_count + test_count
        stats["total"] += total
        stats["train"] += train_count
        stats["val"] += val_count
        stats["test"] += test_count
        stats["per_class"][class_name] = {
            "train": train_count,
            "val": val_count,
            "test": test_count
        }
        
        print(f"    Total: {total} | Train: {train_count} | Val: {val_count} | Test: {test_count}")
    
    return stats


def verify_no_leakage(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR
) -> bool:
    """
    Verify there is no image leakage between splits.
    
    Args:
        train_dir: Training set directory
        val_dir: Validation set directory
        test_dir: Test set directory
        
    Returns:
        True if no leakage detected, False otherwise
    """
    print("\nVerifying no data leakage between splits...")
    
    all_files = defaultdict(set)
    
    for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for f in class_dir.iterdir():
                    if f.is_file():
                        all_files[split_name].add(f.name)
    
    # Check for overlaps
    train_val_overlap = all_files["train"] & all_files["val"]
    train_test_overlap = all_files["train"] & all_files["test"]
    val_test_overlap = all_files["val"] & all_files["test"]
    
    has_leakage = False
    
    if train_val_overlap:
        print(f"  WARNING: {len(train_val_overlap)} files overlap between train and val!")
        has_leakage = True
    
    if train_test_overlap:
        print(f"  WARNING: {len(train_test_overlap)} files overlap between train and test!")
        has_leakage = True
    
    if val_test_overlap:
        print(f"  WARNING: {len(val_test_overlap)} files overlap between val and test!")
        has_leakage = True
    
    if not has_leakage:
        print("  [OK] No data leakage detected!")
    
    return not has_leakage


def print_split_summary(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR
):
    """Print a summary of the dataset split."""
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    
    for split_name, split_dir in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        total = 0
        print(f"\n{split_name} Set ({split_dir}):")
        
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len([f for f in class_dir.iterdir() if f.is_file()])
                print(f"  {class_name}: {count}")
                total += count
        
        print(f"  Total: {total}")


if __name__ == "__main__":
    print("=" * 60)
    print("Brain MRI Dataset Splitting Module")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Check if source directory has images directly (not in split folders)
    has_source_images = False
    for class_name in CLASSES:
        source_class_dir = PROCESSED_DATASET_DIR / class_name
        if source_class_dir.exists():
            files = get_image_files(PROCESSED_DATASET_DIR, class_name)
            if files:
                has_source_images = True
                break
    
    if not has_source_images:
        print(f"\nNo images found in {PROCESSED_DATASET_DIR}")
        print("Please run preprocessing.py first to process the raw images.")
        
        # Check if split directories already have images
        for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            for class_name in CLASSES:
                class_dir = split_dir / class_name
                if class_dir.exists() and list(class_dir.iterdir()):
                    print("\nNote: Split directories already contain images.")
                    print_split_summary()
                    sys.exit(0)
        
        sys.exit(1)
    
    # Perform split
    stats = split_dataset()
    
    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"\nTotal Images: {stats['total']}")
    print(f"  Train: {stats['train']} ({stats['train']/stats['total']*100:.1f}%)")
    print(f"  Val: {stats['val']} ({stats['val']/stats['total']*100:.1f}%)")
    print(f"  Test: {stats['test']} ({stats['test']/stats['total']*100:.1f}%)")
    
    # Verify no leakage
    verify_no_leakage()
    
    # Print summary
    print_split_summary()
