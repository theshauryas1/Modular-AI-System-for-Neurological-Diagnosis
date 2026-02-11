"""
Preprocess ALL MRI image sources into a single processed_dataset directory.
Merges archive, archive (1), and archive (2). Deduplicates via MD5 hashing.
"""

import os
import sys
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ==================== CONFIG ====================

IMAGE_SIZE = 224
PROCESSED_DIR = Path(r"d:\mri\processed_dataset")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Mapping from source folder names -> our class names
NAME_MAP = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
}

CLASSES = ["glioma", "meningioma", "pituitary"]

# All source directories containing class subfolders
SOURCE_DIRS = [
    Path(r"d:\mri\dataset\archive\Data\Tumor"),
    Path(r"d:\mri\dataset\archive (1)\Training"),
    Path(r"d:\mri\dataset\archive (1)\Testing"),
    Path(r"d:\mri\dataset\archive (2)\brain_tumor_dataset\brain_tumor_classification\Training"),
    Path(r"d:\mri\dataset\archive (2)\brain_tumor_dataset\brain_tumor_classification\Testing"),
]


# ==================== FUNCTIONS ====================

def preprocess_image(image_path: Path, output_path: Path) -> bool:
    """Load, resize to 224x224, and save an image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            pil_img = Image.open(image_path)
            image = np.array(pil_img)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(output_path), image)
        return True
    except Exception:
        return False


def main():
    # Create output dirs
    for cls in CLASSES:
        (PROCESSED_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Track hashes for dedup
    seen_hashes = {cls: set() for cls in CLASSES}
    stats = {cls: {"found": 0, "processed": 0, "duplicates": 0, "failed": 0} for cls in CLASSES}
    counters = {cls: 0 for cls in CLASSES}

    print("=" * 60)
    print("MERGING ALL DATASET SOURCES")
    print("=" * 60)

    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"\nSkipping (not found): {src_dir}")
            continue
        print(f"\nScanning: {src_dir}")

        for folder in sorted(src_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_name = folder.name.lower()
            target_class = NAME_MAP.get(folder_name)
            if target_class is None:
                print(f"  Skipping folder: {folder.name} (not a tumor class)")
                continue

            images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXT]
            print(f"  {folder.name} -> {target_class}: {len(images)} images")

            for img_path in images:
                stats[target_class]["found"] += 1

                # Hash for dedup
                try:
                    file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                except Exception:
                    stats[target_class]["failed"] += 1
                    continue

                if file_hash in seen_hashes[target_class]:
                    stats[target_class]["duplicates"] += 1
                    continue
                seen_hashes[target_class].add(file_hash)

                # Preprocess and save
                counters[target_class] += 1
                out_name = f"{target_class}_{counters[target_class]:05d}{img_path.suffix.lower()}"
                out_path = PROCESSED_DIR / target_class / out_name

                if preprocess_image(img_path, out_path):
                    stats[target_class]["processed"] += 1
                else:
                    stats[target_class]["failed"] += 1

            # Progress
            s = stats[target_class]
            print(f"    Running total for {target_class}: {s['processed']} processed, {s['duplicates']} dups skipped")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    total_processed = 0
    for cls in CLASSES:
        s = stats[cls]
        total_processed += s["processed"]
        print(f"\n  {cls}:")
        print(f"    Found:      {s['found']}")
        print(f"    Processed:  {s['processed']}")
        print(f"    Duplicates: {s['duplicates']}")
        print(f"    Failed:     {s['failed']}")

    print(f"\n  Total unique images processed: {total_processed}")
    print(f"  Output directory: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
