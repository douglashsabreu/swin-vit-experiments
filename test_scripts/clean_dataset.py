#!/usr/bin/env python3
"""Script to clean dataset by removing corrupted images."""

import os
from pathlib import Path
from PIL import Image


def _is_valid_image(image_path: Path) -> bool:
    """Check if image file is valid and can be opened.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def clean_dataset(dataset_path: Path) -> None:
    """Remove corrupted images from dataset.
    
    Args:
        dataset_path: Path to dataset directory
    """
    removed_count = 0
    total_count = 0
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        print(f"Cleaning class {class_dir.name}...")
        
        for image_file in class_dir.glob("*.png"):
            total_count += 1
            
            if not _is_valid_image(image_file):
                print(f"Removing corrupted image: {image_file}")
                image_file.unlink()
                removed_count += 1
    
    print(f"Cleaned dataset: removed {removed_count} corrupted images out of {total_count} total")


if __name__ == "__main__":
    dataset_path = Path("spatial_images_dataset_balanced")
    clean_dataset(dataset_path)


