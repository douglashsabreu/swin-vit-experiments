#!/usr/bin/env python3
"""Script to create a properly balanced dataset with 1800 valid images per class."""

import shutil
import random
from pathlib import Path
from typing import List
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


def _get_valid_images(source_dir: Path) -> List[Path]:
    """Get all valid images from source directory.
    
    Args:
        source_dir: Source directory containing images
        
    Returns:
        List of valid image paths
    """
    valid_images = []
    
    for image_file in source_dir.glob("*.png"):
        if _is_valid_image(image_file):
            valid_images.append(image_file)
    
    return valid_images


def _create_balanced_class(source_dir: Path, target_dir: Path, images_per_class: int) -> int:
    """Create balanced class directory with specified number of valid images.
    
    Args:
        source_dir: Source class directory
        target_dir: Target class directory
        images_per_class: Number of images to include
        
    Returns:
        Number of images actually copied
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    valid_images = _get_valid_images(source_dir)
    
    if len(valid_images) < images_per_class:
        print(f"Warning: Class {source_dir.name} has only {len(valid_images)} valid images, need {images_per_class}")
        images_to_copy = valid_images
    else:
        images_to_copy = random.sample(valid_images, images_per_class)
    
    copied_count = 0
    for img_path in images_to_copy:
        target_path = target_dir / img_path.name
        try:
            shutil.copy2(img_path, target_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {img_path}: {e}")
    
    return copied_count


def create_balanced_dataset(source_root: Path, target_root: Path, images_per_class: int = 1800) -> None:
    """Create balanced dataset with equal number of valid images per class.
    
    Args:
        source_root: Root directory of original dataset
        target_root: Root directory for balanced dataset
        images_per_class: Number of images per class
    """
    random.seed(42)
    
    if target_root.exists():
        shutil.rmtree(target_root)
    
    target_root.mkdir(parents=True, exist_ok=True)
    
    class_dirs = [d for d in source_root.iterdir() if d.is_dir()]
    
    total_copied = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        target_class_dir = target_root / class_name
        
        print(f"Processing class {class_name}...")
        
        # First, count valid images
        valid_images = _get_valid_images(class_dir)
        print(f"Found {len(valid_images)} valid images in class {class_name}")
        
        copied_count = _create_balanced_class(class_dir, target_class_dir, images_per_class)
        print(f"Copied {copied_count} images for class {class_name}")
        total_copied += copied_count
    
    print(f"Balanced dataset created successfully! Total images: {total_copied}")


if __name__ == "__main__":
    source_dataset = Path("spatial_images_dataset")
    balanced_dataset = Path("spatial_images_dataset_balanced_v2")
    
    create_balanced_dataset(source_dataset, balanced_dataset, 1800)
    print("New balanced dataset created!")


