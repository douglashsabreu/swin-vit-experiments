#!/usr/bin/env python3
"""Script to create a balanced dataset with 1600 images per class."""

import shutil
import random
from pathlib import Path
from typing import List


def _get_random_images(source_dir: Path, target_count: int) -> List[Path]:
    """Get random selection of images from source directory.
    
    Args:
        source_dir: Source directory containing images
        target_count: Number of images to select
        
    Returns:
        List of selected image paths
    """
    image_files = list(source_dir.glob("*.png"))
    if len(image_files) < target_count:
        raise ValueError(f"Not enough images in {source_dir}: {len(image_files)} < {target_count}")
    
    return random.sample(image_files, target_count)


def _create_balanced_class(source_dir: Path, target_dir: Path, images_per_class: int) -> None:
    """Create balanced class directory with specified number of images.
    
    Args:
        source_dir: Source class directory
        target_dir: Target class directory
        images_per_class: Number of images to include
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    selected_images = _get_random_images(source_dir, images_per_class)
    
    for img_path in selected_images:
        target_path = target_dir / img_path.name
        shutil.copy2(img_path, target_path)


def create_balanced_dataset(source_root: Path, target_root: Path, images_per_class: int = 1600) -> None:
    """Create balanced dataset with equal number of images per class.
    
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
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        target_class_dir = target_root / class_name
        
        print(f"Processing class {class_name}...")
        _create_balanced_class(class_dir, target_class_dir, images_per_class)
        print(f"Created {images_per_class} images for class {class_name}")


if __name__ == "__main__":
    source_dataset = Path("spatial_images_dataset")
    balanced_dataset = Path("spatial_images_dataset_balanced")
    
    create_balanced_dataset(source_dataset, balanced_dataset, 1600)
    print("Balanced dataset created successfully!")


