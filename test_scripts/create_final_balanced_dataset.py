#!/usr/bin/env python3
"""Script to create final balanced dataset using available valid images."""

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


def create_balanced_dataset(source_root: Path, target_root: Path) -> None:
    """Create balanced dataset using only classes with sufficient valid images.
    
    Args:
        source_root: Root directory of original dataset
        target_root: Root directory for balanced dataset
    """
    random.seed(42)
    
    if target_root.exists():
        shutil.rmtree(target_root)
    
    target_root.mkdir(parents=True, exist_ok=True)
    
    # Analyze available valid images per class
    class_valid_counts = {}
    for class_dir in source_root.iterdir():
        if class_dir.is_dir():
            valid_images = _get_valid_images(class_dir)
            class_valid_counts[class_dir.name] = len(valid_images)
            print(f"Classe {class_dir.name}: {len(valid_images)} imagens válidas")
    
    # Filter classes with at least 400 valid images for a reasonable dataset
    min_images_threshold = 400
    usable_classes = {k: v for k, v in class_valid_counts.items() if v >= min_images_threshold}
    
    if not usable_classes:
        print("Nenhuma classe tem imagens válidas suficientes!")
        return
    
    # Use the minimum count among usable classes for balance
    images_per_class = min(usable_classes.values())
    print(f"\nCriando dataset balanceado com {images_per_class} imagens por classe")
    print(f"Classes utilizadas: {list(usable_classes.keys())}")
    
    total_copied = 0
    for class_name in usable_classes.keys():
        class_dir = source_root / class_name
        target_class_dir = target_root / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        valid_images = _get_valid_images(class_dir)
        selected_images = random.sample(valid_images, images_per_class)
        
        copied_count = 0
        for img_path in selected_images:
            target_path = target_class_dir / img_path.name
            try:
                shutil.copy2(img_path, target_path)
                copied_count += 1
            except Exception as e:
                print(f"Erro copiando {img_path}: {e}")
        
        print(f"Copiadas {copied_count} imagens para classe {class_name}")
        total_copied += copied_count
    
    print(f"\nDataset balanceado criado com sucesso!")
    print(f"Total de imagens: {total_copied}")
    print(f"Classes: {len(usable_classes)}")
    print(f"Imagens por classe: {images_per_class}")


if __name__ == "__main__":
    source_dataset = Path("spatial_images_dataset")
    balanced_dataset = Path("spatial_images_dataset_final")
    
    create_balanced_dataset(source_dataset, balanced_dataset)


