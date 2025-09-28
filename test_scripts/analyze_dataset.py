#!/usr/bin/env python3
"""Script to analyze the dataset and count valid images per class."""

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


def analyze_dataset(dataset_path: Path) -> None:
    """Analyze dataset and count valid vs corrupted images per class.
    
    Args:
        dataset_path: Path to dataset directory
    """
    print("Análise do Dataset:")
    print("=" * 50)
    
    total_images = 0
    total_valid = 0
    total_corrupted = 0
    
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        valid_count = 0
        corrupted_count = 0
        
        for image_file in class_dir.glob("*.png"):
            total_images += 1
            
            if _is_valid_image(image_file):
                valid_count += 1
                total_valid += 1
            else:
                corrupted_count += 1
                total_corrupted += 1
        
        total_class = valid_count + corrupted_count
        valid_percentage = (valid_count / total_class * 100) if total_class > 0 else 0
        
        print(f"Classe {class_name}:")
        print(f"  Total: {total_class} imagens")
        print(f"  Válidas: {valid_count} ({valid_percentage:.1f}%)")
        print(f"  Corrompidas: {corrupted_count}")
        print()
    
    print("RESUMO GERAL:")
    print(f"Total de imagens: {total_images}")
    print(f"Imagens válidas: {total_valid} ({total_valid/total_images*100:.1f}%)")
    print(f"Imagens corrompidas: {total_corrupted} ({total_corrupted/total_images*100:.1f}%)")


if __name__ == "__main__":
    dataset_path = Path("spatial_images_dataset")
    analyze_dataset(dataset_path)


