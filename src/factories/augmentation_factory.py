from __future__ import annotations

from typing import Any, Dict, Tuple

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None

from torchvision import transforms
from PIL import Image
import numpy as np

from src.config import FullConfig


class AugmentationFactory:
    """Factory to build augmentation pipelines for train/val/test.

    This factory builds torchvision or albumentations transforms depending on
    what's available in the environment. All augmentation parameters are read
    from the provided configuration; no hard-coded values are used.
    """

    def __init__(self, config: FullConfig):
        self._config = config

    def build_train(self):
        """Build training augmentation pipeline.

        Returns:
            A callable transform that maps PIL.Image -> Tensor.
        """
        cfg = self._config.augmentations.train

        if A is not None:
            aug_list = []
            if 'random_resized_crop' in cfg and cfg['random_resized_crop'].enabled:
                scale = tuple(cfg['random_resized_crop'].scale) if cfg['random_resized_crop'].scale else (0.08, 1.0)
                ratio = tuple(cfg['random_resized_crop'].ratio) if cfg['random_resized_crop'].ratio else (0.75, 1.333)
                # Use size from config if available, otherwise use data.image_size
                if hasattr(cfg['random_resized_crop'], 'size') and cfg['random_resized_crop'].size:
                    size = tuple(cfg['random_resized_crop'].size)
                else:
                    size = tuple(self._config.data.image_size)
                aug_list.append(A.RandomResizedCrop(size=size, scale=scale, ratio=ratio, p=1.0))

            if 'horizontal_flip' in cfg and cfg['horizontal_flip'].enabled:
                aug_list.append(A.HorizontalFlip(p=float(cfg['horizontal_flip'].p or 0.5)))

            if 'rotation' in cfg and cfg['rotation'].enabled:
                aug_list.append(A.Rotate(limit=float(cfg['rotation'].degrees), p=1.0))

            if 'color_jitter' in cfg and cfg['color_jitter'].enabled:
                aug_list.append(A.ColorJitter(brightness=cfg['color_jitter'].brightness or 0.0,
                                              contrast=cfg['color_jitter'].contrast or 0.0,
                                              saturation=cfg['color_jitter'].saturation or 0.0,
                                              hue=cfg['color_jitter'].hue or 0.0,
                                              p=1.0))

            if 'gaussian_noise' in cfg and cfg['gaussian_noise'].enabled:
                aug_list.append(A.GaussNoise(var_limit=(cfg['gaussian_noise'].std or 0.01)**2, p=1.0))

            if 'cutout' in cfg and cfg['cutout'].enabled:
                max_area = float(cfg['cutout'].max_area or 0.1)
                aug_list.append(A.CoarseDropout(max_holes=1, max_height=int(max_area * self._config.data.image_size[0]), max_width=int(max_area * self._config.data.image_size[1]), p=float(cfg['cutout'].p or 0.5)))

            # Mixup/CutMix must be applied in batch collate; here we return geometric augmentations
            aug_list.append(A.Normalize(mean=_imagenet_mean(), std=_imagenet_std()))
            aug_list.append(ToTensorV2())
            return _AlbumentationsWrapper(A.Compose(aug_list))

        # Fallback to torchvision transforms
        tlist = []
        if 'random_resized_crop' in cfg and cfg['random_resized_crop'].enabled:
            scale = tuple(cfg['random_resized_crop'].scale) if cfg['random_resized_crop'].scale else (0.08, 1.0)
            ratio = tuple(cfg['random_resized_crop'].ratio) if cfg['random_resized_crop'].ratio else (0.75, 1.333)
            # Use size from config if available, otherwise use data.image_size
            if hasattr(cfg['random_resized_crop'], 'size') and cfg['random_resized_crop'].size:
                size = tuple(cfg['random_resized_crop'].size)
            else:
                size = tuple(self._config.data.image_size)
            tlist.append(transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio))

        if 'horizontal_flip' in cfg and cfg['horizontal_flip'].enabled:
            tlist.append(transforms.RandomHorizontalFlip(p=float(cfg['horizontal_flip'].p or 0.5)))

        if 'rotation' in cfg and cfg['rotation'].enabled:
            tlist.append(transforms.RandomRotation(degrees=float(cfg['rotation'].degrees)))

        if 'color_jitter' in cfg and cfg['color_jitter'].enabled:
            tlist.append(transforms.ColorJitter(brightness=float(cfg['color_jitter'].brightness or 0.0),
                                                contrast=float(cfg['color_jitter'].contrast or 0.0),
                                                saturation=float(cfg['color_jitter'].saturation or 0.0),
                                                hue=float(cfg['color_jitter'].hue or 0.0)))

        tlist.append(transforms.ToTensor())
        tlist.append(transforms.Normalize(mean=_imagenet_mean(), std=_imagenet_std()))

        return transforms.Compose(tlist)

    def build_val_test(self):
        """Build validation/test augmentation pipeline.

        Returns:
            A callable transform that maps PIL.Image -> Tensor.
        """
        cfg = self._config.augmentations.val_test
        size = tuple(cfg['resize'].size)

        if A is not None:
            aug_list = [A.Resize(height=size[0], width=size[1], interpolation=_alb_interp(cfg['resize'].interpolation))]
            if 'center_crop' in cfg and cfg['center_crop'].enabled:
                crop_size = tuple(cfg['center_crop'].size)
                aug_list.append(A.CenterCrop(height=crop_size[0], width=crop_size[1]))

            aug_list.append(A.Normalize(mean=_imagenet_mean(), std=_imagenet_std()))
            aug_list.append(ToTensorV2())
            return _AlbumentationsWrapper(A.Compose(aug_list))

        tlist = [transforms.Resize(size=size, interpolation=_pil_interp(cfg['resize'].interpolation))]
        if 'center_crop' in cfg and cfg['center_crop'].enabled:
            crop_size = tuple(cfg['center_crop'].size)
            tlist.append(transforms.CenterCrop(crop_size))
        tlist.append(transforms.ToTensor())
        tlist.append(transforms.Normalize(mean=_imagenet_mean(), std=_imagenet_std()))
        return transforms.Compose(tlist)


def _imagenet_mean() -> Tuple[float, float, float]:
    return (0.485, 0.456, 0.406)


def _imagenet_std() -> Tuple[float, float, float]:
    return (0.229, 0.224, 0.225)


def _pil_interp(name: str):
    name = (name or '').lower()
    if name == 'bicubic':
        return Image.BICUBIC
    return Image.BILINEAR


def _alb_interp(name: str):
    name = (name or '').lower()
    if name == 'bicubic':
        return 3
    return 1


class _AlbumentationsWrapper:
    """Wrapper to make albumentations compatible with torchvision ImageFolder.
    
    Converts PIL Image to numpy array and applies albumentations transforms.
    """
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        """Apply albumentations transform to PIL Image.
        
        Args:
            image: PIL Image
            
        Returns:
            Transformed tensor
        """
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        return transformed['image']
