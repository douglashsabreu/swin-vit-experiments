from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

from src.config import FullConfig
from src.factories.augmentation_factory import AugmentationFactory


class DataModule:
    """DataModule providing dataloaders for train/val/test with reproducible splits.

    The module supports stratified or grouped splitting strategies as declared in
    the configuration. Transform pipelines are provided by `AugmentationFactory`.
    """

    def __init__(self, config: FullConfig):
        self._config = config
        self._seed = int(config.experiment.seed)
        self._batch_size = int(config.training.batch_size)
        self._num_workers = int(config.data.num_workers)
        self._pin_memory = bool(config.data.pin_memory)

        self._augment_factory = AugmentationFactory(config)

        self._train_dataset: Optional[ImageFolder] = None
        self._val_dataset: Optional[ImageFolder] = None
        self._test_dataset: Optional[ImageFolder] = None

    def setup(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        train_dir = Path(self._config.data.train_dir)
        val_dir = self._config.data.val_dir
        test_dir = self._config.data.test_dir

        if val_dir:
            train_ds = ImageFolder(str(train_dir), transform=self._augment_factory.build_train())
            val_ds = ImageFolder(str(val_dir), transform=self._augment_factory.build_val_test())
            self._train_dataset = train_ds
            self._val_dataset = val_ds
        else:
            # Need to split from train_dir according to strategy
            full_ds = ImageFolder(str(train_dir), transform=self._augment_factory.build_train())
            targets = [y for _, y in full_ds.samples]
            strategy = self._config.data.split.strategy
            val_ratio = float(self._config.data.split.val_ratio)
            test_ratio = float(self._config.data.split.test_ratio)

            if strategy == 'stratified':
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=self._seed)
                train_idx, holdout_idx = next(splitter.split(np.zeros(len(targets)), targets))

                # split holdout into val and test
                if test_ratio > 0:
                    holdout_targets = [targets[i] for i in holdout_idx]
                    second = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio / (val_ratio + test_ratio), random_state=self._seed)
                    val_idx_rel, test_idx_rel = next(second.split(np.zeros(len(holdout_targets)), holdout_targets))
                    val_idx = [holdout_idx[i] for i in val_idx_rel]
                    test_idx = [holdout_idx[i] for i in test_idx_rel]
                else:
                    val_idx = holdout_idx
                    test_idx = []

            elif strategy == 'grouped':
                group_key = self._config.data.split.group_key
                if group_key is None:
                    raise ValueError('group_key must be provided for grouped split')
                # Extract group ids from dataset samples' paths
                groups = [Path(p).parent.name for p, _ in full_ds.samples]
                splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=self._seed)
                train_idx, holdout_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))

                if test_ratio > 0:
                    holdout_groups = [groups[i] for i in holdout_idx]
                    second = GroupShuffleSplit(n_splits=1, test_size=test_ratio / (val_ratio + test_ratio), random_state=self._seed)
                    val_idx_rel, test_idx_rel = next(second.split(np.zeros(len(holdout_groups)), groups=holdout_groups))
                    val_idx = [holdout_idx[i] for i in val_idx_rel]
                    test_idx = [holdout_idx[i] for i in test_idx_rel]
                else:
                    val_idx = holdout_idx
                    test_idx = []

            else:
                raise ValueError(f'Unsupported split strategy: {strategy}')

            # Build subsets with appropriate transforms
            train_subset = Subset(full_ds, train_idx)
            # override transform for val/test to deterministic
            val_ds = Subset(ImageFolder(str(train_dir), transform=self._augment_factory.build_val_test()), val_idx)
            test_ds = Subset(ImageFolder(str(train_dir), transform=self._augment_factory.build_val_test()), test_idx) if len(test_idx) > 0 else None

            self._train_dataset = train_subset
            self._val_dataset = val_ds
            self._test_dataset = test_ds

        if test_dir and self._test_dataset is None:
            self._test_dataset = ImageFolder(str(test_dir), transform=self._augment_factory.build_val_test())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=self._pin_memory, collate_fn=self._collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_dataset is None:
            return None
        return DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=self._pin_memory)

    def _collate_fn(self, batch):
        # Identity collate; mixup/cutmix should be applied in a separate batch-level hook
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.long)
        return images, targets
