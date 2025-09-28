from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.factories.model_factory import ModelFactory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Instantiate model from config')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml/json')
    return parser.parse_args()


def main() -> None:
    """Load config, build model, and print a brief parameter summary.

    This script is intended for quick verification that the model can be
    instantiated under the provided configuration.
    """
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))

    factory = ModelFactory(cfg)
    model = factory.build()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model built: {cfg.model.backbone}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Device: {cfg.experiment.device}")


if __name__ == '__main__':
    main()
