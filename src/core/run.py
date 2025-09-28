from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.core.trainer import Trainer


def _parse_args():
    parser = argparse.ArgumentParser(description='Run training from config')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml/json')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and exit without running')
    return parser.parse_args()


def main():
    """CLI entrypoint: load config, instantiate Trainer and execute run.

    The function is intentionally minimal: configuration drives behavior and
    all heavy-lifting is delegated to `Trainer` and the factories.
    """
    args = _parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loaded config: {cfg.experiment.name}")

    if args.dry_run:
        logging.info("Dry run: configuration validated. Exiting.")
        return

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
