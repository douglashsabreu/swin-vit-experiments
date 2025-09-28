"""Project entrypoint wrapper delegating to the package runner.

This file enables running the training entrypoint directly with
`python main.py --config path/to/config.yaml` during development.
"""
from __future__ import annotations

import sys

from src.core.run import main as run_main


if __name__ == "__main__":
    run_main()
