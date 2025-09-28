# swin-vit-experiments

This repository provides a configurable pipeline for transfer-learning image classification using a Swin Transformer Tiny backbone. The system is driven entirely by configuration files (YAML/JSON) — no hard-coded training constants.

## Features

- Swin-T backbone (via `timm`) with configurable MLP head
- Linear-probe → selective fine-tuning schedule
- Strong, configurable regularization (augmentations, label smoothing, weight decay, mixup/cutmix)
- Reproducible splits (stratified/grouped)
- AMP, gradient accumulation, clipping, scheduler strategies
- Export to ONNX/TorchScript (config-driven)

## Installation

Create a virtual environment and install dependencies using `uv` as the package manager (project uses a `.venv`):

```bash
python -m venv .venv
source .venv/bin/activate
uv install
```

Or install directly with `pip` using `pyproject.toml`:

```bash
pip install -e .
```

## Usage

Validate a config (dry run):

```bash
instantiate-model --config experiments/example.yaml
run-train --config experiments/train.yaml --dry-run
```

Run training:

```bash
run-train --config experiments/train.yaml
```

Quick model instantiation:

```bash
instantiate-model --config experiments/train.yaml
```

## Development

- Configs live in `experiments/` (create as needed)
- Add datasets as directory trees compatible with `torchvision.datasets.ImageFolder`
- Use `src/core/run.py` as the programmatic entrypoint for CI or scheduler integration
