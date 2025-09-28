from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, validator


class ExperimentConfig(BaseModel):
    name: str
    seed: int
    device: str
    mixed_precision: bool


class SplitConfig(BaseModel):
    strategy: str
    val_ratio: float
    test_ratio: float
    group_key: Optional[str] = None


class DataConfig(BaseModel):
    train_dir: str
    val_dir: Optional[str]
    test_dir: Optional[str]
    split: SplitConfig
    image_size: Tuple[int, int]
    normalization: str
    num_workers: int
    pin_memory: bool


class AugTrainItem(BaseModel):
    enabled: bool
    p: Optional[float] = None
    scale: Optional[Tuple[float, float]] = None
    ratio: Optional[Tuple[float, float]] = None
    degrees: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    hue: Optional[float] = None
    std: Optional[float] = None
    max_area: Optional[float] = None
    alpha: Optional[float] = None


class AugValTestItem(BaseModel):
    size: Tuple[int, int]
    interpolation: str
    enabled: Optional[bool] = True


class AugmentationsConfig(BaseModel):
    train: Dict[str, AugTrainItem]
    val_test: Dict[str, AugValTestItem]


class HeadConfig(BaseModel):
    type: str
    dropout: float
    hidden_dim: Optional[int] = None


class FreezingPolicy(BaseModel):
    phase1_linear_probe_epochs: int
    unfreeze_stages: List[int]
    progressive_unfreeze: Dict[str, Any]


class StochasticDepthConfig(BaseModel):
    enabled: bool
    drop_prob: float


class ModelConfig(BaseModel):
    backbone: str
    pretrained: bool
    head: HeadConfig
    freezing_policy: FreezingPolicy
    stochastic_depth: Optional[StochasticDepthConfig] = None


class OptimizerConfig(BaseModel):
    name: str
    lr_head: float
    lr_backbone: float
    betas: Tuple[float, float]
    weight_decay: float


class SchedulerParams(BaseModel):
    warmup_epochs: Optional[int]
    min_lr: Optional[float]
    factor: Optional[float]
    patience: Optional[int]
    monitor: Optional[str]
    mode: Optional[str]


class SchedulerConfig(BaseModel):
    name: str
    params: SchedulerParams


class GradientClippingConfig(BaseModel):
    enabled: bool
    max_norm: float


class AmpConfig(BaseModel):
    enabled: bool


class OptimizationConfig(BaseModel):
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    gradient_clipping: GradientClippingConfig
    amp: AmpConfig


class EarlyStoppingConfig(BaseModel):
    enabled: bool
    monitor: str
    mode: str
    patience: int
    restore_best: bool


class TrainingConfig(BaseModel):
    epochs_max: int
    batch_size: int
    grad_accum_steps: int
    early_stopping: EarlyStoppingConfig


class LossConfig(BaseModel):
    name: str
    label_smoothing: Optional[float]
    class_weights: Optional[Dict[str, Any]]


class TTAConfig(BaseModel):
    enabled: bool
    horizontal_flip: bool
    five_crop: bool


class ReportsConfig(BaseModel):
    confusion_matrix: bool
    classification_report: bool
    calibration_ece: bool
    attention_rollout_samples: int


class EvaluationConfig(BaseModel):
    tta: TTAConfig
    reports: ReportsConfig


class LoggingConfig(BaseModel):
    backend: str
    run_tags: List[str]
    save_dir: str


class CheckpointConfig(BaseModel):
    monitor: str
    mode: str
    save_top_k: int
    save_last: bool


class ExportConfig(BaseModel):
    onnx: Dict[str, Any]
    torchscript: Dict[str, Any]


class FullConfig(BaseModel):
    experiment: ExperimentConfig
    data: DataConfig
    classes: Dict[str, List[str]]
    augmentations: AugmentationsConfig
    model: ModelConfig
    optimization: OptimizationConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    checkpointing: CheckpointConfig
    export: ExportConfig

    @validator('experiment')
    def device_must_be_cuda_or_cpu(cls, v):
        if v.device not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda' or 'cpu'")
        return v


def load_config(path: str) -> FullConfig:
    """Load YAML/JSON config file and validate against schema.

    Args:
        path: Path to YAML or JSON configuration file.

    Returns:
        Validated FullConfig instance.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if p.suffix.lower() in ('.yaml', '.yml'):
        raw = yaml.safe_load(p.read_text())
    elif p.suffix.lower() == '.json':
        raw = json.loads(p.read_text())
    else:
        raise ValueError("Unsupported config format. Use .yaml, .yml or .json")

    return FullConfig.parse_obj(raw)
