from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

try:
    import timm
except Exception:
    timm = None

from src.config import FullConfig, ModelConfig, HeadConfig


class ModelFactory:
    """Factory to build model backbones and classification heads.

    This factory constructs a pre-trained Swin Transformer Tiny backbone from
    `timm`, replaces the head with a configurable shallow MLP, and applies the
    freezing/unfreezing schedule defined in the configuration.
    """

    def __init__(self, config: FullConfig):
        self._config = config
        self._device = config.experiment.device

    def build(self) -> nn.Module:
        """Builds and returns the composed model ready for training/inference.

        Returns:
            torch.nn.Module: the assembled model.
        """
        backbone = self._build_backbone()
        head = self._build_head(backbone)
        model = _SwinWithHead(backbone=backbone, head=head)
        self._apply_freezing_policy(model)
        return model.to(self._device)

    def _build_backbone(self) -> nn.Module:
        """Instantiate the backbone as defined in config using timm.

        Raises:
            RuntimeError: if `timm` is not available.
        """
        if timm is None:
            raise RuntimeError("timm is required to build the Swin backbone. Install it in the project's .venv using uv.")

        model_cfg: ModelConfig = self._config.model
        backbone = timm.create_model(model_cfg.backbone, pretrained=model_cfg.pretrained, features_only=False)

        # If stochastic depth is requested and supported by model, set it
        sd_cfg = model_cfg.stochastic_depth
        if sd_cfg and sd_cfg.enabled:
            # timm models often expose 'drop_path_rate'
            if hasattr(backbone, 'drop_path_rate'):
                setattr(backbone, 'drop_path_rate', float(sd_cfg.drop_prob))
            elif hasattr(backbone, 'default_cfg'):
                # best-effort: update default_cfg if present
                try:
                    backbone.default_cfg['drop_path_rate'] = float(sd_cfg.drop_prob)
                except Exception:
                    pass

        return backbone

    def _build_head(self, backbone: nn.Module) -> nn.Module:
        """Create the classification head (shallow MLP) and attach dropout.

        The head respects the configured hidden_dim and dropout. No constants
        are hard-coded; dimensions are inferred from the backbone's feature output.
        """
        head_cfg: HeadConfig = self._config.model.head
        num_classes = len(self._config.classes['labels'])

        feature_dim = self._infer_feature_dim(backbone)

        if head_cfg.type.lower() != 'mlp':
            raise ValueError(f"Unsupported head type: {head_cfg.type}")

        hidden_dim = head_cfg.hidden_dim or feature_dim

        layers: List[nn.Module] = []
        if hidden_dim > 0:
            layers.append(nn.Linear(feature_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if head_cfg.dropout and head_cfg.dropout > 0.0:
                layers.append(nn.Dropout(p=head_cfg.dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            # fallback to single linear layer
            layers.append(nn.Linear(feature_dim, num_classes))

        return nn.Sequential(*layers)

    def _infer_feature_dim(self, backbone: nn.Module) -> int:
        """Infer the final feature dimension output by the backbone.

        This implementation inspects common attributes used by timm models to
        determine feature dimension (e.g., `num_features`, `head.in_features`).
        """
        # Common attribute for timm models
        if hasattr(backbone, 'num_features') and getattr(backbone, 'num_features'):
            return int(getattr(backbone, 'num_features'))

        # Some models have head attribute
        head = getattr(backbone, 'head', None) or getattr(backbone, 'fc', None)
        if head is not None and hasattr(head, 'in_features'):
            return int(getattr(head, 'in_features'))

        # As a last resort, attempt a forward pass with dummy tensor to probe shape
        dummy = torch.randn(1, 3, self._config.data.image_size[0], self._config.data.image_size[1])
        backbone.eval()
        with torch.no_grad():
            try:
                out = backbone.forward_features(dummy) if hasattr(backbone, 'forward_features') else backbone.forward(dummy)
            except Exception:
                out = backbone(dummy)

        if isinstance(out, torch.Tensor):
            return int(out.shape[1])

        # If out is tuple/list, take last tensor
        if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[-1], torch.Tensor):
            return int(out[-1].shape[1])

        raise RuntimeError('Unable to infer feature dimension from backbone')

    def _apply_freezing_policy(self, model: nn.Module) -> None:
        """Apply the freezing/unfreezing policy described in the config to model.

        The config provides which stages to unfreeze. This method performs a
        conservative best-effort mapping from stage indices to module names used
        in timm's Swin implementation.
        """
        policy = self._config.model.freezing_policy

        # 1) Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False

        # 2) Unfreeze layers based on 'unfreeze_stages'
        # Map stage numbers to common timm swin module names
        stage_to_name = {
            1: 'layers.0',
            2: 'layers.1',
            3: 'layers.2',
            4: 'layers.3',
        }

        backbone = model.backbone
        for stage in policy.unfreeze_stages:
            name = stage_to_name.get(stage)
            if not name:
                continue
            submodule = _get_submodule(backbone, name)
            if submodule is None:
                continue
            for p in submodule.parameters():
                p.requires_grad = True

        # Always ensure head params are trainable
        for p in model.head.parameters():
            p.requires_grad = True


class _SwinWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Try backbone.forward_features if available
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        # If features are spatial, pool them
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))

        return self.head(features)


def _get_submodule(module: nn.Module, dotted_path: str) -> Optional[nn.Module]:
    """Utility to retrieve nested submodule by dotted path.

    Args:
        module: parent module
        dotted_path: dot-separated path like 'layers.3.blocks'

    Returns:
        The submodule or None if not found.
    """
    parts = dotted_path.split('.')
    sub = module
    for p in parts:
        if not hasattr(sub, p):
            return None
        sub = getattr(sub, p)
    return sub
