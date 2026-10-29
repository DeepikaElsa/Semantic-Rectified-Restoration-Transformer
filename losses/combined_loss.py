"""
Combined training objective for SRRT.

    L_total = lambda_1 * L_L1
            + lambda_p * L_perceptual
            + lambda_f * L_flow          (only when Rectified Flow is enabled)
            + lambda_c * L_LCR           (only when LCR is enabled)

Every weight is YAML-configurable. The flow and LCR terms are additive only if
the caller supplies them, so the Restormer baseline trains on L1 + perceptual
without any dead terms contributing zero gradients.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class VGGPerceptualLoss(nn.Module):
    """
    L1 distance between frozen ImageNet-VGG19 features.

    Layers are the outputs of relu1_2, relu2_2, relu3_4 and relu4_4 -- the
    standard set for restoration/super-resolution perceptual losses.

    Inputs must be in [0, 1] (matching data/transforms.py) and are normalized
    to ImageNet statistics internally.

    Shapes:
        pred, target: (B, 3, H, W) -> scalar
    """

    FEATURE_LAYERS = (3, 8, 17, 26)

    def __init__(self, layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)) -> None:
        super().__init__()

        if len(layer_weights) != len(self.FEATURE_LAYERS):
            raise ValueError(
                f"layer_weights must have {len(self.FEATURE_LAYERS)} entries, got {len(layer_weights)}"
            )

        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad_(False)

        self.vgg = vgg
        self.layer_weights = layer_weights

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "VGGPerceptualLoss":
        # The frozen feature extractor must never leave eval mode, otherwise
        # model.train() in the training loop would re-enable dropout in VGG.
        super().train(mode)
        self.vgg.eval()
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std

        loss = pred.new_zeros(())
        layer_idx = 0
        last = self.FEATURE_LAYERS[-1]

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            with torch.no_grad():
                y = layer(y)

            if i == self.FEATURE_LAYERS[layer_idx]:
                loss = loss + self.layer_weights[layer_idx] * F.l1_loss(x, y)
                layer_idx += 1

            if i == last:
                break

        return loss


class LPIPSPerceptualLoss(nn.Module):
    """
    LPIPS as a *training* loss.

    WARNING: LPIPS is also the evaluation metric in this project. Optimizing it
    directly inflates the reported score and weakens any comparison. Prefer
    VGG19 for training. Provided only so the choice can be ablated.

    Shapes:
        pred, target: (B, 3, H, W) in [0, 1] -> scalar
    """

    def __init__(self, net: str = "alex") -> None:
        super().__init__()

        import lpips  # imported lazily: only needed when this variant is used

        self.metric = lpips.LPIPS(net=net)
        for param in self.metric.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "LPIPSPerceptualLoss":
        super().train(mode)
        self.metric.eval()
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects inputs scaled to [-1, 1].
        return self.metric(pred * 2 - 1, target * 2 - 1).mean()


def build_perceptual_loss(perceptual_type: str, lpips_net: str = "alex") -> nn.Module:
    if perceptual_type == "vgg19":
        return VGGPerceptualLoss()
    if perceptual_type == "lpips":
        return LPIPSPerceptualLoss(net=lpips_net)
    raise ValueError(f"perceptual_type must be 'vgg19' or 'lpips', got {perceptual_type!r}")


class CombinedLoss(nn.Module):
    """
    Args:
        l1_weight / perceptual_weight / flow_weight / consistency_weight:
            loss coefficients (lambda_1, lambda_p, lambda_f, lambda_c).
        perceptual_type: 'vgg19' (default) or 'lpips'.
        use_perceptual: set False to skip building the perceptual network
            entirely (useful for fast tests and memory probes).

    Returns from forward():
        (total, components) where `components` holds every individual term
        for per-loss logging and ablation.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        flow_weight: float = 0.1,
        consistency_weight: float = 0.1,
        perceptual_type: str = "vgg19",
        lpips_net: str = "alex",
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight

        self.l1 = nn.L1Loss()

        enable_perceptual = use_perceptual and perceptual_weight > 0
        self.perceptual = build_perceptual_loss(perceptual_type, lpips_net) if enable_perceptual else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        flow_loss: torch.Tensor | None = None,
        lcr_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}

        components["l1"] = self.l1(pred, target)
        total = self.l1_weight * components["l1"]

        if self.perceptual is not None:
            components["perceptual"] = self.perceptual(pred, target)
            total = total + self.perceptual_weight * components["perceptual"]

        if flow_loss is not None:
            components["flow"] = flow_loss
            total = total + self.flow_weight * flow_loss

        if lcr_loss is not None:
            components["lcr"] = lcr_loss
            total = total + self.consistency_weight * lcr_loss

        components["total"] = total
        return total, components

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CombinedLoss":
        loss_cfg = cfg["loss"]
        eval_cfg = cfg.get("evaluation", {})

        return cls(
            l1_weight=loss_cfg.get("l1_weight", 1.0),
            perceptual_weight=loss_cfg.get("perceptual_weight", 0.1),
            flow_weight=loss_cfg.get("flow_weight", 0.1),
            consistency_weight=loss_cfg.get("consistency_weight", 0.1),
            perceptual_type=loss_cfg.get("perceptual_type", "vgg19"),
            lpips_net=eval_cfg.get("lpips_net", "alex"),
        )
