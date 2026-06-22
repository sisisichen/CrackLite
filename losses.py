from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_bchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask.float()


def dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = _ensure_bchw(probs)
    targets = _ensure_bchw(targets)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def soft_erode(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] != 1:
        raise ValueError("soft_erode expects a single-channel mask.")
    erode_h = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    erode_w = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(erode_h, erode_w)


def soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def soft_open(mask: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(mask))


def soft_skeletonize(mask: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    mask = _ensure_bchw(mask).clamp(0.0, 1.0)
    opened = soft_open(mask)
    skeleton = F.relu(mask - opened)

    for _ in range(iterations):
        mask = soft_erode(mask)
        opened = soft_open(mask)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton.clamp(0.0, 1.0)


def cldice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = _ensure_bchw(probs).clamp(0.0, 1.0)
    targets = _ensure_bchw(targets).clamp(0.0, 1.0)
    pred_skeleton = soft_skeletonize(probs, iterations=iterations)
    target_skeleton = soft_skeletonize(targets, iterations=iterations)

    dims = (1, 2, 3)
    topological_precision = (pred_skeleton * targets).sum(dim=dims) / (
        pred_skeleton.sum(dim=dims) + eps
    )
    topological_sensitivity = (target_skeleton * probs).sum(dim=dims) / (
        target_skeleton.sum(dim=dims) + eps
    )
    cldice = (
        2.0
        * topological_precision
        * topological_sensitivity
        + eps
    ) / (topological_precision + topological_sensitivity + eps)
    return 1.0 - cldice.mean()


def boundary_target(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    mask = _ensure_bchw(mask).clamp(0.0, 1.0)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


class HybridCrackLoss(nn.Module):
    """
    Manuscript loss:
    L = BCE + Dice + 0.5 * clDice + 0.3 * Lsk + 0.3 * Lbd.
    """

    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_cldice: float = 0.5,
        lambda_centerline: float = 0.3,
        lambda_boundary: float = 0.3,
        skeleton_iterations: int = 20,
        boundary_radius: int = 2,
    ):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_cldice = lambda_cldice
        self.lambda_centerline = lambda_centerline
        self.lambda_boundary = lambda_boundary
        self.skeleton_iterations = skeleton_iterations
        self.boundary_radius = boundary_radius

    def forward(
        self,
        outputs: torch.Tensor | dict,
        targets: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        targets = _ensure_bchw(targets).clamp(0.0, 1.0)

        if isinstance(outputs, dict):
            logits = outputs["out"]
            aux_outputs = outputs.get("aux", {})
        else:
            logits = outputs
            aux_outputs = {}

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(probs, targets)
        cldice = cldice_loss(
            probs,
            targets,
            iterations=self.skeleton_iterations,
        )

        loss = (
            self.lambda_bce * bce
            + self.lambda_dice * dice
            + self.lambda_cldice * cldice
        )

        centerline_loss = logits.new_tensor(0.0)
        boundary_loss = logits.new_tensor(0.0)
        if aux_outputs:
            centerline = soft_skeletonize(
                targets,
                iterations=self.skeleton_iterations,
            ).detach()
            boundary = boundary_target(targets, radius=self.boundary_radius).detach()
            centerline_logits = aux_outputs.get("centerline")
            boundary_logits = aux_outputs.get("boundary")

            if centerline_logits is not None:
                centerline_loss = F.binary_cross_entropy_with_logits(
                    centerline_logits,
                    centerline,
                )
                loss = loss + self.lambda_centerline * centerline_loss

            if boundary_logits is not None:
                boundary_loss = F.binary_cross_entropy_with_logits(
                    boundary_logits,
                    boundary,
                )
                loss = loss + self.lambda_boundary * boundary_loss

        if not return_components:
            return loss

        components = {
            "total": float(loss.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "dice": float(dice.detach().cpu()),
            "cldice": float(cldice.detach().cpu()),
            "centerline": float(centerline_loss.detach().cpu()),
            "boundary": float(boundary_loss.detach().cpu()),
        }
        return loss, components
