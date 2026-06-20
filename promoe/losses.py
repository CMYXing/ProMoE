from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: list[float] | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.eps = float(eps)
        if class_weights is None:
            class_weights = [1.0] * self.num_classes
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights / weights.sum())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target = target.squeeze(1)
        target = target.long()
        probs = F.softmax(logits, dim=1)
        dice = self._dice_loss(probs, target)
        focal = self._focal_loss(probs, target)
        return self.dice_weight * dice + self.focal_weight * focal

    def _dice_loss(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = probs.new_tensor(0.0)
        reduce_dims = tuple(range(1, target.ndim))
        for class_idx in range(self.num_classes):
            pred = probs[:, class_idx]
            mask = (target == class_idx).to(probs.dtype)
            intersect = (pred * mask).sum(dim=reduce_dims)
            denom = pred.sum(dim=reduce_dims) + mask.sum(dim=reduce_dims)
            dice = (2.0 * intersect + self.eps) / (denom + self.eps)
            total = total + (1.0 - dice.mean()) * self.class_weights[class_idx].to(probs.device)
        return total

    def _focal_loss(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = probs.permute(0, *range(2, probs.ndim), 1).reshape(-1, self.num_classes)
        target = target.reshape(-1)
        pt = probs[torch.arange(probs.shape[0], device=probs.device), target].clamp_min(1e-8)
        alpha = self.class_weights.to(probs.device)[target]
        return (-alpha * (1.0 - pt).pow(self.focal_gamma) * pt.log()).mean()


def entropy_loss(weights: torch.Tensor) -> torch.Tensor:
    return -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()


def prototype_alignment_loss(weights: torch.Tensor, joint_similarity: torch.Tensor) -> torch.Tensor:
    return (weights * (1.0 - joint_similarity)).sum(dim=-1).mean()

