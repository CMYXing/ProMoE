from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_prototypes(path: str | Path) -> tuple[list[str], np.ndarray]:
    raw = load_json(path)
    names = list(raw.keys())
    # JSON key order defines expert order; keep it stable for checkpoint compatibility.
    values = np.asarray([raw[name] for name in names], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Prototype file must contain a 2D table, got shape {values.shape}.")
    return names, values


@dataclass
class PhysiologicalAtlas:
    """Deterministic atlas that maps organ occupancy and tracer type to PE2 vectors."""

    organ_labels: dict[str, int]
    uptake_atlas: dict[str, dict[str, dict[str, Any]]]
    structure_order: list[str]

    @classmethod
    def from_files(
        cls,
        organ_label_path: str | Path,
        uptake_atlas_path: str | Path,
        structure_order_path: str | Path | None = None,
    ) -> "PhysiologicalAtlas":
        organ_labels = load_json(organ_label_path)
        uptake_atlas = load_json(uptake_atlas_path)
        if structure_order_path is None:
            structure_order = infer_structure_order(uptake_atlas)
        else:
            structure_order = load_json(structure_order_path)
        return cls(organ_labels=organ_labels, uptake_atlas=uptake_atlas, structure_order=structure_order)

    @property
    def dim(self) -> int:
        return len(self.structure_order)

    def organ_ratios(self, organ_mask: np.ndarray) -> dict[str, float]:
        organ_mask = np.asarray(organ_mask)
        foreground = organ_mask != int(self.organ_labels.get("background", 0))
        foreground_count = int(foreground.sum())
        ratios: dict[str, float] = {}
        for organ_name, label_value in self.organ_labels.items():
            if organ_name == "background":
                continue
            count = int((organ_mask == int(label_value)).sum())
            ratios[organ_name] = count / foreground_count if foreground_count and count else 0.0
        return ratios

    def encode(self, organ_mask: np.ndarray, tracer: str) -> np.ndarray:
        tracer = normalize_tracer_name(tracer)
        ratios = self.organ_ratios(organ_mask)
        # PE2 uses structure_order, not organ label order; this must match prototypes.
        uptake_stats = {name: {"ratio": 0.0, "uptake": 0.0} for name in self.structure_order}

        for organ_name, details in self.uptake_atlas.items():
            ratio = float(ratios.get(organ_name, 0.0))
            for structure_name, info in details.items():
                if structure_name not in uptake_stats:
                    continue
                uptake_stats[structure_name]["ratio"] += float(info.get("RATIO", 0.0)) * ratio
                if ratio > 0:
                    uptake = info.get("UPTAKE", {})
                    if tracer not in uptake:
                        raise KeyError(f"Tracer {tracer!r} is not present in atlas entry {organ_name}/{structure_name}.")
                    level = int(uptake[tracer])
                    # Atlas levels are 0-5; the paper uses positive support for present structures.
                    uptake_stats[structure_name]["uptake"] = max(
                        uptake_stats[structure_name]["uptake"],
                        (level + 1) / 6.0,
                    )

        return np.asarray([uptake_stats[name]["uptake"] for name in self.structure_order], dtype=np.float32)


class PrototypeRouter(nn.Module):
    """Prototype-driven expert routing with anatomical and physiological similarity."""

    def __init__(
        self,
        num_experts: int,
        text_dim: int,
        temperature: float = 0.1,
        physiological_gamma: float = 2.0,
        threshold: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.text_dim = int(text_dim)
        self.temperature = float(temperature)
        self.physiological_gamma = float(physiological_gamma)
        self.threshold = 1.0 / self.num_experts if threshold is None else float(threshold)
        self.eps = float(eps)
        self.prototypes = nn.Parameter(torch.randn(self.num_experts, self.text_dim))

    def load_predefined(self, values: np.ndarray | torch.Tensor, freeze: bool = False) -> None:
        tensor = torch.as_tensor(values, dtype=self.prototypes.dtype, device=self.prototypes.device)
        if tensor.shape != self.prototypes.shape:
            raise ValueError(f"Expected prototypes with shape {tuple(self.prototypes.shape)}, got {tuple(tensor.shape)}.")
        self.prototypes = nn.Parameter(tensor.clone(), requires_grad=not freeze)

    def forward(self, pe2: torch.Tensor) -> dict[str, torch.Tensor]:
        if pe2.ndim != 2:
            raise ValueError(f"PE2 embeddings must have shape [B, D], got {tuple(pe2.shape)}.")
        if pe2.shape[1] != self.text_dim:
            raise ValueError(f"PE2 dimension mismatch: expected {self.text_dim}, got {pe2.shape[1]}.")

        # Anatomical alignment is Dice over nonzero PE2/prototype supports.
        support_x = (pe2 > 0).to(pe2.dtype)
        support_p = (self.prototypes > 0).to(pe2.dtype)
        intersection = torch.einsum("bd,kd->bk", support_x, support_p)
        denom = support_x.sum(dim=1, keepdim=True) + support_p.sum(dim=1).unsqueeze(0) + self.eps
        anatomical = (2.0 * intersection / denom).clamp_min(0.0)

        # Physiological consistency is cosine similarity only on mutual support.
        mutual = support_x.unsqueeze(1) * support_p.unsqueeze(0)
        x_masked = pe2.unsqueeze(1) * mutual
        p_masked = self.prototypes.unsqueeze(0) * mutual
        dot = (x_masked * p_masked).sum(dim=-1)
        norm_x = torch.sqrt((x_masked.square()).sum(dim=-1) + self.eps)
        norm_p = torch.sqrt((p_masked.square()).sum(dim=-1) + self.eps)
        physiological = (dot / (norm_x * norm_p + self.eps)).clamp_min(0.0)
        joint = anatomical * physiological.pow(self.physiological_gamma)

        # Threshold before softmax as described in the paper; keep top-1 to avoid empty routing.
        valid_support = support_x.sum(dim=1) > 0
        keep = joint >= self.threshold
        keep = keep | _top1_mask(joint)
        keep = torch.where(valid_support.unsqueeze(1), keep, torch.ones_like(keep, dtype=torch.bool))
        logits = joint / max(self.temperature, self.eps)
        logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        return {
            "weights": weights,
            "joint_similarity": joint,
            "anatomical_similarity": anatomical,
            "physiological_similarity": physiological,
            "active_mask": keep,
        }


def infer_structure_order(uptake_atlas: dict[str, dict[str, Any]]) -> list[str]:
    order: list[str] = []
    for details in uptake_atlas.values():
        for structure_name in details:
            if structure_name not in order:
                order.append(structure_name)
    return order


def normalize_tracer_name(tracer: str) -> str:
    tracer = str(tracer).upper()
    if "PSMA" in tracer:
        return "PSMA"
    if "FAPI" in tracer:
        return "FAPI"
    if "CD70" in tracer:
        return "CD70"
    if "FDG" in tracer:
        return "FDG"
    return tracer


def _top1_mask(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.argmax(dim=-1, keepdim=True)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(1, indices, True)
