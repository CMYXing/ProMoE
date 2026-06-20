from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict):
        return state
    raise ValueError(f"Checkpoint {path} is not a dictionary.")


def extract_model_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["model", "state_dict", "net"]:
        value = state.get(key)
        if isinstance(value, dict):
            return strip_module_prefix(value)
    if all(hasattr(v, "shape") for v in state.values()):
        return strip_module_prefix(state)
    raise ValueError("Could not find model weights in checkpoint.")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def load_partial(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], prefix: str | None = None) -> tuple[list[str], list[str]]:
    if prefix:
        prefix = prefix.rstrip(".") + "."
        state_dict = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    current = module.state_dict()
    compatible = {key: value for key, value in state_dict.items() if key in current and current[key].shape == value.shape}
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    skipped = [key for key in state_dict if key not in compatible]
    return list(missing), skipped + list(unexpected)

