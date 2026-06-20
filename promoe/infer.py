from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import extract_model_state, load_state
from .data import resolve_case
from .io import (
    choose_device,
    ensure_shape,
    load_volume,
    load_yaml,
    normalize,
    pad_to_shape,
    repo_root_from_config,
    resolve_path,
    save_mask,
    sliding_window_slices,
)
from .pe2 import PhysiologicalAtlas, normalize_tracer_name
from .train import build_model


def infer_single(
    cfg: dict[str, Any],
    root: Path,
    checkpoint: str | Path,
    pet_path: str | Path,
    ct_path: str | Path,
    organ_path: str | Path,
    tracer: str,
    output_path: str | Path,
    sampling_path: str | Path | None = None,
) -> Path:
    device = choose_device(str(cfg.get("device", "auto")))
    model = build_model(cfg, root, device)
    state = load_state(checkpoint, map_location=device)
    model.load_state_dict(extract_model_state(state), strict=False)
    model.eval()

    atlas = PhysiologicalAtlas.from_files(
        resolve_path(cfg["paths"]["organ_label"], root),
        resolve_path(cfg["paths"]["uptake_atlas"], root),
        resolve_path(cfg["paths"].get("structure_order", "assets/pe2_structure_order.json"), root),
    )
    pet_ref = load_volume(pet_path, dtype=np.float32)
    pet = pet_ref.data
    ct = ensure_shape(load_volume(ct_path, dtype=np.float32).data, pet.shape, order=1)
    organ = ensure_shape(load_volume(organ_path, dtype=np.float32).data, pet.shape, order=0).astype(np.int16)
    if sampling_path is not None:
        sampling = ensure_shape(load_volume(sampling_path, dtype=np.float32).data, pet.shape, order=0)
    else:
        # Fallback sampling channel mirrors the training convention when no mask is provided.
        threshold = float(cfg.get("infer", {}).get("sampling_threshold", 2.5))
        sampling = (pet > threshold).astype(np.float32)

    probs = sliding_window_predict(cfg, model, atlas, pet, ct, organ, sampling, normalize_tracer_name(tracer), device)
    mask = probs.argmax(axis=0).astype(np.uint8)
    save_mask(mask, pet_ref, output_path)
    return Path(output_path)


def sliding_window_predict(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    atlas: PhysiologicalAtlas,
    pet: np.ndarray,
    ct: np.ndarray,
    organ: np.ndarray,
    sampling: np.ndarray,
    tracer: str,
    device: torch.device,
) -> np.ndarray:
    infer_cfg = cfg.get("infer", {})
    patch_size = tuple(int(v) for v in infer_cfg.get("patch_size", cfg["data"].get("crop_size", [96, 96, 96])))
    overlap = float(infer_cfg.get("overlap", 0.5))
    num_classes = int(cfg["model"]["num_classes"])
    prob_acc = np.zeros((num_classes, *pet.shape), dtype=np.float32)
    count_acc = np.zeros(pet.shape, dtype=np.float32)

    for slices in tqdm(sliding_window_slices(pet.shape, patch_size, overlap), desc="inference", leave=False):
        pet_crop = pet[slices]
        ct_crop = ct[slices]
        organ_crop = organ[slices]
        sampling_crop = sampling[slices]
        pet_patch, valid = pad_to_shape(pet_crop, patch_size, value=0.0)
        ct_patch, _ = pad_to_shape(ct_crop, patch_size, value=-1024.0)
        organ_patch, _ = pad_to_shape(organ_crop, patch_size, value=0)
        sampling_patch, _ = pad_to_shape(sampling_crop, patch_size, value=0.0)

        pet_tensor = torch.from_numpy(build_pet_channels(pet_patch, sampling_patch, tracer, cfg)).unsqueeze(0).to(device)
        ct_tensor = torch.from_numpy(build_ct_channels(ct_patch, cfg)).unsqueeze(0).to(device)
        pe2 = torch.from_numpy(atlas.encode(organ_patch, tracer)).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(pet_tensor, ct_tensor, pe2)["prob"][0].detach().cpu().numpy()
        prob = prob[(slice(None), *valid)]
        # Overlapping patch probabilities are averaged in native PET grid space.
        prob_acc[(slice(None), *slices)] += prob
        count_acc[slices] += 1.0

    prob_acc /= np.maximum(count_acc[None], 1.0)
    return prob_acc


def build_pet_channels(pet: np.ndarray, sampling: np.ndarray, tracer: str, cfg: dict[str, Any]) -> np.ndarray:
    tracer = normalize_tracer_name(tracer)
    norm_cfg = cfg["normalization"]["pet"].get(tracer, cfg["normalization"]["pet"]["FDG"])
    channels = []
    for name in cfg["data"].get("pet_channels", ["pet_image", "sampling"]):
        if name == "pet_image":
            channels.append(normalize(pet, **norm_cfg))
        elif name == "sampling":
            channels.append((sampling > 0).astype(np.float32))
        else:
            raise ValueError(f"Unknown PET channel: {name}.")
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def build_ct_channels(ct: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    channels = []
    for window in cfg["data"].get("ct_windows", ["soft_tissue", "lung", "bone"]):
        channels.append(normalize(ct, **cfg["normalization"]["ct"][window]))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def infer_from_csv(config_path: str | Path, checkpoint: str | Path, csv_path: str | Path, output_dir: str | Path) -> None:
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    cfg = load_yaml(config_path)
    rows = []
    import csv as csv_module

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv_module.DictReader(f))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        case = resolve_case(row, root)
        output_path = output_dir / f"{case['case_id']}_promoe.nii.gz"
        infer_single(cfg, root, checkpoint, case["pet"], case["ct"], case["organ"], case["tracer"], output_path, case.get("sampling"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ProMoE inference.")
    parser.add_argument("--config", default="configs/promoe_k12.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pet")
    parser.add_argument("--ct")
    parser.add_argument("--organ")
    parser.add_argument("--sampling")
    parser.add_argument("--tracer")
    parser.add_argument("--output")
    parser.add_argument("--csv")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if args.csv:
        if not args.output_dir:
            raise ValueError("--output-dir is required with --csv.")
        infer_from_csv(args.config, args.checkpoint, args.csv, args.output_dir)
        return

    required = [args.pet, args.ct, args.organ, args.tracer, args.output]
    if any(v is None for v in required):
        raise ValueError("Single-case inference requires --pet --ct --organ --tracer --output.")
    config_path = Path(args.config).resolve()
    root = repo_root_from_config(config_path)
    cfg = load_yaml(config_path)
    infer_single(cfg, root, args.checkpoint, args.pet, args.ct, args.organ, args.tracer, args.output, args.sampling)


if __name__ == "__main__":
    main()
