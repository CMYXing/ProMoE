from __future__ import annotations

import csv
from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from scipy.ndimage import zoom


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def resolve_path(path: str | Path | None, root: str | Path) -> Path | None:
    if path in (None, "", "null"):
        return None
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


class Volume:
    def __init__(self, data: np.ndarray, affine: np.ndarray | None = None, header: Any | None = None):
        self.data = data
        self.affine = np.eye(4, dtype=np.float32) if affine is None else affine
        self.header = header


def load_volume(path: str | Path, dtype: np.dtype = np.float32) -> Volume:
    path = Path(path)
    if path.suffix == ".npy":
        return Volume(np.load(path).astype(dtype, copy=False))
    if path.suffix == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else sorted(data.files)[0]
        return Volume(data[key].astype(dtype, copy=False))
    if path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        image = nib.load(str(path))
        return Volume(np.asarray(image.get_fdata(dtype=np.float32), dtype=dtype), image.affine, image.header)
    raise ValueError(
        f"Unsupported image format: {path}. ProMoE release code supports .nii/.nii.gz/.npy/.npz. "
        "Please convert internal .image3dd files to NIfTI before release/use."
    )


def save_mask(mask: np.ndarray, reference: Volume, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    nib.save(image, str(output_path))


def ensure_shape(array: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    if tuple(array.shape) == tuple(shape):
        return array
    factors = [dst / src for dst, src in zip(shape, array.shape)]
    return zoom(array, factors, order=order).astype(array.dtype, copy=False)


def normalize(array: np.ndarray, mean: float, std: float, clip: bool = False) -> np.ndarray:
    out = (array.astype(np.float32) - float(mean)) / max(float(std), 1e-6)
    return np.clip(out, -1.0, 1.0) if clip else out


def crop_with_padding(array: np.ndarray, center: np.ndarray, size: tuple[int, int, int], pad_value: float = 0) -> np.ndarray:
    center = np.asarray(center, dtype=np.int64)
    size_arr = np.asarray(size, dtype=np.int64)
    start = center - size_arr // 2
    end = start + size_arr

    src_slices = []
    dst_slices = []
    for axis in range(3):
        src_start = max(int(start[axis]), 0)
        src_end = min(int(end[axis]), array.shape[axis])
        dst_start = src_start - int(start[axis])
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    out = np.full(tuple(size_arr), pad_value, dtype=array.dtype)
    out[tuple(dst_slices)] = array[tuple(src_slices)]
    return out


def random_center(mask: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray:
    if mask is not None and np.any(mask > 0):
        coords = np.argwhere(mask > 0)
        return coords[np.random.randint(0, len(coords))]
    return np.asarray([np.random.randint(0, dim) for dim in shape], dtype=np.int64)


def random_center_in_box(mask: np.ndarray | None, shape: tuple[int, int, int], padding: int = 10) -> np.ndarray:
    if mask is None or not np.any(mask > 0):
        return random_center(None, shape)
    coords = np.argwhere(mask > 0)
    low = np.maximum(coords.min(axis=0) - padding, 0)
    high = np.minimum(coords.max(axis=0) + padding + 1, np.asarray(shape))
    return np.asarray([np.random.randint(low[idx], max(high[idx], low[idx] + 1)) for idx in range(3)], dtype=np.int64)


def sliding_window_slices(
    shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    overlap: float,
) -> list[tuple[slice, slice, slice]]:
    starts_per_axis = []
    for dim, patch in zip(shape, patch_size):
        if dim <= patch:
            starts_per_axis.append([0])
            continue
        step = max(int(patch * (1.0 - overlap)), 1)
        starts = list(range(0, dim - patch + 1, step))
        if starts[-1] != dim - patch:
            starts.append(dim - patch)
        starts_per_axis.append(starts)
    slices = []
    for x in starts_per_axis[0]:
        for y in starts_per_axis[1]:
            for z in starts_per_axis[2]:
                slices.append((slice(x, min(x + patch_size[0], shape[0])), slice(y, min(y + patch_size[1], shape[1])), slice(z, min(z + patch_size[2], shape[2]))))
    return slices


def pad_to_shape(array: np.ndarray, shape: tuple[int, int, int], value: float = 0) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    out = np.full(shape, value, dtype=array.dtype)
    slices = tuple(slice(0, min(array.shape[idx], shape[idx])) for idx in range(3))
    out[slices] = array[slices]
    return out, slices


def choose_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=True)
    return torch.autocast(device_type=device.type, enabled=False)


def repo_root_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))
