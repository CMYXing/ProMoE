from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .io import crop_with_padding, ensure_shape, load_volume, normalize, random_center, random_center_in_box, read_csv, resolve_path
from .pe2 import PhysiologicalAtlas, normalize_tracer_name


class ProMoEDataset(Dataset):
    """Patch dataset for aligned PET, CT, lesion mask, organ mask, and tracer CSV rows."""

    def __init__(
        self,
        csv_path: str | Path,
        root: str | Path,
        atlas: PhysiologicalAtlas,
        data_config: dict[str, Any],
        normalization_config: dict[str, Any],
    ):
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.rows = read_csv(self.csv_path)
        if not self.rows:
            raise ValueError(f"No rows found in {self.csv_path}.")
        self.atlas = atlas
        self.data_config = data_config
        self.normalization_config = normalization_config
        self.crop_size = tuple(int(v) for v in data_config.get("crop_size", [96, 96, 96]))
        self.pet_channels = list(data_config.get("pet_channels", ["pet_image", "sampling"]))
        self.ct_windows = list(data_config.get("ct_windows", ["soft_tissue", "lung", "bone"]))
        self.sampling_method = str(data_config.get("sampling_method", "mixed")).lower()
        self.mixed_rate = float(data_config.get("mixed_rate", 0.5))
        self.augment = data_config.get("augment", {})
        self.cases = [resolve_case(row, self.root) for row in self.rows]
        self.tracers = [case["tracer"] for case in self.cases]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index]
        pet_ref = load_volume(case["pet"], dtype=np.float32)
        pet = pet_ref.data
        ct = ensure_shape(load_volume(case["ct"], dtype=np.float32).data, pet.shape, order=1)
        seg = ensure_shape(load_volume(case["segmentation"], dtype=np.float32).data, pet.shape, order=0)
        organ = ensure_shape(load_volume(case["organ"], dtype=np.float32).data, pet.shape, order=0).astype(np.int16)
        sampling = None
        if case.get("sampling") is not None:
            sampling = ensure_shape(load_volume(case["sampling"], dtype=np.float32).data, pet.shape, order=0)

        # Training crops are sampled from the lesion/sampling mask, then PE2 is computed on the crop.
        center = self._sample_center(sampling, seg, pet.shape)
        pet_crop = crop_with_padding(pet, center, self.crop_size, pad_value=0.0)
        ct_crop = crop_with_padding(ct, center, self.crop_size, pad_value=-1024.0)
        seg_crop = crop_with_padding(seg, center, self.crop_size, pad_value=0.0)
        organ_crop = crop_with_padding(organ, center, self.crop_size, pad_value=0)
        sampling_crop = None if sampling is None else crop_with_padding(sampling, center, self.crop_size, pad_value=0.0)

        pet_crop, ct_crop, seg_crop, organ_crop, sampling_crop = self._augment(pet_crop, ct_crop, seg_crop, organ_crop, sampling_crop)

        tracer = case["tracer"]
        pet_tensor = torch.from_numpy(self._build_pet_channels(pet_crop, sampling_crop, tracer))
        ct_tensor = torch.from_numpy(self._build_ct_channels(ct_crop))
        pe2 = torch.from_numpy(self.atlas.encode(organ_crop, tracer))
        mask = torch.from_numpy((seg_crop > 0).astype(np.int64, copy=False))

        return {
            "pet": pet_tensor,
            "ct": ct_tensor,
            "pe2": pe2,
            "mask": mask,
            "tracer": tracer,
            "case_id": case["case_id"],
        }

    def _sample_center(self, sampling: np.ndarray | None, seg: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        source = sampling if sampling is not None and np.any(sampling > 0) else (seg > 0)
        if self.sampling_method == "global":
            return random_center(None, shape)
        if self.sampling_method == "mask":
            return random_center(source, shape)
        if self.sampling_method == "box":
            return random_center_in_box(source, shape)
        if self.sampling_method == "mixed":
            if np.random.rand() < self.mixed_rate:
                return random_center(source, shape)
            return random_center_in_box(source, shape)
        raise ValueError(f"Unknown sampling_method: {self.sampling_method}.")

    def _augment(
        self,
        pet: np.ndarray,
        ct: np.ndarray,
        seg: np.ndarray,
        organ: np.ndarray,
        sampling: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if not self.augment.get("random_flip", False):
            return pet, ct, seg, organ, sampling
        axes = self.augment.get("flip_axes", [0, 1])
        arrays = [pet, ct, seg, organ]
        if sampling is not None:
            arrays.append(sampling)
        for axis in axes:
            if np.random.rand() < 0.5:
                arrays = [np.flip(array, axis=int(axis)).copy() for array in arrays]
        if sampling is None:
            return arrays[0], arrays[1], arrays[2], arrays[3], None
        return arrays[0], arrays[1], arrays[2], arrays[3], arrays[4]

    def _build_pet_channels(self, pet: np.ndarray, sampling: np.ndarray | None, tracer: str) -> np.ndarray:
        tracer = normalize_tracer_name(tracer)
        pet_norm_config = self.normalization_config["pet"].get(tracer, self.normalization_config["pet"].get("FDG"))
        channels = []
        for name in self.pet_channels:
            if name == "pet_image":
                channels.append(normalize(pet, **pet_norm_config))
            elif name == "sampling":
                # The second PET channel follows the internal experiment: a coarse hot-region mask.
                if sampling is None:
                    sampling = (pet > 2.5).astype(np.float32)
                channels.append((sampling > 0).astype(np.float32))
            else:
                raise ValueError(f"Unknown PET channel: {name}.")
        return np.stack(channels, axis=0).astype(np.float32, copy=False)

    def _build_ct_channels(self, ct: np.ndarray) -> np.ndarray:
        channels = []
        for window in self.ct_windows:
            channels.append(normalize(ct, **self.normalization_config["ct"][window]))
        return np.stack(channels, axis=0).astype(np.float32, copy=False)


def resolve_case(row: dict[str, str], root: Path) -> dict[str, Any]:
    # Preferred public schema: pet_path, ct_path, lesion_mask, sampling_mask, organ_mask, tracer.
    pet = first_present(row, ["pet_path", "pet_image"])
    ct = first_present(row, ["ct_path", "ct_image"])
    segmentation = first_present(row, ["lesion_mask", "segmentation", "label"])
    sampling = first_present(row, ["sampling_mask", "sampling"], required=False)
    organ = first_present(row, ["organ_mask", "organ_path", "organ"], required=False)
    if organ is None:
        raise ValueError("Each row must provide an organ mask via organ_mask or organ_path.")
    tracer = normalize_tracer_name(row.get("tracer") or row.get("pet_tracer") or infer_tracer_from_row(row))
    case_id = row.get("case_id") or row.get("id") or Path(pet).parent.name
    return {
        "pet": resolve_path(pet, root),
        "ct": resolve_path(ct, root),
        "segmentation": resolve_path(segmentation, root),
        "sampling": resolve_path(sampling, root),
        "organ": resolve_path(organ, root),
        "tracer": tracer,
        "case_id": case_id,
    }


def first_present(row: dict[str, str], keys: list[str], required: bool = True) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "nan"):
            return value
    if required:
        raise ValueError(f"Missing required column. Expected one of: {keys}.")
    return None


def infer_tracer_from_row(row: dict[str, str]) -> str:
    joined = " ".join(str(v) for v in row.values()).upper()
    for tracer in ["CD70", "PSMA", "FAPI", "FDG"]:
        if tracer in joined:
            return tracer
    raise ValueError("Cannot infer tracer from row. Add a pet_tracer column with FDG/PSMA/FAPI/CD70.")


def make_weighted_sampler(dataset: ProMoEDataset, tracer_weights: dict[str, float] | None) -> WeightedRandomSampler | None:
    if not tracer_weights:
        return None
    weights = [float(tracer_weights.get(tracer, 1.0)) for tracer in dataset.tracers]
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True)
