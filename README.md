# ProMoE

Prototype-Driven Mixture-of-Experts (ProMoE) for multi-tracer PET lesion segmentation. This repository provides the official code for ProMoE, presented in our MICCAI 2026 paper.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate promoe
```

Alternatively, install the Python dependencies with pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Assets

The reproducibility assets are under `assets/`:

- `organ_label.json`: CT organ label map used to compute patch-level anatomical occupancy.
- `uptake_atlas.json`: deterministic tracer-specific physiological uptake atlas, including FDG, PSMA, FAPI, and CD70 priors.
- `pe2_structure_order.json`: explicit 34-dimensional PE2 structure order.
- `expert_prototypes_k12.json`: K=12 expert prototypes used by the default configuration.

## Data CSV

Use aligned NIfTI files (`.nii`/`.nii.gz`) or NumPy volumes (`.npy`/`.npz`).

Required columns:

```csv
pet_path,ct_path,lesion_mask,sampling_mask,organ_mask,tracer
case001_pet.nii.gz,case001_ct.nii.gz,case001_label.nii.gz,case001_sampling.nii.gz,case001_organ.nii.gz,FDG
```

The loader also accepts semantic aliases `pet_image`, `ct_image`, `segmentation`, `sampling`, and `pet_tracer`.

`organ_mask` is the organ label map used by PE2. For inference, provide the precomputed organ mask or the released CT branch output converted to labels.

## Training

Edit `configs/promoe_k12.yaml`:

- `paths.train_csv`: training CSV.
- `paths.output_dir`: checkpoint/log directory.
- `paths.ct_checkpoint`: optional pretrained CT organ checkpoint.
- `data.tracer_weights`: tracer-balanced sampling weights.

Run:

```bash
python scripts/train.py --config configs/promoe_k12.yaml
```

Checkpoints are saved to `outputs/promoe_k12/checkpoints/epoch_XXXX.pth`.

## Inference

Single case:

```bash
python scripts/infer.py \
  --config configs/promoe_k12.yaml \
  --checkpoint outputs/promoe_k12/checkpoints/epoch_3000.pth \
  --pet case_pet.nii.gz \
  --ct case_ct.nii.gz \
  --organ case_organ.nii.gz \
  --tracer PSMA \
  --output case_promoe_mask.nii.gz
```

Batch CSV:

```bash
python scripts/infer.py \
  --config configs/promoe_k12.yaml \
  --checkpoint outputs/promoe_k12/checkpoints/epoch_3000.pth \
  --csv data/test.csv \
  --output-dir outputs/predictions
```

For CD70, pass `--tracer CD70`. No CD70 images or labels are required for training; inference uses the fixed CD70 physiological prior in `assets/uptake_atlas.json`.

## Routing Inspection

Inspect the active experts for a tracer/organ-mask pair:

```bash
python scripts/inspect_routing.py --organ case_organ.nii.gz --tracer CD70 --topk 5
```
