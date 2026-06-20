from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import extract_model_state, load_partial, load_state
from .data import ProMoEDataset, make_weighted_sampler
from .io import choose_device, load_yaml, maybe_autocast, repo_root_from_config, resolve_path, save_yaml
from .losses import DiceFocalLoss, entropy_loss, prototype_alignment_loss
from .model import ProMoE
from .pe2 import PhysiologicalAtlas, load_prototypes


def train_from_config(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 0)))
    device = choose_device(str(cfg.get("device", "auto")))

    paths = cfg["paths"]
    output_dir = resolve_path(paths["output_dir"], root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "config.yaml")

    atlas = PhysiologicalAtlas.from_files(
        resolve_path(paths["organ_label"], root),
        resolve_path(paths["uptake_atlas"], root),
        resolve_path(paths.get("structure_order", "assets/pe2_structure_order.json"), root),
    )
    train_csv = resolve_path(paths["train_csv"], root)
    assert train_csv is not None
    dataset = ProMoEDataset(train_csv, root, atlas, cfg["data"], cfg["normalization"])
    # Weighted sampling compensates for the FDG-dominant multi-tracer distribution.
    sampler = make_weighted_sampler(dataset, cfg["data"].get("tracer_weights"))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)) and device.type == "cuda",
    )

    model = build_model(cfg, root, device)
    optimizer = build_optimizer(model, cfg["train"])
    scheduler = build_scheduler(optimizer, cfg["train"].get("scheduler", {}))
    loss_cfg = {key: value for key, value in cfg["loss"].items() if key != "aux_weights"}
    criterion = DiceFocalLoss(num_classes=cfg["model"]["num_classes"], **loss_cfg).to(device)

    start_epoch = 0
    resume = resolve_path(paths.get("resume"), root)
    if resume is not None:
        state = load_state(resume, map_location=device)
        model.load_state_dict(extract_model_state(state), strict=True)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state and scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", -1)) + 1

    amp = bool(cfg["train"].get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    epochs = int(cfg["train"]["epochs"])
    save_every = int(cfg["train"].get("save_every", 20))
    log_every = int(cfg["train"].get("log_every", 20))

    for epoch in range(start_epoch, epochs):
        model.train()
        if cfg["model"].get("freeze_ct", False):
            model.ct_module.eval()
        running = []
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for step, batch in enumerate(progress):
            pet = batch["pet"].to(device, non_blocking=True)
            ct = batch["ct"].to(device, non_blocking=True)
            pe2 = batch["pe2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(device, amp):
                outputs = model(pet, ct, pe2)
                loss, parts = compute_loss(outputs, mask, criterion, cfg["loss"].get("aux_weights", {}))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running.append(float(loss.detach().cpu()))
            if step % log_every == 0:
                progress.set_postfix(loss=f"{np.mean(running[-log_every:]):.4f}", **parts)

        if scheduler is not None:
            scheduler.step()
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pth", model, optimizer, scheduler, cfg, epoch)


def build_model(cfg: dict[str, Any], root: Path, device: torch.device) -> ProMoE:
    model_cfg = cfg["model"]
    threshold = model_cfg.get("routing_threshold")
    threshold = None if threshold in (None, "auto") else float(threshold)
    model = ProMoE(
        pet_in_channels=int(model_cfg["pet_in_channels"]),
        ct_in_channels=int(model_cfg["ct_in_channels"]),
        num_classes=int(model_cfg["num_classes"]),
        ct_num_classes=int(model_cfg["ct_num_classes"]),
        text_dim=int(model_cfg["text_dim"]),
        num_experts=int(model_cfg["num_experts"]),
        temperature=float(model_cfg["temperature"]),
        physiological_gamma=float(model_cfg["physiological_gamma"]),
        routing_threshold=threshold,
    ).to(device)

    prototype_path = resolve_path(cfg["paths"].get("expert_prototypes"), root)
    if prototype_path is not None:
        _, prototypes = load_prototypes(prototype_path)
        # Release default uses fixed K=12 prototypes; unfreeze in config for prototype learning.
        model.router.load_predefined(prototypes, freeze=bool(model_cfg.get("freeze_expert_prototypes", False)))

    ct_checkpoint = resolve_path(cfg["paths"].get("ct_checkpoint"), root)
    if ct_checkpoint is not None:
        state = load_state(ct_checkpoint, map_location=device)
        missing, skipped = load_partial(model.ct_module, extract_model_state(state), prefix="ct_module")
        if len(skipped) == len(extract_model_state(state)):
            missing, skipped = load_partial(model.ct_module, extract_model_state(state), prefix=None)

    if model_cfg.get("freeze_ct", False):
        model.freeze_ct()
    if model_cfg.get("freeze_expert_prototypes", False):
        model.freeze_expert_prototypes()
    return model


def build_optimizer(model: ProMoE, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = str(train_cfg.get("optimizer", "adam")).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    if name == "sgd":
        return torch.optim.SGD(params, lr=float(train_cfg["lr"]), momentum=0.9, weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    return torch.optim.Adam(params, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_cfg: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not scheduler_cfg:
        return None
    if str(scheduler_cfg.get("name", "step")).lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 1000)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    return None


def compute_loss(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    criterion: DiceFocalLoss,
    aux_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, str]]:
    seg = criterion(outputs["logits"], mask)
    ent = entropy_loss(outputs["weights"])
    proto = prototype_alignment_loss(outputs["weights"], outputs["joint_similarity"])
    # Auxiliary routing losses are configurable so the same code supports paper/default ablations.
    loss = seg + float(aux_weights.get("entropy", 0.0)) * ent + float(aux_weights.get("prototype", 0.0)) * proto
    return loss, {"seg": f"{seg.item():.4f}", "ent": f"{ent.item():.4f}", "proto": f"{proto.item():.4f}"}


def save_checkpoint(
    path: Path,
    model: ProMoE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cfg: dict[str, Any],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "config": cfg,
        },
        path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProMoE.")
    parser.add_argument("--config", default="configs/promoe_k12.yaml", help="Path to a ProMoE YAML config.")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
