from pathlib import Path
import argparse
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promoe.io import load_volume, load_yaml, repo_root_from_config, resolve_path
from promoe.pe2 import PhysiologicalAtlas, PrototypeRouter, load_prototypes, normalize_tracer_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ProMoE prototype routing for one organ mask.")
    parser.add_argument("--config", default="configs/promoe_k12.yaml")
    parser.add_argument("--organ", required=True)
    parser.add_argument("--tracer", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    root = repo_root_from_config(config_path)
    cfg = load_yaml(config_path)
    atlas = PhysiologicalAtlas.from_files(
        resolve_path(cfg["paths"]["organ_label"], root),
        resolve_path(cfg["paths"]["uptake_atlas"], root),
        resolve_path(cfg["paths"].get("structure_order", "assets/pe2_structure_order.json"), root),
    )
    names, prototypes = load_prototypes(resolve_path(cfg["paths"]["expert_prototypes"], root))
    router = PrototypeRouter(
        num_experts=cfg["model"]["num_experts"],
        text_dim=cfg["model"]["text_dim"],
        temperature=cfg["model"]["temperature"],
        physiological_gamma=cfg["model"]["physiological_gamma"],
    )
    router.load_predefined(prototypes, freeze=True)
    organ = load_volume(args.organ).data.astype("int16")
    pe2 = torch.from_numpy(atlas.encode(organ, normalize_tracer_name(args.tracer))).unsqueeze(0)
    with torch.no_grad():
        out = router(pe2)
    weights = out["weights"][0]
    topk = min(args.topk, len(names))
    values, indices = torch.topk(weights, k=topk)
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"{index:02d}\t{value:.4f}\t{names[index]}")


if __name__ == "__main__":
    main()
