from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from umvad.data import RealIADMultiViewDataset  # noqa: E402
from umvad.engine.trainer import IDIFTrainer, TrainerConfig  # noqa: E402
from umvad.models import IDIFModel  # noqa: E402
from umvad.utils.config import load_config  # noqa: E402


def build_datasets(cfg: Dict[str, Any]) -> tuple[RealIADMultiViewDataset, RealIADMultiViewDataset]:
    data_cfg = cfg["data"]
    image_root = data_cfg["image_root"]
    json_root = data_cfg["json_root"]
    categories = data_cfg.get("categories")
    expected_views = tuple(data_cfg.get("expected_views", [1, 2, 3, 4, 5]))
    image_size = int(data_cfg.get("image_size", 256))
    normalize = bool(data_cfg.get("normalize", False))
    drop_incomplete = bool(data_cfg.get("drop_incomplete", True))

    common = dict(
        image_root=image_root,
        json_root=json_root,
        categories=categories,
        expected_views=expected_views,
        image_size=image_size,
        normalize=normalize,
        include_masks=True,
        drop_incomplete=drop_incomplete,
    )
    train_ds = RealIADMultiViewDataset(split="train", **common)
    val_ds = RealIADMultiViewDataset(split="test", **common)
    return train_ds, val_ds


def build_loaders(cfg: Dict[str, Any], train_ds: RealIADMultiViewDataset, val_ds: RealIADMultiViewDataset):
    loader_cfg = cfg.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 8))
    num_workers = int(loader_cfg.get("num_workers", 4))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(cfg: Dict[str, Any]) -> IDIFModel:
    model_cfg = cfg.get("model", {})
    return IDIFModel(
        num_views=int(model_cfg.get("num_views", 5)),
        input_size=int(model_cfg.get("input_size", 256)),
        cb_latent_channels=int(model_cfg.get("cb_latent_channels", 64)),
        voxel_shape=tuple(model_cfg.get("voxel_shape", [4, 8, 8])),
        num_fusion_blocks=int(model_cfg.get("num_fusion_blocks", 2)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
        teacher_pretrained=bool(model_cfg.get("teacher_pretrained", True)),
        student_encoder_pretrained=bool(model_cfg.get("student_encoder_pretrained", False)),
        use_cb=bool(model_cfg.get("use_cb", True)),
        fusion_mode=str(model_cfg.get("fusion_mode", "ivc")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IDIF on Real-IAD")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_loaders(cfg, train_ds, val_ds)
    model = build_model(cfg)

    train_cfg_dict = cfg.get("trainer", {})
    trainer_cfg = TrainerConfig(**train_cfg_dict)
    trainer = IDIFTrainer(model, config=trainer_cfg)

    # Snapshot config used for this run.
    save_dir = Path(trainer_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    result = trainer.fit(train_loader, val_loader=val_loader)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
