from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def build_eval_dataset_and_loader(cfg: Dict[str, Any]) -> DataLoader:
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("loader", {})
    ds = RealIADMultiViewDataset(
        image_root=data_cfg["image_root"],
        json_root=data_cfg["json_root"],
        split="test",
        categories=data_cfg.get("categories"),
        expected_views=tuple(data_cfg.get("expected_views", [1, 2, 3, 4, 5])),
        image_size=int(data_cfg.get("image_size", 256)),
        normalize=bool(data_cfg.get("normalize", False)),
        include_masks=True,
        drop_incomplete=bool(data_cfg.get("drop_incomplete", True)),
    )
    return DataLoader(
        ds,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
    )


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
    parser = argparse.ArgumentParser(description="Evaluate IDIF checkpoints on Real-IAD")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--missing-views", type=int, default=0, help="Randomly hide N views per sample at test")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat eval with different missing-view seeds")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)

    trainer_cfg_dict = dict(cfg.get("trainer", {}))
    trainer_cfg_dict.setdefault("epochs", 1)
    trainer_cfg = TrainerConfig(**trainer_cfg_dict)
    trainer = IDIFTrainer(model, config=trainer_cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(trainer.device)

    loader = build_eval_dataset_and_loader(cfg)
    summaries = []
    for i in range(max(1, args.repeat)):
        seed_i = args.seed + i if args.missing_views > 0 else None
        summary = trainer.evaluate(loader, missing_views=args.missing_views, missing_seed=seed_i)
        summaries.append(summary)

    payload: Dict[str, Any]
    if len(summaries) == 1:
        s = summaries[0]
        payload = {
            "missing_views": args.missing_views,
            "repeat": 1,
            "s_auroc": s.s_auroc,
            "p_auroc": s.p_auroc,
            "pro": s.pro,
            "per_category": s.per_category,
        }
    else:
        s_vals = [s.s_auroc for s in summaries]
        p_vals = [s.p_auroc for s in summaries]
        pro_vals = [s.pro for s in summaries]
        payload = {
            "missing_views": args.missing_views,
            "repeat": len(summaries),
            "s_auroc_mean": statistics.mean(s_vals),
            "s_auroc_std": statistics.pstdev(s_vals) if len(s_vals) > 1 else 0.0,
            "p_auroc_mean": statistics.mean(p_vals),
            "p_auroc_std": statistics.pstdev(p_vals) if len(p_vals) > 1 else 0.0,
            "pro_mean": statistics.mean(pro_vals),
            "pro_std": statistics.pstdev(pro_vals) if len(pro_vals) > 1 else 0.0,
        }

    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    print(txt)
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
