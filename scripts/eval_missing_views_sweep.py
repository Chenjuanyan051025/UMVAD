from __future__ import annotations

import argparse
import json
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


def _build(cfg: Dict[str, Any]) -> tuple[IDIFTrainer, DataLoader]:
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("loader", {})
    model_cfg = cfg.get("model", {})
    trainer_cfg = TrainerConfig(**cfg.get("trainer", {}))

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
    loader = DataLoader(
        ds,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    model = IDIFModel(
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
    trainer = IDIFTrainer(model, config=trainer_cfg)
    return trainer, loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-missing", type=int, default=4)
    ap.add_argument("--save-json", type=str, default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    trainer, loader = _build(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(trainer.device)

    rows: List[Dict[str, Any]] = []
    for missing in range(0, args.max_missing + 1):
        metrics = []
        for r in range(max(1, args.repeats)):
            summ = trainer.evaluate(loader, missing_views=missing, missing_seed=args.seed + 1000 * missing + r)
            metrics.append({"s_auroc": summ.s_auroc, "p_auroc": summ.p_auroc, "pro": summ.pro})
        row = {
            "missing_views": missing,
            "repeat": len(metrics),
            "s_auroc_values": [m["s_auroc"] for m in metrics],
            "p_auroc_values": [m["p_auroc"] for m in metrics],
            "pro_values": [m["pro"] for m in metrics],
        }
        row["s_auroc_mean"] = float(sum(row["s_auroc_values"]) / len(row["s_auroc_values"]))
        row["p_auroc_mean"] = float(sum(row["p_auroc_values"]) / len(row["p_auroc_values"]))
        row["pro_mean"] = float(sum(row["pro_values"]) / len(row["pro_values"]))
        rows.append(row)

    out = {"results": rows}
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if args.save_json:
        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
