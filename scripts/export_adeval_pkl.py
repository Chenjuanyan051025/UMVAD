from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
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


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _build_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("loader", {})
    ds = RealIADMultiViewDataset(
        image_root=data_cfg["image_root"],
        json_root=data_cfg["json_root"],
        split=split,
        categories=data_cfg.get("categories"),
        expected_views=tuple(data_cfg.get("expected_views", [1, 2, 3, 4, 5])),
        image_size=int(data_cfg.get("image_size", 256)),
        normalize=bool(data_cfg.get("normalize", False)),
        include_masks=True,
        drop_incomplete=bool(data_cfg.get("drop_incomplete", True)),
        return_paths=True,
    )
    return DataLoader(
        ds,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
    )


def _build_model(cfg: Dict[str, Any]) -> IDIFModel:
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


def _restore_path_matrix(field: Any, batch_size: int, n_views: int) -> List[List[str]]:
    # Default PyTorch collation transposes List[str] fields into [V][B].
    if isinstance(field, (list, tuple)) and len(field) == n_views and all(
        isinstance(x, (list, tuple)) for x in field
    ):
        return [[str(field[v][b]) for v in range(n_views)] for b in range(batch_size)]

    if isinstance(field, (list, tuple)) and len(field) == batch_size and all(
        isinstance(x, (list, tuple)) for x in field
    ):
        return [[str(x) for x in row] for row in field]

    if batch_size == 1 and isinstance(field, (list, tuple)) and len(field) == n_views and all(
        isinstance(x, str) for x in field
    ):
        return [[str(x) for x in field]]

    raise TypeError(f"Unsupported collated path field structure: {type(field)!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export UMVAD predictions to ADEval pickle format")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--output", required=True, type=str, help="Path to output .pkl")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = _build_model(cfg)

    trainer_cfg_dict = dict(cfg.get("trainer", {}))
    trainer_cfg_dict.setdefault("epochs", 1)
    trainer = IDIFTrainer(model, config=TrainerConfig(**trainer_cfg_dict))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(trainer.device)
    trainer.model.eval()

    loader = _build_loader(cfg, split=args.split)

    result: Dict[str, tuple[np.ndarray, float, str | None]] = {}
    num_samples = 0
    num_views = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(dict(raw_batch), trainer.device)
            images: torch.Tensor = batch["images"]  # [B,V,3,H,W]
            view_valid: torch.Tensor = batch["view_valid"]  # [B,V]

            _clean_rgb, clean_norm = trainer._prepare_clean_rgb_and_norm(images)
            output = trainer.model(
                student_images=clean_norm,
                teacher_images=clean_norm,
                view_valid=view_valid,
                view_keep_mask=None,
            )

            anomaps = output.anomaly_maps.detach().cpu().to(torch.float32)  # [B,V,H,W]
            view_scores = anomaps.amax(dim=(2, 3))  # [B,V]
            valid_cpu = view_valid.detach().cpu()

            bsz, n_views = anomaps.shape[:2]
            image_paths = _restore_path_matrix(raw_batch["image_paths"], bsz, n_views)
            mask_paths = _restore_path_matrix(raw_batch["mask_paths"], bsz, n_views)

            for bi in range(bsz):
                num_samples += 1
                for vi in range(n_views):
                    if not bool(valid_cpu[bi, vi].item()):
                        continue
                    key = image_paths[bi][vi]
                    if not key:
                        continue
                    if key in result:
                        raise RuntimeError(f"Duplicate image path key while exporting: {key}")
                    mask_path = mask_paths[bi][vi].strip() if isinstance(mask_paths[bi][vi], str) else ""
                    result[key] = (
                        anomaps[bi, vi].numpy().astype(np.float32, copy=False),
                        float(view_scores[bi, vi].item()),
                        mask_path if mask_path else None,
                    )
                    num_views += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Exported ADEval pickle to {out_path} "
        f"(split={args.split}, grouped_samples={num_samples}, image_views={num_views})"
    )


if __name__ == "__main__":
    main()
