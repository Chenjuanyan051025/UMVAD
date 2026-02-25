from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


_VIEW_RE = re.compile(r"_C(\d)_")


def _pil_to_tensor(img: Image.Image) -> Tensor:
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    c = len(img.getbands())
    h, w = img.size[1], img.size[0]
    arr = arr.view(h, w, c).permute(2, 0, 1).contiguous()
    return arr.float() / 255.0


def _parse_view_id(filename: str) -> Optional[int]:
    m = _VIEW_RE.search(filename)
    return int(m.group(1)) if m else None


def _sample_signature(rel_path: str) -> str:
    path = PurePosixPath(rel_path)
    normalized_name = _VIEW_RE.sub("_C*_", path.name)
    # Include parent folder(s) to avoid collisions across OK/NG and different Sxxxx folders.
    parent = str(path.parent)
    return f"{parent}/{normalized_name}"


@dataclass
class _ViewEntry:
    image_path: Path
    mask_path: Optional[Path]
    view_id: int
    anomaly_class: str


@dataclass
class _GroupSample:
    category: str
    normal_class: str
    prefix: str
    group_key: str
    views: Dict[int, _ViewEntry]
    anomaly_classes: set[str]


class RealIADMultiViewDataset(Dataset):
    """Loads Real-IAD samples grouped into multi-view sets using JSON split files.

    Each JSON file is category-specific. This dataset groups per-view entries into a single
    sample using folder + filename signature (camera id removed from filename).
    """

    def __init__(
        self,
        image_root: str | Path,
        json_root: str | Path,
        split: str = "train",
        categories: Optional[Sequence[str]] = None,
        expected_views: Sequence[int] = (1, 2, 3, 4, 5),
        image_size: int = 256,
        normalize: bool = True,
        include_masks: bool = True,
        drop_incomplete: bool = False,
        return_paths: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.image_root = Path(image_root)
        self.json_root = Path(json_root)
        self.split = split
        self.expected_views = tuple(int(v) for v in expected_views)
        self.image_size = int(image_size)
        self.normalize = normalize
        self.include_masks = include_masks
        self.drop_incomplete = drop_incomplete
        self.return_paths = return_paths

        if not self.image_root.exists():
            raise FileNotFoundError(f"image_root not found: {self.image_root}")
        if not self.json_root.exists():
            raise FileNotFoundError(f"json_root not found: {self.json_root}")

        wanted = set(categories) if categories else None
        json_files = sorted(self.json_root.glob("*.json"))
        if wanted is not None:
            json_files = [p for p in json_files if p.stem in wanted]
            missing = sorted(wanted - {p.stem for p in json_files})
            if missing:
                raise FileNotFoundError(f"Missing category JSON(s): {missing}")

        self.samples: List[_GroupSample] = []
        for fp in json_files:
            self.samples.extend(self._load_category_split(fp))

        if self.drop_incomplete:
            self.samples = [
                s for s in self.samples if all(v in s.views for v in self.expected_views)
            ]

        if not self.samples:
            raise RuntimeError(
                f"No grouped samples found for split={split} under {self.json_root}"
            )

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _load_category_split(self, json_file: Path) -> List[_GroupSample]:
        with json_file.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        meta = obj.get("meta", {})
        prefix = meta.get("prefix", f"{json_file.stem}/")
        normal_class = meta.get("normal_class", "OK")
        items = obj.get(self.split, [])
        if not isinstance(items, list):
            raise ValueError(f"{json_file}: {self.split} must be a list")

        groups: Dict[str, _GroupSample] = {}
        for item in items:
            rel_img = item["image_path"]
            view_id = _parse_view_id(PurePosixPath(rel_img).name)
            if view_id is None:
                continue

            group_key = _sample_signature(rel_img)
            abs_img = self.image_root / prefix / rel_img
            rel_mask = item.get("mask_path")
            abs_mask = (self.image_root / prefix / rel_mask) if rel_mask else None

            if group_key not in groups:
                groups[group_key] = _GroupSample(
                    category=item.get("category", json_file.stem),
                    normal_class=normal_class,
                    prefix=prefix,
                    group_key=group_key,
                    views={},
                    anomaly_classes=set(),
                )

            g = groups[group_key]
            anomaly_class = item.get("anomaly_class", normal_class)
            g.views[view_id] = _ViewEntry(
                image_path=abs_img,
                mask_path=abs_mask,
                view_id=view_id,
                anomaly_class=anomaly_class,
            )
            g.anomaly_classes.add(anomaly_class)

        return sorted(groups.values(), key=lambda s: (s.category, s.group_key))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Tensor:
        img = Image.open(path).convert("RGB")
        if self.image_size > 0:
            img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        x = _pil_to_tensor(img)
        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    def _load_mask(self, path: Path) -> Tensor:
        mask = Image.open(path).convert("L")
        if self.image_size > 0:
            mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        m = _pil_to_tensor(mask).mean(dim=0, keepdim=True)
        return (m > 0).float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        v_images: List[Tensor] = []
        v_masks: List[Tensor] = []
        v_valid: List[bool] = []
        v_has_mask: List[bool] = []
        v_paths: List[str] = []
        v_mask_paths: List[str] = []

        for view_id in self.expected_views:
            entry = sample.views.get(view_id)
            if entry is None:
                x = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
                m = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
                v_valid.append(False)
                v_has_mask.append(False)
                v_paths.append("")
                v_mask_paths.append("")
            else:
                x = self._load_image(entry.image_path)
                if self.include_masks and entry.mask_path is not None and entry.mask_path.exists():
                    m = self._load_mask(entry.mask_path)
                    has_mask = True
                else:
                    m = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
                    has_mask = False
                v_valid.append(True)
                v_has_mask.append(has_mask)
                v_paths.append(str(entry.image_path))
                v_mask_paths.append(str(entry.mask_path) if entry.mask_path is not None else "")
            v_images.append(x)
            v_masks.append(m)

        images = torch.stack(v_images, dim=0)  # [V, 3, H, W]
        masks = torch.stack(v_masks, dim=0)  # [V, 1, H, W]
        view_valid = torch.tensor(v_valid, dtype=torch.bool)
        view_has_mask = torch.tensor(v_has_mask, dtype=torch.bool)

        view_anomaly_classes: List[str] = []
        for view_id in self.expected_views:
            entry = sample.views.get(view_id)
            view_anomaly_classes.append(entry.anomaly_class if entry is not None else sample.normal_class)
        unique_classes = sorted(set(view_anomaly_classes))
        if len(unique_classes) == 1:
            sample_anomaly_class = unique_classes[0]
        else:
            sample_anomaly_class = "MIXED"
        is_anomaly = any(c != sample.normal_class for c in view_anomaly_classes)
        out: Dict[str, Any] = {
            "images": images,
            "masks": masks,
            "view_valid": view_valid,
            "view_has_mask": view_has_mask,
            "view_ids": torch.tensor(self.expected_views, dtype=torch.long),
            "is_anomaly": torch.tensor(is_anomaly, dtype=torch.bool),
            "category": sample.category,
            "anomaly_class": sample_anomaly_class,
            "view_anomaly_class": view_anomaly_classes,
            "normal_class": sample.normal_class,
            "sample_key": sample.group_key,
        }
        if self.return_paths:
            out["image_paths"] = v_paths
            out["mask_paths"] = v_mask_paths
        return out


def summarize_dataset(ds: RealIADMultiViewDataset) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": len(ds),
        "split": ds.split,
        "expected_views": list(ds.expected_views),
        "categories": sorted({s.category for s in ds.samples}),
    }
    anomaly_counts: Dict[str, int] = {}
    for s in ds.samples:
        if len(s.anomaly_classes) == 1:
            key = next(iter(s.anomaly_classes))
        else:
            key = "MIXED"
        anomaly_counts[key] = anomaly_counts.get(key, 0) + 1
    summary["anomaly_counts"] = dict(sorted(anomaly_counts.items()))
    return summary
