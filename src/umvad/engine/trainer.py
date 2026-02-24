from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from umvad.data.pseudo_anomaly import PseudoAnomalySynthesizer
from umvad.engine.losses import LossBreakdown, compute_idif_loss
from umvad.engine.metrics import EvaluationSummary, summarize_metrics
from umvad.models import IDIFModel


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _normalize_rgb(x: Tensor) -> Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def _denormalize_rgb(x: Tensor) -> Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return x * std + mean


def _make_grad_scaler(device: torch.device, enabled: bool):
    """Compatibility wrapper across torch versions."""
    if not enabled:
        class _NoopScaler:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): return None
            def unscale_(self, opt): return None
        return _NoopScaler()

    if device.type != "cuda":
        class _NoopScaler:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): return None
            def unscale_(self, opt): return None
        return _NoopScaler()

    # torch >= 2.0 style
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=True)
        except TypeError:
            return amp_mod.GradScaler(enabled=True)

    # older torch style
    cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    if cuda_amp is not None and hasattr(cuda_amp, "GradScaler"):
        return cuda_amp.GradScaler(enabled=True)

    class _NoopScaler:
        def scale(self, x): return x
        def step(self, opt): opt.step()
        def update(self): return None
        def unscale_(self, opt): return None
    return _NoopScaler()


def _autocast_context(device: torch.device, enabled: bool):
    """Compatibility wrapper across torch versions."""
    if not enabled or device.type != "cuda":
        return nullcontext()

    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast(device_type=device.type, enabled=True)
        except TypeError:
            return amp_mod.autocast(enabled=True)

    cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    if cuda_amp is not None and hasattr(cuda_amp, "autocast"):
        return cuda_amp.autocast(enabled=True)

    return nullcontext()


def sample_view_keep_mask(view_valid: Tensor, drop_prob: float) -> Tensor:
    """Sample view-wise dropout mask while ensuring at least one valid view per sample."""
    if drop_prob <= 0:
        return view_valid.clone()
    b, v = view_valid.shape
    rand = torch.rand(b, v, device=view_valid.device)
    keep = (rand > drop_prob) & view_valid

    valid_counts = view_valid.sum(dim=1)
    keep_counts = keep.sum(dim=1)
    need_fix = (valid_counts > 0) & (keep_counts == 0)
    if need_fix.any():
        idxs = need_fix.nonzero(as_tuple=False).flatten()
        for bi in idxs.tolist():
            valid_idx = view_valid[bi].nonzero(as_tuple=False).flatten()
            chosen = valid_idx[torch.randint(0, len(valid_idx), (1,), device=view_valid.device)].item()
            keep[bi, chosen] = True
    return keep


@dataclass
class TrainerConfig:
    device: str = "cuda"
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 50
    grad_clip_norm: Optional[float] = None
    amp: bool = True
    save_dir: str = "runs/idif"
    log_interval: int = 20
    view_dropout_prob: float = 0.0
    inputs_are_normalized: bool = False
    distill_weight: float = 1.0
    ib_weight: float = 1.0
    ib_kl_weight: float = 1.0
    ib_recon_weight: float = 1.0
    scale_weights: Optional[Dict[str, float]] = None
    dtd_root: Optional[str] = None
    eval_every: int = 1
    seed: int = 42


class IDIFTrainer:
    def __init__(
        self,
        model: IDIFModel,
        *,
        config: TrainerConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = scheduler
        self.scaler = _make_grad_scaler(self.device, enabled=(config.amp and self.device.type == "cuda"))
        self.pseudo = PseudoAnomalySynthesizer(dtd_root=config.dtd_root)
        self.save_dir = _ensure_dir(config.save_dir)
        self.metrics_log_path = self.save_dir / "metrics.jsonl"

        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _prepare_clean_rgb_and_norm(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if self.cfg.inputs_are_normalized:
            clean_norm = images
            clean_rgb = _denormalize_rgb(images).clamp(0.0, 1.0)
        else:
            clean_rgb = images.clamp(0.0, 1.0)
            clean_norm = _normalize_rgb(clean_rgb)
        return clean_rgb, clean_norm

    def _train_step(self, batch: Mapping[str, Any]) -> tuple[LossBreakdown, Dict[str, float]]:
        self.model.train()
        batch = _to_device(batch, self.device)
        images: Tensor = batch["images"]
        view_valid: Tensor = batch["view_valid"]

        clean_rgb, clean_norm = self._prepare_clean_rgb_and_norm(images)
        pseudo_rgb, _pseudo_mask = self.pseudo(clean_rgb, view_valid=view_valid)
        pseudo_norm = _normalize_rgb(pseudo_rgb)

        view_keep_mask = sample_view_keep_mask(view_valid, self.cfg.view_dropout_prob)

        self.optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self.device, enabled=(self.cfg.amp and self.device.type == "cuda")):
            output = self.model(
                student_images=pseudo_norm,
                teacher_images=clean_norm,
                view_valid=view_valid,
                view_keep_mask=view_keep_mask,
            )
            losses = compute_idif_loss(
                output,
                distill_weight=self.cfg.distill_weight,
                ib_weight=self.cfg.ib_weight,
                ib_kl_weight=self.cfg.ib_kl_weight,
                ib_recon_weight=self.cfg.ib_recon_weight,
                scale_weights=self.cfg.scale_weights,
            )

        self.scaler.scale(losses.total).backward()
        if self.cfg.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.student.parameters(), self.cfg.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        stats = {
            "loss_total": float(losses.total.detach().cpu()),
            "loss_distill": float(losses.distill.detach().cpu()),
            "loss_ib": float(losses.ib.detach().cpu()),
            "loss_ib_recon": float(losses.ib_recon.detach().cpu()),
            "loss_ib_kl": float(losses.ib_kl.detach().cpu()),
            "keep_views_mean": float(view_keep_mask.float().sum(dim=1).mean().detach().cpu()),
        }
        return losses, stats

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        *,
        missing_views: int = 0,
        missing_seed: Optional[int] = None,
    ) -> EvaluationSummary:
        self.model.eval()
        rng = random.Random(missing_seed) if missing_seed is not None else random

        sample_labels: List[int] = []
        sample_scores: List[float] = []
        pixel_labels: List[np.ndarray] = []
        pixel_scores: List[np.ndarray] = []
        pixel_eval_mask: List[np.ndarray] = []
        categories: List[str] = []

        for batch in loader:
            batch = _to_device(batch, self.device)
            images: Tensor = batch["images"]
            view_valid: Tensor = batch["view_valid"]
            masks: Tensor = batch["masks"]
            if missing_views > 0:
                view_valid = view_valid.clone()
                bsz, n_views = view_valid.shape
                for bi in range(bsz):
                    valid_idx = [int(i) for i in torch.nonzero(view_valid[bi], as_tuple=False).flatten().tolist()]
                    if len(valid_idx) <= 1:
                        continue
                    k = min(int(missing_views), len(valid_idx) - 1)
                    drop_idx = rng.sample(valid_idx, k=k)
                    view_valid[bi, drop_idx] = False

            _clean_rgb, clean_norm = self._prepare_clean_rgb_and_norm(images)
            output = self.model(
                student_images=clean_norm,
                teacher_images=clean_norm,
                view_valid=view_valid,
                view_keep_mask=None,
            )

            pred_maps = output.anomaly_maps.detach().cpu().numpy()  # [B,V,H,W]
            gt_masks = masks.squeeze(2).detach().cpu().numpy()  # [B,V,H,W]
            valid_mask = view_valid[:, :, None, None].detach().cpu().numpy().astype(bool)

            labels_np = batch["is_anomaly"].detach().cpu().numpy().astype(np.uint8)
            scores_np = output.sample_scores.detach().cpu().numpy().astype(np.float32)

            b = pred_maps.shape[0]
            for i in range(b):
                sample_labels.append(int(labels_np[i]))
                sample_scores.append(float(scores_np[i]))
                pixel_labels.append(gt_masks[i])
                pixel_scores.append(pred_maps[i])
                pixel_eval_mask.append(np.broadcast_to(valid_mask[i], gt_masks[i].shape))
            categories.extend(list(batch["category"]))

        return summarize_metrics(
            sample_labels=sample_labels,
            sample_scores=sample_scores,
            pixel_labels=pixel_labels,
            pixel_scores=pixel_scores,
            pixel_eval_mask=pixel_eval_mask,
            categories=categories,
        )

    def _write_log(self, payload: Mapping[str, Any]) -> None:
        with self.metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_checkpoint(self, name: str, *, epoch: int, extra: Optional[Dict[str, Any]] = None) -> Path:
        path = self.save_dir / name
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            ckpt["scheduler"] = self.scheduler.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        return path

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        best_metric = -math.inf
        best_epoch = -1
        history: List[Dict[str, Any]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            running: Dict[str, float] = {}
            n_steps = 0
            for step, batch in enumerate(train_loader, start=1):
                _losses, stats = self._train_step(batch)
                n_steps += 1
                for k, v in stats.items():
                    running[k] = running.get(k, 0.0) + float(v)

                if self.cfg.log_interval > 0 and (step % self.cfg.log_interval == 0):
                    avg = {k: v / n_steps for k, v in running.items()}
                    self._write_log({"type": "train_step", "epoch": epoch, "step": step, **avg})

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_stats = {k: v / max(1, n_steps) for k, v in running.items()}
            epoch_stats["epoch"] = epoch
            epoch_stats["time_sec"] = time.time() - t0

            if (
                val_loader is not None
                and self.cfg.eval_every is not None
                and self.cfg.eval_every > 0
                and (epoch % self.cfg.eval_every == 0)
            ):
                eval_summary = self.evaluate(val_loader)
                epoch_stats.update(
                    {
                        "val_s_auroc": eval_summary.s_auroc,
                        "val_p_auroc": eval_summary.p_auroc,
                        "val_pro": eval_summary.pro,
                    }
                )
                metric = eval_summary.s_auroc
                if np.isfinite(metric) and metric > best_metric:
                    best_metric = float(metric)
                    best_epoch = epoch
                    self.save_checkpoint(
                        "best.pt",
                        epoch=epoch,
                        extra={"best_metric": best_metric, "best_epoch": best_epoch},
                    )
                    # Save per-category metrics for best model
                    with (self.save_dir / "best_eval_summary.json").open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "s_auroc": eval_summary.s_auroc,
                                "p_auroc": eval_summary.p_auroc,
                                "pro": eval_summary.pro,
                                "per_category": eval_summary.per_category,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

            self.save_checkpoint("last.pt", epoch=epoch)
            self._write_log({"type": "epoch", **epoch_stats})
            history.append(epoch_stats)

        # If evaluation is disabled, still provide a convenient best.pt alias.
        if best_epoch < 0:
            self.save_checkpoint("best.pt", epoch=self.cfg.epochs, extra={"best_metric": None, "best_epoch": None})

        return {"best_metric": best_metric, "best_epoch": best_epoch, "history": history}
