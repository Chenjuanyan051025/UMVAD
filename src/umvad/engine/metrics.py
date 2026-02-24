from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

try:
    from scipy import ndimage  # type: ignore
    _SCIPY_NDIMAGE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local env
    ndimage = None  # type: ignore[assignment]
    _SCIPY_NDIMAGE_IMPORT_ERROR = exc


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_pro(
    pred_maps: np.ndarray,  # [N,H,W] float
    gt_masks: np.ndarray,  # [N,H,W] bool/0-1
    *,
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    if ndimage is None:
        # Allow training/evaluation to continue when scipy is broken or unavailable.
        # PRO will be unavailable in this case.
        return float("nan")

    pred_maps = np.asarray(pred_maps, dtype=np.float32)
    gt_masks = np.asarray(gt_masks).astype(bool)
    if pred_maps.shape != gt_masks.shape:
        raise ValueError(f"Shape mismatch: {pred_maps.shape} vs {gt_masks.shape}")

    # Keep only samples with at least one positive pixel for PRO region overlap.
    has_pos = gt_masks.reshape(gt_masks.shape[0], -1).any(axis=1)
    if not np.any(has_pos):
        return float("nan")
    pred_maps = pred_maps[has_pos]
    gt_masks = gt_masks[has_pos]

    lo = float(pred_maps.min())
    hi = float(pred_maps.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    if hi <= lo:
        return 0.0

    thresholds = np.linspace(hi, lo, num=num_thresholds, endpoint=True)
    fprs: List[float] = []
    pros: List[float] = []

    neg_total = (~gt_masks).sum()
    neg_total = int(neg_total)
    if neg_total <= 0:
        return float("nan")

    for th in thresholds:
        pred_bin = pred_maps >= th
        fp = np.logical_and(pred_bin, ~gt_masks).sum()
        fpr = float(fp) / float(neg_total)

        per_region_scores: List[float] = []
        for i in range(gt_masks.shape[0]):
            labeled, n_comp = ndimage.label(gt_masks[i])
            if n_comp == 0:
                continue
            for comp_id in range(1, n_comp + 1):
                region = labeled == comp_id
                region_size = int(region.sum())
                if region_size == 0:
                    continue
                overlap = np.logical_and(pred_bin[i], region).sum()
                per_region_scores.append(float(overlap) / float(region_size))
        if not per_region_scores:
            continue
        fprs.append(fpr)
        pros.append(float(np.mean(per_region_scores)))

    if not fprs:
        return float("nan")
    fprs_np = np.asarray(fprs, dtype=np.float64)
    pros_np = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs_np)
    fprs_np = fprs_np[order]
    pros_np = pros_np[order]

    keep = fprs_np <= max_fpr
    if not np.any(keep):
        return 0.0
    x = fprs_np[keep]
    y = pros_np[keep]
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < max_fpr:
        x = np.concatenate([x, [max_fpr]])
        y = np.concatenate([y, [y[-1]]])
    return float(auc(x, y) / max_fpr)


@dataclass
class EvaluationSummary:
    s_auroc: float
    p_auroc: float
    pro: float
    per_category: Dict[str, Dict[str, float]]


def summarize_metrics(
    *,
    sample_labels: Sequence[int],
    sample_scores: Sequence[float],
    pixel_labels: Sequence[np.ndarray],
    pixel_scores: Sequence[np.ndarray],
    pixel_eval_mask: Optional[Sequence[np.ndarray]] = None,
    categories: Optional[Sequence[str]] = None,
) -> EvaluationSummary:
    sample_labels_np = np.asarray(sample_labels, dtype=np.uint8)
    sample_scores_np = np.asarray(sample_scores, dtype=np.float64)

    s_auroc = safe_auroc(sample_labels_np, sample_scores_np)

    px_true_parts: List[np.ndarray] = []
    px_score_parts: List[np.ndarray] = []
    pro_score_maps: List[np.ndarray] = []
    pro_gt_maps: List[np.ndarray] = []
    for i, (gt, pred) in enumerate(zip(pixel_labels, pixel_scores)):
        gt_np = np.asarray(gt).astype(np.uint8)
        pred_np = np.asarray(pred, dtype=np.float32)
        if gt_np.shape != pred_np.shape:
            raise ValueError(f"Pixel shape mismatch at idx={i}: {gt_np.shape} vs {pred_np.shape}")
        if pixel_eval_mask is None:
            mask = np.ones_like(gt_np, dtype=bool)
        else:
            mask = np.asarray(pixel_eval_mask[i]).astype(bool)
            if mask.shape != gt_np.shape:
                raise ValueError(f"Pixel eval mask mismatch at idx={i}: {mask.shape} vs {gt_np.shape}")
        if np.any(mask):
            px_true_parts.append(gt_np[mask].reshape(-1))
            px_score_parts.append(pred_np[mask].reshape(-1))
            pro_pred = pred_np * mask.astype(pred_np.dtype)
            pro_gt = (gt_np > 0) & mask
            if pro_pred.ndim < 2:
                raise ValueError(f"Pixel maps must be at least 2D, got ndim={pro_pred.ndim}")
            if pro_pred.ndim == 2:
                pro_pred = pro_pred[None, ...]
                pro_gt = pro_gt[None, ...]
            else:
                flat_n = int(np.prod(pro_pred.shape[:-2]))
                pro_pred = pro_pred.reshape(flat_n, pro_pred.shape[-2], pro_pred.shape[-1])
                pro_gt = pro_gt.reshape(flat_n, pro_gt.shape[-2], pro_gt.shape[-1])
            pro_score_maps.extend([x for x in pro_pred])
            pro_gt_maps.extend([x for x in pro_gt])

    if px_true_parts:
        p_auroc = safe_auroc(np.concatenate(px_true_parts), np.concatenate(px_score_parts))
    else:
        p_auroc = float("nan")
    if pro_score_maps:
        pro = compute_pro(np.stack(pro_score_maps, axis=0), np.stack(pro_gt_maps, axis=0))
    else:
        pro = float("nan")

    per_category: Dict[str, Dict[str, float]] = {}
    if categories is not None:
        cat_list = list(categories)
        if len(cat_list) != len(sample_labels_np):
            raise ValueError("categories length must equal number of samples")
        cat_to_indices: Dict[str, List[int]] = {}
        for idx, cat in enumerate(cat_list):
            cat_to_indices.setdefault(cat, []).append(idx)
        for cat, indices in sorted(cat_to_indices.items()):
            s_y = sample_labels_np[indices]
            s_s = sample_scores_np[indices]
            s_val = safe_auroc(s_y, s_s)

            cat_px_true: List[np.ndarray] = []
            cat_px_score: List[np.ndarray] = []
            for i in indices:
                gt_np = np.asarray(pixel_labels[i]).astype(np.uint8)
                pred_np = np.asarray(pixel_scores[i], dtype=np.float32)
                if pixel_eval_mask is None:
                    m = np.ones_like(gt_np, dtype=bool)
                else:
                    m = np.asarray(pixel_eval_mask[i]).astype(bool)
                if np.any(m):
                    cat_px_true.append(gt_np[m].reshape(-1))
                    cat_px_score.append(pred_np[m].reshape(-1))
            if cat_px_true:
                p_val = safe_auroc(np.concatenate(cat_px_true), np.concatenate(cat_px_score))
            else:
                p_val = float("nan")
            per_category[cat] = {"s_auroc": s_val, "p_auroc": p_val}

    return EvaluationSummary(s_auroc=s_auroc, p_auroc=p_auroc, pro=pro, per_category=per_category)
