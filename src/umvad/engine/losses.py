from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from umvad.models.idif import IDIFForwardOutput


@dataclass
class LossBreakdown:
    total: Tensor
    distill: Tensor
    ib: Tensor
    ib_recon: Tensor
    ib_kl: Tensor


def _masked_feature_cosine_loss(
    teacher: Tensor,  # [B,V,C,H,W]
    student: Tensor,  # [B,V,C,H,W]
    mask: Tensor,  # [B,V]
) -> Tensor:
    cos = F.cosine_similarity(teacher, student, dim=2, eps=1e-8)  # [B,V,H,W]
    dist = 1.0 - cos
    # Average spatially per view, then apply view mask.
    dist = dist.mean(dim=(-1, -2))  # [B,V]
    w = mask.to(dtype=dist.dtype)
    denom = w.sum().clamp_min(1.0)
    return (dist * w).sum() / denom


def compute_idif_loss(
    output: IDIFForwardOutput,
    *,
    distill_weight: float = 1.0,
    ib_weight: float = 1.0,
    ib_kl_weight: float = 1.0,
    ib_recon_weight: float = 1.0,
    scale_weights: Optional[Mapping[str, float]] = None,
) -> LossBreakdown:
    if scale_weights is None:
        scale_weights = {"layer1": 1.0, "layer2": 1.0, "layer3": 1.0}

    distill_terms = []
    distill_term_weights = []
    for name, w in scale_weights.items():
        if name not in output.teacher_features or name not in output.student_features:
            continue
        if w <= 0:
            continue
        loss_i = _masked_feature_cosine_loss(
            output.teacher_features[name],
            output.student_features[name],
            output.effective_view_mask,
        )
        distill_terms.append(loss_i)
        distill_term_weights.append(float(w))

    if not distill_terms:
        raise RuntimeError("No distillation terms available.")
    w_tensor = torch.tensor(distill_term_weights, device=distill_terms[0].device, dtype=distill_terms[0].dtype)
    stacked = torch.stack(distill_terms)
    distill = (stacked * w_tensor).sum() / w_tensor.sum().clamp_min(1e-8)

    cb = output.cb_aux
    ib_recon = F.mse_loss(cb.recon_pred, cb.recon_target)
    mu = cb.mu
    logvar = cb.logvar
    ib_kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
    ib = ib_recon_weight * ib_recon + ib_kl_weight * ib_kl

    total = float(distill_weight) * distill + float(ib_weight) * ib
    return LossBreakdown(total=total, distill=distill, ib=ib, ib_recon=ib_recon, ib_kl=ib_kl)

