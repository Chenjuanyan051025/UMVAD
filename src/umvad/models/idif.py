from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbones import ResNet18FeatureExtractor, flatten_views, unflatten_views
from .idif_modules import (
    CBAux,
    ConsistencyBottleneck,
    Conv3DFusion,
    ImplicitVoxelConstruction,
    ViewFeatureDecoder,
)


TeacherFeatures = Dict[str, Tensor]
StudentFeatures = Dict[str, Tensor]


@dataclass
class IDIFForwardOutput:
    teacher_features: TeacherFeatures
    student_features: StudentFeatures
    anomaly_maps: Tensor  # [B, V, H, W]
    sample_scores: Tensor  # [B]
    effective_view_mask: Tensor  # [B, V]
    cb_aux: CBAux


class IDIFStudent(nn.Module):
    def __init__(
        self,
        *,
        num_views: int = 5,
        feature_hw: Tuple[int, int] = (16, 16),
        cb_latent_channels: int = 64,
        voxel_shape: Sequence[int] = (4, 8, 8),
        num_fusion_blocks: int = 2,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        encoder_pretrained: bool = False,
        use_cb: bool = True,
        fusion_mode: str = "ivc",  # {"ivc", "3dconv", "none"}
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.use_cb = bool(use_cb)
        self.fusion_mode = str(fusion_mode).lower()
        self.encoder = ResNet18FeatureExtractor(pretrained=encoder_pretrained, trainable=True)
        self.cb = ConsistencyBottleneck(channels=256, latent_channels=cb_latent_channels)
        if self.fusion_mode == "ivc":
            self.fusion = ImplicitVoxelConstruction(
                channels=256,
                num_views=num_views,
                feature_hw=feature_hw,
                voxel_shape=voxel_shape,
                num_fusion_blocks=num_fusion_blocks,
                num_heads=num_heads,
                dropout=attn_dropout,
            )
        elif self.fusion_mode == "3dconv":
            self.fusion = Conv3DFusion(channels=256, num_blocks=num_fusion_blocks)
        elif self.fusion_mode == "none":
            self.fusion = None
        else:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}")
        self.view_decoders = nn.ModuleList([ViewFeatureDecoder(channels=(64, 128, 256)) for _ in range(num_views)])

    def forward(
        self,
        images: Tensor,  # [B,V,3,H,W]
        *,
        view_valid: Optional[Tensor] = None,  # [B,V]
        view_keep_mask: Optional[Tensor] = None,  # [B,V]
    ) -> Tuple[StudentFeatures, Tensor, CBAux]:
        if images.ndim != 5:
            raise ValueError(f"Expected [B,V,3,H,W], got {tuple(images.shape)}")
        b, v, _, _, _ = images.shape
        if v != self.num_views:
            raise ValueError(f"Expected num_views={self.num_views}, got {v}")

        if view_valid is None:
            view_valid = torch.ones(b, v, dtype=torch.bool, device=images.device)
        if view_keep_mask is None:
            view_keep_mask = view_valid
        effective_mask = view_valid & view_keep_mask

        flat, _, _ = flatten_views(images)
        enc = self.encoder(flat)
        f1 = unflatten_views(enc["layer1"], b, v)
        f2 = unflatten_views(enc["layer2"], b, v)
        f3 = unflatten_views(enc["layer3"], b, v)

        if self.use_cb:
            fc, fs, cb_aux = self.cb(f3, effective_mask)
        else:
            fc = torch.zeros_like(f3[:, 0])
            fs = f3 * effective_mask[:, :, None, None, None].to(dtype=f3.dtype)
            zero_stats = torch.zeros_like(fc[:, :1])
            cb_aux = CBAux(
                mu=zero_stats,
                logvar=zero_stats,
                common_feature=fc,
                recon_target=fc.detach(),
                recon_pred=fc,
                valid_mask=effective_mask,
            )

        if self.fusion_mode == "ivc":
            assert self.fusion is not None
            fp = self.fusion(fs, effective_mask)
            deep = fp + fc.unsqueeze(1)
        elif self.fusion_mode == "3dconv":
            assert self.fusion is not None
            fp = self.fusion(fs, effective_mask)
            deep = fp + fc.unsqueeze(1)
        else:  # none
            deep = fs + fc.unsqueeze(1)

        out_l1 = []
        out_l2 = []
        out_l3 = []
        for view_idx in range(v):
            decoded = self.view_decoders[view_idx](
                deep=deep[:, view_idx],
                skip2=f2[:, view_idx],
                skip1=f1[:, view_idx],
            )
            out_l1.append(decoded["layer1"])
            out_l2.append(decoded["layer2"])
            out_l3.append(decoded["layer3"])

        student_features: StudentFeatures = {
            "layer1": torch.stack(out_l1, dim=1),
            "layer2": torch.stack(out_l2, dim=1),
            "layer3": torch.stack(out_l3, dim=1),
        }
        return student_features, effective_mask, cb_aux


class IDIFModel(nn.Module):
    """Teacher-student IDIF model with anomaly-map computation."""

    def __init__(
        self,
        *,
        num_views: int = 5,
        input_size: int = 256,
        cb_latent_channels: int = 64,
        voxel_shape: Sequence[int] = (4, 8, 8),
        num_fusion_blocks: int = 2,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        teacher_pretrained: bool = True,
        student_encoder_pretrained: bool = False,
        use_cb: bool = True,
        fusion_mode: str = "ivc",
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.input_size = int(input_size)
        self.teacher = ResNet18FeatureExtractor(pretrained=teacher_pretrained, trainable=False)
        self.student = IDIFStudent(
            num_views=num_views,
            feature_hw=(16, 16),
            cb_latent_channels=cb_latent_channels,
            voxel_shape=voxel_shape,
            num_fusion_blocks=num_fusion_blocks,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            encoder_pretrained=student_encoder_pretrained,
            use_cb=use_cb,
            fusion_mode=fusion_mode,
        )

    @torch.no_grad()
    def _forward_teacher(self, images: Tensor) -> TeacherFeatures:
        flat, b, v = flatten_views(images)
        feats = self.teacher(flat)
        return {k: unflatten_views(val, b, v) for k, val in feats.items() if k in {"layer1", "layer2", "layer3"}}

    def _compute_anomaly_maps(
        self,
        teacher_features: Mapping[str, Tensor],
        student_features: Mapping[str, Tensor],
        *,
        output_hw: Tuple[int, int],
        view_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        maps = []
        for name in ("layer1", "layer2", "layer3"):
            if name not in teacher_features or name not in student_features:
                continue
            t = teacher_features[name]
            s = student_features[name]
            if t.shape != s.shape:
                raise ValueError(f"Feature shape mismatch at {name}: {tuple(t.shape)} vs {tuple(s.shape)}")
            dist = 1.0 - F.cosine_similarity(t, s, dim=2, eps=1e-8)  # [B,V,H,W]
            b, v, h, w = dist.shape
            up = F.interpolate(
                dist.view(b * v, 1, h, w),
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            ).view(b, v, output_hw[0], output_hw[1])
            maps.append(up)
        if not maps:
            raise RuntimeError("No feature maps available for anomaly scoring.")
        anomaly_maps = torch.stack(maps, dim=0).mean(dim=0)  # [B,V,H,W]
        valid = view_valid[:, :, None, None]
        anomaly_maps = anomaly_maps * valid.to(dtype=anomaly_maps.dtype)

        masked_scores = anomaly_maps.masked_fill(~valid, float("-inf"))
        sample_scores = masked_scores.amax(dim=(1, 2, 3))
        sample_scores = torch.where(torch.isfinite(sample_scores), sample_scores, torch.zeros_like(sample_scores))
        return anomaly_maps, sample_scores

    def forward(
        self,
        student_images: Tensor,
        *,
        teacher_images: Optional[Tensor] = None,
        view_valid: Optional[Tensor] = None,
        view_keep_mask: Optional[Tensor] = None,
    ) -> IDIFForwardOutput:
        if teacher_images is None:
            teacher_images = student_images
        if view_valid is None:
            b, v = student_images.shape[:2]
            view_valid = torch.ones(b, v, dtype=torch.bool, device=student_images.device)

        with torch.no_grad():
            teacher_features = self._forward_teacher(teacher_images)

        student_features, effective_mask, cb_aux = self.student(
            student_images,
            view_valid=view_valid,
            view_keep_mask=view_keep_mask,
        )
        anomaly_maps, sample_scores = self._compute_anomaly_maps(
            teacher_features=teacher_features,
            student_features=student_features,
            output_hw=(student_images.shape[-2], student_images.shape[-1]),
            view_valid=view_valid,
        )
        return IDIFForwardOutput(
            teacher_features=dict(teacher_features),
            student_features=student_features,
            anomaly_maps=anomaly_maps,
            sample_scores=sample_scores,
            effective_view_mask=effective_mask,
            cb_aux=cb_aux,
        )
