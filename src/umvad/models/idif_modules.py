from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    w = mask.to(dtype=x.dtype)
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    denom = w.sum(dim=dim, keepdim=False).clamp_min(1.0)
    num = (x * w).sum(dim=dim, keepdim=False)
    return num / denom


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )


@dataclass
class CBAux:
    mu: Tensor
    logvar: Tensor
    common_feature: Tensor
    recon_target: Tensor
    recon_pred: Tensor
    valid_mask: Tensor


class ConsistencyBottleneck(nn.Module):
    """Variational bottleneck that extracts a shared common feature across views."""

    def __init__(self, channels: int, latent_channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, latent_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.mu_head = nn.Conv2d(hidden, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(hidden, latent_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(
        self,
        f_e: Tensor,  # [B, V, C, H, W]
        view_keep_mask: Tensor,  # [B, V]
    ) -> Tuple[Tensor, Tensor, CBAux]:
        if f_e.ndim != 5:
            raise ValueError(f"Expected [B,V,C,H,W], got {tuple(f_e.shape)}")
        if view_keep_mask.ndim != 2:
            raise ValueError(f"Expected [B,V], got {tuple(view_keep_mask.shape)}")

        f_mean = _masked_mean(f_e, view_keep_mask, dim=1)  # [B, C, H, W]
        h = self.encoder(f_mean)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-10.0, max=10.0)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu

        f_c = self.decoder(z)  # projected common feature in original channel dim
        f_s = f_e - f_c.unsqueeze(1)

        # Exclude dropped views from subsequent fusion and losses.
        mask = view_keep_mask[:, :, None, None, None].to(dtype=f_s.dtype)
        f_s = f_s * mask

        aux = CBAux(
            mu=mu,
            logvar=logvar,
            common_feature=f_c,
            recon_target=f_mean.detach(),
            recon_pred=f_c,
            valid_mask=view_keep_mask,
        )
        return f_c, f_s, aux


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FusionBlock(nn.Module):
    """Self-attn on voxel tokens + cross-attn from voxel to view-specific tokens."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn1 = MLPBlock(dim=dim, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm4 = nn.LayerNorm(dim)
        self.ffn2 = MLPBlock(dim=dim, dropout=dropout)

    def forward(
        self,
        voxel_tokens: Tensor,  # [B, Nv, C]
        cond_tokens: Tensor,  # [B, Nk, C]
        cond_key_padding_mask: Optional[Tensor] = None,  # [B, Nk], True=ignore
    ) -> Tensor:
        x = voxel_tokens
        q = self.norm1(x)
        x = x + self.self_attn(q, q, q, need_weights=False)[0]
        x = x + self.ffn1(self.norm2(x))
        q = self.norm3(x)
        x = x + self.cross_attn(
            q,
            cond_tokens,
            cond_tokens,
            key_padding_mask=cond_key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.ffn2(self.norm4(x))
        return x


def _build_3d_rotation_y(theta: Tensor) -> Tensor:
    # theta: [B] in radians
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    mat = torch.stack(
        [
            torch.stack([c, zeros, s, zeros], dim=-1),
            torch.stack([zeros, ones, zeros, zeros], dim=-1),
            torch.stack([-s, zeros, c, zeros], dim=-1),
        ],
        dim=1,
    )
    return mat  # [B, 3, 4]


class STNProjector3D(nn.Module):
    """Project implicit voxel features to 2D using a learnable per-view rotation."""

    def __init__(
        self,
        channels: int,
        num_views: int,
        voxel_shape: Sequence[int],
        target_hw: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_views = int(num_views)
        self.voxel_shape = tuple(int(v) for v in voxel_shape)
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))
        # Initialize angles roughly spread over views.
        init = torch.linspace(0.0, 2.0 * math.pi, steps=self.num_views + 1)[:-1]
        self.view_angles = nn.Parameter(init)
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, voxel_tokens: Tensor) -> Tensor:
        # voxel_tokens: [B, Nv, C]
        b, nv, c = voxel_tokens.shape
        d, h, w = self.voxel_shape
        if nv != d * h * w:
            raise ValueError(f"voxel token count mismatch: {nv} vs {d*h*w}")
        if c != self.channels:
            raise ValueError(f"channel mismatch: {c} vs {self.channels}")

        volume = voxel_tokens.transpose(1, 2).contiguous().view(b, c, d, h, w)
        outputs = []
        for v in range(self.num_views):
            theta = _build_3d_rotation_y(self.view_angles[v].expand(b))
            grid = F.affine_grid(theta, size=volume.shape, align_corners=False)
            rotated = F.grid_sample(
                volume,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            feat2d = rotated.mean(dim=2)  # depth collapse -> [B,C,H,W]
            feat2d = F.interpolate(
                feat2d,
                size=self.target_hw,
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(self.post(feat2d))
        return torch.stack(outputs, dim=1)  # [B, V, C, H, W]


class ImplicitVoxelConstruction(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_views: int,
        feature_hw: Tuple[int, int],
        voxel_shape: Sequence[int] = (4, 8, 8),
        num_fusion_blocks: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_views = int(num_views)
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.voxel_shape = tuple(int(v) for v in voxel_shape)
        nv = math.prod(self.voxel_shape)
        self.voxel_prototype = nn.Parameter(torch.randn(1, nv, channels) * 0.02)
        self.voxel_pos = nn.Parameter(torch.randn(1, nv, channels) * 0.02)
        self.view_embed = nn.Parameter(torch.randn(1, self.num_views, channels) * 0.02)
        self.cond_norm = nn.LayerNorm(channels)
        self.blocks = nn.ModuleList(
            [FusionBlock(dim=channels, num_heads=num_heads, dropout=dropout) for _ in range(num_fusion_blocks)]
        )
        self.projector = STNProjector3D(
            channels=channels,
            num_views=num_views,
            voxel_shape=voxel_shape,
            target_hw=feature_hw,
        )

    def _prepare_cond_tokens(self, f_s: Tensor) -> Tuple[Tensor, Tensor]:
        # f_s: [B,V,C,H,W]
        b, v, c, h, w = f_s.shape
        if v != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {v}")
        x = f_s.permute(0, 1, 3, 4, 2).contiguous().view(b, v, h * w, c)
        x = x + self.view_embed[:, :, None, :]
        x = x.view(b, v * h * w, c)
        x = self.cond_norm(x)
        return x, torch.arange(v * h * w, device=f_s.device)

    def _build_token_key_padding_mask(self, view_keep_mask: Tensor, h: int, w: int) -> Tensor:
        # MultiheadAttention expects True for positions that should be ignored.
        keep = view_keep_mask[:, :, None].expand(-1, -1, h * w).reshape(view_keep_mask.shape[0], -1)
        return ~keep

    def forward(
        self,
        f_s: Tensor,  # [B,V,C,H,W]
        view_keep_mask: Tensor,  # [B,V]
    ) -> Tensor:
        b, _, _, h, w = f_s.shape
        cond_tokens, _ = self._prepare_cond_tokens(f_s)
        cond_key_padding_mask = self._build_token_key_padding_mask(view_keep_mask, h, w)

        voxel_tokens = self.voxel_prototype.expand(b, -1, -1) + self.voxel_pos
        for block in self.blocks:
            voxel_tokens = block(
                voxel_tokens=voxel_tokens,
                cond_tokens=cond_tokens,
                cond_key_padding_mask=cond_key_padding_mask,
            )
        return self.projector(voxel_tokens)


class ViewFeatureDecoder(nn.Module):
    """Decode fused deep feature to ResNet18 layer3/layer2/layer1 feature shapes."""

    def __init__(self, channels: Sequence[int] = (64, 128, 256)) -> None:
        super().__init__()
        c1, c2, c3 = [int(c) for c in channels]
        self.deep_refine = _conv_block(c3, c3)
        self.mid_reduce = nn.Conv2d(c3, c2, kernel_size=1)
        self.mid_fuse = _conv_block(c2 + c2, c2)
        self.shallow_reduce = nn.Conv2d(c2, c1, kernel_size=1)
        self.shallow_fuse = _conv_block(c1 + c1, c1)
        self.out1 = nn.Conv2d(c1, c1, kernel_size=1)
        self.out2 = nn.Conv2d(c2, c2, kernel_size=1)
        self.out3 = nn.Conv2d(c3, c3, kernel_size=1)

    def forward(self, deep: Tensor, skip2: Tensor, skip1: Tensor) -> Dict[str, Tensor]:
        # deep: [B,256,16,16], skip2:[B,128,32,32], skip1:[B,64,64,64]
        f3 = self.out3(self.deep_refine(deep))

        mid = F.interpolate(f3, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        mid = self.mid_reduce(mid)
        mid = self.mid_fuse(torch.cat([mid, skip2], dim=1))
        f2 = self.out2(mid)

        sh = F.interpolate(f2, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        sh = self.shallow_reduce(sh)
        sh = self.shallow_fuse(torch.cat([sh, skip1], dim=1))
        f1 = self.out1(sh)

        return {"layer1": f1, "layer2": f2, "layer3": f3}


class Conv3DFusion(nn.Module):
    """Ablation module: fuse multi-view features via 3D convolutions."""

    def __init__(self, channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        blocks = []
        for _ in range(max(1, int(num_blocks))):
            blocks.extend(
                [
                    nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(channels),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor, view_keep_mask: Tensor) -> Tensor:
        # x: [B,V,C,H,W] -> [B,C,V,H,W]
        m = view_keep_mask[:, :, None, None, None].to(dtype=x.dtype)
        x = x * m
        y = x.permute(0, 2, 1, 3, 4).contiguous()
        y = self.net(y)
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        y = y * m
        return y
