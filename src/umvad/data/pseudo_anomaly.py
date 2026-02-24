from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def _pil_rgb_to_tensor(img: Image.Image) -> Tensor:
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    h, w = img.size[1], img.size[0]
    arr = arr.view(h, w, 3).permute(2, 0, 1).contiguous()
    return arr.float() / 255.0


class DTDTexturePool:
    """Lightweight DTD texture sampler. Expects image files under root recursively."""

    def __init__(self, root: str | Path, *, exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp")) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"DTD root not found: {self.root}")
        exts_l = {e.lower() for e in exts}
        self.files: List[Path] = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in exts_l]
        if not self.files:
            raise RuntimeError(f"No texture images found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def sample(self, size: Tuple[int, int]) -> Tensor:
        path = random.choice(self.files)
        img = Image.open(path).convert("RGB")
        img = img.resize((size[1], size[0]), resample=Image.BILINEAR)
        return _pil_rgb_to_tensor(img)


def _random_smooth_mask(
    batch: int,
    views: int,
    h: int,
    w: int,
    *,
    p_range: Tuple[float, float] = (0.08, 0.35),
    coarse_hw: Tuple[int, int] = (32, 32),
) -> Tensor:
    ch, cw = coarse_hw
    noise = torch.rand(batch, views, 1, ch, cw)
    noise = F.interpolate(noise.view(batch * views, 1, ch, cw), size=(h, w), mode="bilinear", align_corners=False)
    noise = noise.view(batch, views, 1, h, w)
    thresh = torch.empty(batch, views, 1, 1, 1).uniform_(1.0 - p_range[1], 1.0 - p_range[0])
    mask = (noise > thresh).float()

    # Add one random rectangle sometimes to create larger contiguous anomalies.
    if random.random() < 0.7:
        for b in range(batch):
            for v in range(views):
                if random.random() < 0.6:
                    rh = random.randint(max(8, h // 16), max(8, h // 4))
                    rw = random.randint(max(8, w // 16), max(8, w // 4))
                    y0 = random.randint(0, max(0, h - rh))
                    x0 = random.randint(0, max(0, w - rw))
                    mask[b, v, :, y0 : y0 + rh, x0 : x0 + rw] = 1.0
    return mask


class PseudoAnomalySynthesizer:
    """Generate pseudo-anomalous multi-view images from normal images.

    This is a configurable approximation of the DeSTSeg-style synthesis used in IDIF.
    """

    def __init__(
        self,
        *,
        dtd_root: Optional[str | Path] = None,
        blend_alpha_range: Tuple[float, float] = (0.3, 0.8),
        mask_ratio_range: Tuple[float, float] = (0.08, 0.35),
    ) -> None:
        self.textures = DTDTexturePool(dtd_root) if dtd_root else None
        self.blend_alpha_range = tuple(float(x) for x in blend_alpha_range)
        self.mask_ratio_range = tuple(float(x) for x in mask_ratio_range)

    def _sample_texture_batch(self, b: int, v: int, h: int, w: int, device: torch.device) -> Tensor:
        tex = torch.empty(b, v, 3, h, w, dtype=torch.float32)
        if self.textures is None:
            tex.uniform_(0.0, 1.0)
            # smooth random color texture fallback
            tex = F.avg_pool2d(tex.view(b * v, 3, h, w), kernel_size=7, stride=1, padding=3).view(b, v, 3, h, w)
            return tex.to(device)
        for bi in range(b):
            for vi in range(v):
                tex[bi, vi] = self.textures.sample((h, w))
        return tex.to(device)

    @torch.no_grad()
    def __call__(self, images: Tensor, *, view_valid: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        # images expected in [0,1] range
        if images.ndim != 5:
            raise ValueError(f"Expected [B,V,C,H,W], got {tuple(images.shape)}")
        b, v, c, h, w = images.shape
        if c != 3:
            raise ValueError(f"Expected RGB input, got channels={c}")
        if view_valid is None:
            view_valid = torch.ones(b, v, dtype=torch.bool, device=images.device)

        tex = self._sample_texture_batch(b, v, h, w, images.device)
        mask = _random_smooth_mask(
            b, v, h, w, p_range=self.mask_ratio_range, coarse_hw=(max(8, h // 8), max(8, w // 8))
        ).to(device=images.device, dtype=images.dtype)
        alpha = torch.empty(b, v, 1, 1, 1, device=images.device, dtype=images.dtype).uniform_(
            self.blend_alpha_range[0], self.blend_alpha_range[1]
        )

        mixed = alpha * tex + (1.0 - alpha) * images
        pseudo = images * (1.0 - mask) + mixed * mask

        valid = view_valid[:, :, None, None, None].to(dtype=images.dtype)
        pseudo = pseudo * valid + images * (1.0 - valid)
        mask = mask * valid
        return pseudo.clamp(0.0, 1.0), mask

