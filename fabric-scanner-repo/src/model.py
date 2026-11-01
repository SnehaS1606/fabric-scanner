"""
model.py
Three-branch fusion classifier:
  1. Spatial branch  -> pretrained ResNet18 (transfer learning)
  2. Frequency branch -> 2D DWT sub-bands -> small CNN
  3. Sheen branch    -> small CNN on the flash-difference image

All three feature vectors are concatenated and passed through a
classification head.
"""

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop fc
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        feat = self.features(x).flatten(1)   # (B, 512)
        return self.proj(feat)                # (B, out_dim)


class FrequencyBranch(nn.Module):
    """Applies a 2D DWT per-channel, stacks the LH/HL/HH sub-bands
    (the high-frequency detail coefficients -> weave/micro-texture),
    and runs a small CNN over them."""

    def __init__(self, out_dim=128, wavelet="haar"):
        super().__init__()
        self.wavelet = wavelet
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def dwt_batch(self, x):
        # x: (B, 3, H, W) tensor -> compute DWT per image/channel on CPU numpy,
        # then stack LH, HL, HH sub-bands back into a tensor.
        b = x.shape[0]
        out = []
        x_np = x.detach().cpu().numpy()
        for i in range(b):
            bands = []
            for c in range(3):
                _, (lh, hl, hh) = pywt.dwt2(x_np[i, c], self.wavelet)
                bands.extend([lh, hl, hh])
            stacked = torch.tensor(np.array(bands), dtype=torch.float32)  # (9, H/2, W/2)
            out.append(stacked)
        return torch.stack(out).to(x.device)

    def forward(self, x):
        freq_maps = self.dwt_batch(x)     # (B, 9, H/2, W/2)
        feat = self.cnn(freq_maps).flatten(1)
        return self.proj(feat)


class GaborBranch(nn.Module):
    """Alternative to the DWT frequency branch: a fixed (non-learned) bank of Gabor filters
    (4 orientations x 3 wavelengths) applied to the grey flash-OFF image, followed by a small CNN."""

    def __init__(self, out_dim=128, orientations=4, wavelengths=(3, 6, 12), ksize=15):
        super().__init__()
        kernels = []
        for lam in wavelengths:
            for k in range(orientations):
                theta = np.pi * k / orientations
                kern = cv2.getGaborKernel((ksize, ksize), 0.56 * lam, theta, lam, 0.5, 0)
                kernels.append(kern - kern.mean())          # zero-mean: flat areas give no response
        bank = torch.tensor(np.stack(kernels), dtype=torch.float32).unsqueeze(1)   # (12, 1, k, k)
        self.register_buffer("bank", bank)
        self.cnn = nn.Sequential(
            nn.Conv2d(bank.shape[0], 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)                                  # (B, 1, H, W)
        resp = torch.abs(F.conv2d(gray, self.bank, padding=self.bank.shape[-1] // 2))
        return self.proj(self.cnn(resp).flatten(1))


class SheenBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        feat = self.cnn(x).flatten(1)
        return self.proj(feat)


class FusionFabricClassifier(nn.Module):
    def __init__(self, num_classes, spatial_dim=256, freq_dim=128, sheen_dim=128,
                 use_frequency=True, use_sheen=True, freq_type="dwt"):
        super().__init__()
        self.use_frequency = use_frequency
        self.use_sheen = use_sheen

        self.spatial = SpatialBranch(spatial_dim)
        fused_dim = spatial_dim

        if use_frequency:
            self.frequency = GaborBranch(freq_dim) if freq_type == "gabor" else FrequencyBranch(freq_dim)
            fused_dim += freq_dim
        if use_sheen:
            self.sheen = SheenBranch(sheen_dim)
            fused_dim += sheen_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, off_image, diff_image):
        parts = [self.spatial(off_image)]
        if self.use_frequency:
            parts.append(self.frequency(off_image))
        if self.use_sheen:
            parts.append(self.sheen(diff_image))
        fused = torch.cat(parts, dim=1)
        return self.head(fused)
