"""Reference-guided convex interpolation of cascade search centers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedCenterUpsampler(nn.Module):
    """Reweight four bilinear neighbors, starting from bilinear interpolation.

    The output stays between neighboring coarse depths. This changes only the
    next stage's center, not its number of hypotheses or uniform spacing.
    """

    def __init__(self, feature_channels=32, residual_limit=2.0):
        super().__init__()
        self.residual_limit = residual_limit
        self.features = nn.Sequential(
            nn.Conv2d(feature_channels + 4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.logits = nn.Conv2d(16, 4, 3, padding=1)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    @staticmethod
    def neighbors_and_prior(depth, target_size):
        batch, height, width = depth.shape
        out_h, out_w = target_size
        y = ((torch.arange(out_h, device=depth.device, dtype=depth.dtype) + 0.5)
             * height / out_h - 0.5).clamp(0, height - 1)
        x = ((torch.arange(out_w, device=depth.device, dtype=depth.dtype) + 0.5)
             * width / out_w - 0.5).clamp(0, width - 1)
        y0, x0 = y.floor().long(), x.floor().long()
        y1, x1 = (y0 + 1).clamp(max=height - 1), (x0 + 1).clamp(max=width - 1)
        fy, fx = (y - y0).view(-1, 1), (x - x0).view(1, -1)
        indices = torch.stack([yy[:, None] * width + xx[None, :]
                               for yy, xx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))])
        neighbors = depth.reshape(batch, -1)[:, indices.reshape(-1)].reshape(batch, 4, out_h, out_w)
        prior = torch.stack(((1-fy)*(1-fx), (1-fy)*fx, fy*(1-fx), fy*fx)).unsqueeze(0)
        return neighbors, prior

    def forward(self, depth, reference_features, base_interval):
        target_size = reference_features.shape[-2:]
        neighbors, prior = self.neighbors_and_prior(depth, target_size)
        bilinear = F.interpolate(depth.unsqueeze(1), size=target_size,
                                 mode='bilinear', align_corners=False)
        interval = base_interval.reshape(depth.shape[0], 1, 1, 1).clamp(min=1e-6)
        relative_depths = ((neighbors - bilinear) / interval).detach().clamp(-16.0, 16.0)
        features = torch.cat((reference_features, relative_depths), dim=1)
        residual = self.residual_limit * torch.tanh(self.logits(self.features(features)))
        weights = prior * torch.exp(residual)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (weights * neighbors).sum(dim=1)
