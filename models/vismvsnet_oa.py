import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .dynamic_visibility import ground_truth_pair_visibility


# =============================================================================
# Feature Extractor: multi-scale, outputs at 1/8, 1/4, 1/2 resolution
# =============================================================================
class MultiScaleFeatureNet(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        # 1/2 scale
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        # 1/4 scale
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)

        # 1/8 scale
        self.conv7 = ConvBnReLU(32, 32, 5, 2, 2)
        self.conv8 = ConvBnReLU(32, 32, 3, 1, 1)

        # output convolutions — one per scale
        self.feat2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.feat4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.feat8 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))

        x2 = self.conv4(self.conv3(self.conv2(x)))      # 1/2
        x4 = self.conv6(self.conv5(x2))                  # 1/4
        x8 = self.conv8(self.conv7(x4))                  # 1/8

        return self.feat8(x8), self.feat4(x4), self.feat2(x2)


# =============================================================================
# 3D Regularization Networks
# =============================================================================
class RegNet3D(nn.Module):
    """3D encoder-decoder for per-pair cost volume (3-level: 8→16→32→64→32→16→8)."""

    def __init__(self, in_channels=8):
        super(RegNet3D, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(32))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(16))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(8))

    def forward(self, x):
        conv0 = self.conv0(x)                               # [B, 8, D, H, W]
        conv2 = self.conv2(self.conv1(conv0))               # [B, 16, D, H/2, W/2]
        conv4 = self.conv4(self.conv3(conv2))               # [B, 32, D, H/4, W/4]
        conv6 = self.conv6(self.conv5(conv4))               # [B, 64, D, H/8, W/8]
        x = self.deconv5(conv6)
        x = x[:, :, :conv4.shape[2], :conv4.shape[3], :conv4.shape[4]]
        x = F.relu(x + conv4, inplace=True)
        x = self.deconv3(x)
        x = x[:, :, :conv2.shape[2], :conv2.shape[3], :conv2.shape[4]]
        x = F.relu(x + conv2, inplace=True)
        x = self.deconv1(x)
        x = x[:, :, :conv0.shape[2], :conv0.shape[3], :conv0.shape[4]]
        x = F.relu(x + conv0, inplace=True)
        return x


class RegPair(nn.Module):
    """Per-pair score head: 8 → 1."""

    def __init__(self):
        super(RegPair, self).__init__()
        self.conv = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class RegFuse(nn.Module):
    """Fused cost volume regularization (3-level) + final score head."""

    def __init__(self):
        super(RegFuse, self).__init__()
        self.conv0 = ConvBnReLU3D(8, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(32))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(16))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(8))

        self.final = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        conv6 = self.conv6(self.conv5(conv4))
        x = self.deconv5(conv6)
        x = x[:, :, :conv4.shape[2], :conv4.shape[3], :conv4.shape[4]]
        x = F.relu(x + conv4, inplace=True)
        x = self.deconv3(x)
        x = x[:, :, :conv2.shape[2], :conv2.shape[3], :conv2.shape[4]]
        x = F.relu(x + conv2, inplace=True)
        x = self.deconv1(x)
        x = x[:, :, :conv0.shape[2], :conv0.shape[3], :conv0.shape[4]]
        x = F.relu(x + conv0, inplace=True)
        x = self.final(x)
        return x


# =============================================================================
# Uncertainty Network
# =============================================================================
class UncertNet(nn.Module):
    """2D CNN: entropy map → uncertainty + occlusion logits (two heads)."""

    def __init__(self):
        super(UncertNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.uncert_head = nn.Conv2d(8, 1, 3, 1, 1, bias=False)
        self.occ_head = nn.Conv2d(8, 1, 3, 1, 1, bias=False)

    def forward(self, entropy_map):
        out = self.conv1(entropy_map)
        out = self.conv2(out) + entropy_map
        uncert = self.uncert_head(out)
        occ = self.occ_head(out)
        return uncert, occ


class HypothesisWeightNet(nn.Module):
    """Predict a residual source weight for every depth hypothesis."""

    def __init__(self, in_channels=11):
        super(HypothesisWeightNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(in_channels, 8, 3, 1, 1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        )
        self.logit = nn.Conv3d(8, 1, 1, 1, 0, bias=True)
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)

    def forward(self, features):
        return self.logit(self.feature(features))


# =============================================================================
# Single Cascaded Stage
# =============================================================================
class SingleStage(nn.Module):
    """One stage of the cascaded pipeline.

    For each source view:
      1. Warp + groupwise correlation → pair cost volume [B, 8, D, H, W]
      2. 3D regularization → intermediate features
      3. Score → softmax → depth + entropy → uncertainty
      4. Uncertainty-weighted accumulation into fused cost volume
    After all pairs: final regularization → fused depth.
    """

    def __init__(self, hypothesis_fusion=False,
                 hypothesis_residual_scale=1.0):
        super(SingleStage, self).__init__()
        if hypothesis_residual_scale < 0.0:
            raise ValueError("hypothesis_residual_scale must be non-negative")
        self.reg = RegNet3D(8)
        self.reg_pair = RegPair()
        self.reg_fuse = RegFuse()
        self.uncert_net = UncertNet()
        self.hypothesis_fusion = hypothesis_fusion
        self.hypothesis_residual_scale = hypothesis_residual_scale
        self.hypothesis_weight_net = (
            HypothesisWeightNet() if hypothesis_fusion else None)

    @staticmethod
    def _normalized_depth_coordinate(depth_values, height, width):
        if depth_values.dim() == 2:
            values = depth_values.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, height, width)
        elif depth_values.dim() == 4:
            values = depth_values
        else:
            raise ValueError("depth_values must have shape [B,D] or [B,D,H,W]")
        depth_min = values[:, :1]
        depth_span = (values[:, -1:] - depth_min).clamp(min=1e-6)
        return 2.0 * (values - depth_min) / depth_span - 1.0

    def forward(self, ref_feat, ref_proj, src_feats, src_projs, depth_values, mode='soft'):
        """
        Args:
            ref_feat:  [B, C, H, W] reference feature
            ref_proj:  [B, 4, 4]   reference projection matrix
            src_feats: list of [B, C, H, W] source features
            src_projs: list of [B, 4, 4] source projection matrices
            depth_values: [B, D] or [B, D, H, W] depth hypotheses
            mode: fusion mode ('soft' | 'average' | 'maxpool' | 'uwta' | 'hard')

        Returns:
            final_depth:  [B, H, W] fused depth
            confidence:   [B, H, W] Vis-MVSNet prob_map (sum of probs within ±2 of argmin)
            pair_results: list of pair depth, uncertainty, visibility logit,
                and optional per-hypothesis visibility logits
            final_std: probability-weighted depth standard deviation [B,H,W]
        """
        B, C, H, W = ref_feat.shape
        D = depth_values.shape[1]

        ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)  # [B, C, D, H, W]

        pair_results = []

        # ---- fusion buffers ----
        if mode in ('soft', 'hard'):
            weight_depth = D if mode == 'soft' and self.hypothesis_fusion else 1
            weight_sum = torch.zeros(
                B, 1, weight_depth, H, W,
                device=ref_feat.device, dtype=ref_feat.dtype)
        fused_interm = torch.zeros(B, 8, D, H, W, device=ref_feat.device, dtype=ref_feat.dtype)

        # per-pair early initialisation for uwta / maxpool
        min_weight = None
        maxpool_init = True

        # ---- per source view ----
        for src_feat, src_proj in zip(src_feats, src_projs):
            warped_src = homo_warping(src_feat, src_proj, ref_proj, depth_values)
            cost_volume = groupwise_correlation(ref_volume, warped_src, 8, dim=1)   # [B, 8, D, H, W]

            interm = self.reg(cost_volume)                                           # [B, 8, D, H, W]
            score = self.reg_pair(interm).squeeze(1)                                 # [B, D, H, W]
            prob = F.softmax(score, dim=1)                                           # [B, D, H, W]
            est_depth = depth_regression(prob, depth_values)                         # [B, H, W]

            entropy_map = entropy(prob, dim=1, keepdim=True)                         # [B, 1, H, W]
            uncert, occ = self.uncert_net(entropy_map)                               # [B,1,H,W], [B,1,H,W]
            hypothesis_logit = None

            # ---- fuse intermediate features ----
            if mode == 'soft':
                if self.hypothesis_fusion:
                    depth_coordinate = self._normalized_depth_coordinate(
                        depth_values, H, W).unsqueeze(1)
                    hypothesis_features = torch.cat((
                        interm,
                        torch.tanh(score).unsqueeze(1),
                        prob.unsqueeze(1),
                        depth_coordinate,
                    ), dim=1)
                    hypothesis_logit = self.hypothesis_weight_net(
                        hypothesis_features)
                    residual = self.hypothesis_residual_scale * torch.tanh(
                        hypothesis_logit)
                    weight = (-uncert.unsqueeze(2) + residual).exp()
                else:
                    weight = (-uncert).exp().unsqueeze(2)        # [B, 1, 1, H, W]
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            elif mode == 'hard':
                weight = (uncert < 0).to(interm.dtype).unsqueeze(2) + 1e-4
                weight_sum = weight_sum + weight
                fused_interm = fused_interm + interm * weight
            elif mode == 'average':
                fused_interm = fused_interm + interm
            elif mode == 'uwta':
                w = uncert.unsqueeze(2)
                if min_weight is None:
                    min_weight = w
                    mask = torch.ones_like(w).to(interm.dtype)
                else:
                    mask = (w < min_weight).to(interm.dtype)
                    min_weight = w * mask + min_weight * (1 - mask)
                fused_interm = interm * mask + fused_interm * (1 - mask)
            elif mode == 'maxpool':
                if maxpool_init:
                    fused_interm = fused_interm + interm
                    maxpool_init = False
                else:
                    fused_interm = torch.max(fused_interm, interm)

            if hypothesis_logit is None:
                pair_results.append((est_depth, uncert, occ))
            else:
                pair_results.append((est_depth, uncert, occ, hypothesis_logit))

            if not self.training:
                del prob, score, interm, cost_volume, warped_src

        # ---- normalise fused features ----
        if mode == 'soft':
            fused_interm = fused_interm / weight_sum
        elif mode == 'hard':
            fused_interm = fused_interm / weight_sum
        elif mode == 'average':
            fused_interm = fused_interm / len(src_feats)

        # ---- final score after fusion ----
        final_score = self.reg_fuse(fused_interm).squeeze(1)     # [B, D, H, W]
        final_prob = F.softmax(final_score, dim=1)                # [B, D, H, W]
        final_depth = depth_regression(final_prob, depth_values)  # [B, H, W]
        confidence = prob_map(final_prob)                          # [B, H, W]
        if depth_values.dim() == 2:
            depth_volume = depth_values.unsqueeze(-1).unsqueeze(-1)
        else:
            depth_volume = depth_values
        final_variance = (
            final_prob * (depth_volume - final_depth.unsqueeze(1)).pow(2)
        ).sum(dim=1)
        final_std = torch.sqrt(final_variance.clamp(min=1e-12))

        return final_depth, confidence, pair_results, final_std


# =============================================================================
# Full Cascaded Vis-MVSNet Model
# =============================================================================
class VisMVSModel(nn.Module):
    """3-stage cascaded MVS network with pair-wise uncertainty-aware fusion.

    Stage 1 (1/8 scale): coarse depth, wide range
    Stage 2 (1/4 scale): refined depth, range narrowed by stage 1
    Stage 3 (1/2 scale): finest depth, range narrowed by stage 2
    """

    def __init__(self, mode='soft',
                 stage1_depth_num=48, stage1_interval_scale=4,
                 stage2_depth_num=32, stage2_interval_scale=2,
                 stage3_depth_num=16, stage3_interval_scale=1,
                 adaptive_range=False, range_sigma_scale=2.0,
                 range_min_scale=1.0, range_max_scale=2.0,
                 hypothesis_fusion=False,
                 hypothesis_residual_scales=(1.0, 1.0, 1.0)):
        super(VisMVSModel, self).__init__()
        if range_sigma_scale < 0.0:
            raise ValueError("range_sigma_scale must be non-negative")
        if range_min_scale <= 0.0 or range_max_scale < range_min_scale:
            raise ValueError(
                "range scales require 0 < range_min_scale <= range_max_scale")
        if len(hypothesis_residual_scales) != 3:
            raise ValueError("hypothesis_residual_scales must contain three values")
        self.feature = MultiScaleFeatureNet()
        self.stage1 = SingleStage(
            hypothesis_fusion=hypothesis_fusion,
            hypothesis_residual_scale=hypothesis_residual_scales[0])
        self.stage2 = SingleStage(
            hypothesis_fusion=hypothesis_fusion,
            hypothesis_residual_scale=hypothesis_residual_scales[1])
        self.stage3 = SingleStage(
            hypothesis_fusion=hypothesis_fusion,
            hypothesis_residual_scale=hypothesis_residual_scales[2])
        self.mode = mode
        self.adaptive_range = adaptive_range
        self.range_sigma_scale = range_sigma_scale
        self.range_min_scale = range_min_scale
        self.range_max_scale = range_max_scale
        self.hypothesis_fusion = hypothesis_fusion

        self.s1_dnum = stage1_depth_num
        self.s1_iscale = stage1_interval_scale
        self.s2_dnum = stage2_depth_num
        self.s2_iscale = stage2_interval_scale
        self.s3_dnum = stage3_depth_num
        self.s3_iscale = stage3_interval_scale

    def _build_depth_range(self, depth_start, depth_interval, depth_num, B, H, W, device, dtype):
        """Build spatially-uniform depth hypotheses [B, D]."""
        d = torch.arange(depth_num, device=device, dtype=dtype)
        if depth_start.dim() == 2:
            depth_start = depth_start.view(B, 1)
            depth_interval = depth_interval.view(B, 1)
        return depth_start + depth_interval * d  # [B, D]

    def _build_per_pixel_depth_range(self, pred_depth, depth_num, interval, B, H, W, device, dtype):
        """Build per-pixel depth hypotheses [B, D, H, W] centred on pred_depth."""
        half = depth_num // 2
        d = torch.arange(depth_num, device=device, dtype=dtype).view(1, depth_num, 1, 1)
        # interval is [B, 1]; add trailing dims so it broadcasts with [B,H,W] and [1,D,1,1]
        interval = interval.view(B, 1, 1)
        depth_start = pred_depth - half * interval
        if depth_start.dim() == 3:
            depth_start = depth_start.unsqueeze(1)  # [B, 1, H, W]
        return depth_start + interval.view(B, 1, 1, 1) * d  # [B, D, H, W]

    def _build_adaptive_depth_range(
            self, pred_depth, pred_std, depth_num, interval,
            global_min, global_max):
        """Build a conservative uncertainty-adaptive range [B,D,H,W]."""
        batch = pred_depth.shape[0]
        interval_map = interval.view(batch, 1, 1)
        fixed_half_width = (depth_num // 2) * interval_map
        uncertainty_half_width = self.range_sigma_scale * pred_std
        half_width = torch.maximum(
            uncertainty_half_width,
            self.range_min_scale * fixed_half_width)
        half_width = torch.minimum(
            half_width,
            self.range_max_scale * fixed_half_width)

        range_min = torch.maximum(
            pred_depth - half_width, global_min.view(batch, 1, 1))
        range_max = torch.minimum(
            pred_depth + half_width, global_max.view(batch, 1, 1))
        range_max = torch.maximum(range_max, range_min + 1e-6)

        position = torch.linspace(
            0.0, 1.0, depth_num,
            device=pred_depth.device, dtype=pred_depth.dtype,
        ).view(1, depth_num, 1, 1)
        return (
            range_min.unsqueeze(1) +
            (range_max - range_min).unsqueeze(1) * position)

    def forward(self, imgs, proj_matrices, depth_values_orig):
        """
        Args:
            imgs:               [B, V, 3, H, W]
            proj_matrices:      [B, V, 4, 4]
            depth_values_orig:  [B, D_orig]  original depth hypotheses from dataset

        Returns:
            outputs:     list of [stage_depth, pair_results] per stage
            final_depth: [B, 1, H, W]
            conf_maps:   list of [conf_1, conf_2, conf_3] upsampled confidence maps [B, 1, H, W]
        """
        imgs_list = list(torch.unbind(imgs, 1))
        proj_list = list(torch.unbind(proj_matrices, 1))
        ref_img, src_imgs = imgs_list[0], imgs_list[1:]
        ref_proj, src_projs = proj_list[0], proj_list[1:]

        B, _, H_full, W_full = ref_img.shape
        D_orig = depth_values_orig.shape[1]
        device = ref_img.device
        dtype = ref_img.dtype

        # ---- multi-scale feature extraction (all views batched) ----
        V = len(imgs_list)
        all_imgs = torch.cat([ref_img] + src_imgs, dim=0)  # [V*B, 3, H, W]
        feat8_all, feat4_all, feat2_all = self.feature(all_imgs)

        def split_feats(f_all):
            return [f_all[i * B:(i + 1) * B] for i in range(V)]

        ref_feat8, *srcs_feat8 = split_feats(feat8_all)
        ref_feat4, *srcs_feat4 = split_feats(feat4_all)
        ref_feat2, *srcs_feat2 = split_feats(feat2_all)

        # ---- base depth params ----
        base_interval = depth_values_orig[:, 1:2] - depth_values_orig[:, 0:1]  # [B, 1]
        base_start = depth_values_orig[:, 0:1]                                  # [B, 1]
        global_max = depth_values_orig[:, -1:]

        def _scale_proj(proj, s):
            """Scale the first two rows of a projection matrix by factor s.

            Dataset proj_matrices have intrinsics at 1/4 image scale.
            For a feature map at 1/S scale, multiply by (4/S) to convert.
            """
            p = proj.clone()
            p[:, :2, :] *= s
            return p

        # =====================================================================
        # Stage 1 — 1/8 scale, coarsest
        # =====================================================================
        # intrinsics are at 1/4 scale; features at 1/8 → scale by 0.5
        s1_ref_proj = _scale_proj(ref_proj, 0.5)
        s1_src_projs = [_scale_proj(p, 0.5) for p in src_projs]

        s1_interval = base_interval * self.s1_iscale
        depth_values_1 = self._build_depth_range(
            base_start, s1_interval, self.s1_dnum, B,
            ref_feat8.shape[2], ref_feat8.shape[3], device, dtype)  # [B, D1]

        depth_1, conf_1, pairs_1, std_1 = self.stage1(
            ref_feat8, s1_ref_proj, srcs_feat8, s1_src_projs,
            depth_values_1, mode=self.mode)

        conf_1_up = F.interpolate(conf_1.unsqueeze(1), scale_factor=4, mode='bilinear',
                                  align_corners=False)  # [B,1,4H8,4W8]

        # =====================================================================
        # Stage 2 — 1/4 scale, range narrowed by stage 1
        # =====================================================================
        H4, W4 = ref_feat4.shape[2], ref_feat4.shape[3]
        depth_1_up = F.interpolate(depth_1.unsqueeze(1), size=(H4, W4),
                                   mode='bilinear', align_corners=False).squeeze(1)  # [B, H4, W4]
        std_1_up = F.interpolate(
            std_1.unsqueeze(1), size=(H4, W4),
            mode='bilinear', align_corners=False).squeeze(1)

        s2_interval = base_interval * self.s2_iscale       # per-pixel interval [B, 1]
        if self.adaptive_range:
            depth_values_2 = self._build_adaptive_depth_range(
                depth_1_up, std_1_up, self.s2_dnum, s2_interval,
                base_start, global_max)
        else:
            depth_values_2 = self._build_per_pixel_depth_range(
                depth_1_up, self.s2_dnum, s2_interval,
                B, H4, W4, device, dtype)  # [B, D2, H4, W4]

        depth_2, conf_2, pairs_2, std_2 = self.stage2(
            ref_feat4, ref_proj, srcs_feat4, src_projs,
            depth_values_2, mode=self.mode)

        conf_2_up = F.interpolate(conf_2.unsqueeze(1), scale_factor=2, mode='bilinear',
                                  align_corners=False)  # [B,1,2H4,2W4]

        # =====================================================================
        # Stage 3 — 1/2 scale, finest
        # =====================================================================
        # intrinsics are at 1/4 scale; features at 1/2 → scale by 2.0
        s3_ref_proj = _scale_proj(ref_proj, 2.0)
        s3_src_projs = [_scale_proj(p, 2.0) for p in src_projs]

        H2, W2 = ref_feat2.shape[2], ref_feat2.shape[3]
        depth_2_up = F.interpolate(depth_2.unsqueeze(1), size=(H2, W2),
                                   mode='bilinear', align_corners=False).squeeze(1)  # [B, H2, W2]
        std_2_up = F.interpolate(
            std_2.unsqueeze(1), size=(H2, W2),
            mode='bilinear', align_corners=False).squeeze(1)

        s3_interval = base_interval * self.s3_iscale
        if self.adaptive_range:
            depth_values_3 = self._build_adaptive_depth_range(
                depth_2_up, std_2_up, self.s3_dnum, s3_interval,
                base_start, global_max)
        else:
            depth_values_3 = self._build_per_pixel_depth_range(
                depth_2_up, self.s3_dnum, s3_interval,
                B, H2, W2, device, dtype)  # [B, D3, H2, W2]

        depth_3, conf_3, pairs_3, _ = self.stage3(
            ref_feat2, s3_ref_proj, srcs_feat2, s3_src_projs,
            depth_values_3, mode=self.mode)

        conf_3_up = conf_3.unsqueeze(1)  # [B, 1, H2, W2]

        # ---- final depth at stage 3 scale ----
        final_depth = depth_3.unsqueeze(1)  # [B, 1, H2, W2]

        # Keep the hypotheses used by every stage. They add no model parameters
        # and make cascade range coverage directly measurable during training.
        outputs = [
            [depth_1, pairs_1, depth_values_1],
            [depth_2, pairs_2, depth_values_2],
            [depth_3, pairs_3, depth_values_3],
        ]

        return outputs, final_depth, [conf_1_up, conf_2_up, conf_3_up]


# =============================================================================
# Loss
# =============================================================================
class VisMVSLoss(nn.Module):
    """Configurable objective for the independent A/B/C ablation.

    With ``occlusion_aware_supervision=False``, fused depth, pair depth and
    uncertainty reproduce the original Vis-MVSNet objective. Enabling it adds
    factor A: source-specific pair masking, detached occluded pair errors and
    true visibility supervision for the original occ head. Factor C has its
    own hypothesis-visibility loss and can be trained with factor A disabled.
    """

    def __init__(self, pair_l1_weight=1.0, uncertainty_weight=1.0,
                 visibility_weight=0.2, visibility_focal_gamma=2.0,
                 hypothesis_visibility_weight=0.0,
                 occ_abs_tol=2.0, occ_rel_tol=0.01,
                 stage_weights=(0.5, 1.0, 2.0),
                 occlusion_aware_supervision=True):
        super(VisMVSLoss, self).__init__()
        if len(stage_weights) != 3:
            raise ValueError("stage_weights must contain three values")
        if min(pair_l1_weight, uncertainty_weight, visibility_weight,
               hypothesis_visibility_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if visibility_focal_gamma < 0.0:
            raise ValueError("visibility_focal_gamma must be non-negative")
        if occ_abs_tol < 0.0 or occ_rel_tol < 0.0:
            raise ValueError("occlusion tolerances must be non-negative")

        self.stage_weights = tuple(stage_weights)
        self.pair_l1_weight = pair_l1_weight
        self.uncertainty_weight = uncertainty_weight
        self.visibility_weight = visibility_weight
        self.hypothesis_visibility_weight = hypothesis_visibility_weight
        self.visibility_focal_gamma = visibility_focal_gamma
        self.occ_abs_tol = occ_abs_tol
        self.occ_rel_tol = occ_rel_tol
        self.occlusion_aware_supervision = occlusion_aware_supervision

    @staticmethod
    def _masked_mean(values, valid):
        valid = valid.bool()
        if not valid.any():
            return values.sum() * 0.0
        return values[valid].mean()

    def _balanced_focal_bce(self, logits, target, supervised):
        supervised = supervised.bool()
        if not supervised.any():
            return logits.sum() * 0.0

        target_values = target[supervised]
        visible_fraction = target_values.mean()
        visible_weight = (0.5 / visible_fraction.clamp(min=1e-3)).clamp(max=10.0)
        occluded_weight = (
            0.5 / (1.0 - visible_fraction).clamp(min=1e-3)
        ).clamp(max=10.0)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        probability = torch.sigmoid(logits)
        target_probability = (
            probability * target + (1.0 - probability) * (1.0 - target)
        )
        focal = (1.0 - target_probability).pow(self.visibility_focal_gamma)
        class_weight = target * visible_weight + (1.0 - target) * occluded_weight
        weighted = bce * focal * class_weight
        return weighted[supervised].sum() / class_weight[supervised].sum().clamp(min=1e-6)

    @staticmethod
    def _depth_range_coverage(depth_hypotheses, gt_depth, valid):
        if depth_hypotheses.dim() == 2:
            depth_min = depth_hypotheses[:, 0].view(-1, 1, 1)
            depth_max = depth_hypotheses[:, -1].view(-1, 1, 1)
        elif depth_hypotheses.dim() == 4:
            depth_min = depth_hypotheses[:, 0]
            depth_max = depth_hypotheses[:, -1]
        else:
            raise ValueError("depth hypotheses must have shape [B,D] or [B,D,H,W]")
        in_range = (gt_depth >= depth_min) & (gt_depth <= depth_max)
        if not valid.any():
            return gt_depth.sum() * 0.0
        return in_range[valid].float().mean()

    @staticmethod
    def _hypothesis_logit_at_gt(
            hypothesis_logit, depth_hypotheses, gt_depth):
        logits = hypothesis_logit.squeeze(1)
        if depth_hypotheses.dim() == 2:
            hypotheses = depth_hypotheses.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, gt_depth.shape[-2], gt_depth.shape[-1])
        elif depth_hypotheses.dim() == 4:
            hypotheses = depth_hypotheses
        else:
            raise ValueError("depth hypotheses must have shape [B,D] or [B,D,H,W]")
        nearest = (hypotheses - gt_depth.unsqueeze(1)).abs().argmin(
            dim=1, keepdim=True)
        return torch.gather(logits, 1, nearest).squeeze(1)

    def forward(self, stage_outputs, depth_gt, mask, depth_interval,
                visibility_depths=None, visibility_masks=None,
                proj_matrices=None):
        if depth_interval.dim() == 0:
            depth_interval = depth_interval.view(1)
        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        needs_pair_visibility = (
            self.occlusion_aware_supervision or
            self.hypothesis_visibility_weight > 0.0
        )
        if needs_pair_visibility:
            if (visibility_depths is None or visibility_masks is None or
                    proj_matrices is None):
                raise ValueError(
                    "A or C supervision requires source-view visibility GT")
            if visibility_depths.dim() != 4 or visibility_masks.dim() != 4:
                raise ValueError("visibility GT must have shape [B,V,H,W]")
            if proj_matrices.dim() != 4:
                raise ValueError("proj_matrices must have shape [B,V,4,4]")

        projection_scales = (0.5, 1.0, 2.0)
        stage_losses = []
        scalar_stats = {}

        aggregate_predictions = []
        aggregate_targets = []
        aggregate_hypothesis_predictions = []
        aggregate_hypothesis_targets = []

        for stage_idx, (stage_output, stage_weight, projection_scale) in enumerate(zip(
                stage_outputs, self.stage_weights, projection_scales)):
            if len(stage_output) < 3:
                raise ValueError(
                    "OA model stage output must contain depth, pair results and hypotheses")
            est_depth, pair_results, depth_hypotheses = stage_output[:3]
            _, height, width = est_depth.shape
            stage_number = stage_idx + 1

            gt_ds = F.interpolate(
                depth_gt, size=(height, width), mode='bilinear',
                align_corners=False).squeeze(1)
            valid = F.interpolate(
                mask, size=(height, width), mode='nearest').squeeze(1) > 0.5
            if needs_pair_visibility:
                view_depths = F.interpolate(
                    visibility_depths, size=(height, width), mode='nearest')
                view_masks = F.interpolate(
                    visibility_masks, size=(height, width), mode='nearest') > 0.5

                stage_projs = proj_matrices.clone()
                stage_projs[:, :, :2, :] *= projection_scale
                ref_depth = view_depths[:, 0]
                ref_valid = view_masks[:, 0]
                ref_proj = stage_projs[:, 0]

            abs_err_scaled = (
                (est_depth - gt_ds).abs() /
                depth_interval.view(-1, 1, 1)
            )
            fused_l1 = self._masked_mean(abs_err_scaled, valid)

            pair_l1_losses = []
            uncertainty_losses = []
            visibility_losses = []
            hypothesis_visibility_losses = []
            visible_count = 0
            occluded_count = 0
            supervised_count = 0
            possible_count = 0

            for pair_idx, pair_result in enumerate(pair_results):
                if len(pair_result) == 3:
                    pair_depth, pair_uncert, pair_visibility_logit = pair_result
                    hypothesis_logit = None
                elif len(pair_result) == 4:
                    pair_depth, pair_uncert, pair_visibility_logit, hypothesis_logit = pair_result
                else:
                    raise ValueError("pair result must contain three or four tensors")
                if needs_pair_visibility:
                    source_idx = pair_idx + 1
                    if source_idx >= view_depths.shape[1]:
                        raise ValueError(
                            "not enough source-view GT maps for pair supervision")

                    target, supervised = ground_truth_pair_visibility(
                        ref_depth, view_depths[:, source_idx],
                        ref_valid, view_masks[:, source_idx],
                        ref_proj, stage_projs[:, source_idx], height, width,
                        occ_abs_tol=self.occ_abs_tol,
                        occ_rel_tol=self.occ_rel_tol,
                    )
                    target = target.squeeze(1)
                    supervised = supervised.squeeze(1).bool()
                    visible = supervised & (target > 0.5)
                    occluded = supervised & ~visible

                pair_error = (
                    (pair_depth - gt_ds).abs() /
                    depth_interval.view(-1, 1, 1)
                )
                uncertainty = pair_uncert.squeeze(1)
                if self.occlusion_aware_supervision:
                    pair_l1_losses.append(
                        self._masked_mean(pair_error, visible))
                    # Visible errors train both depth and uncertainty. Occluded
                    # errors train uncertainty only, because no match exists.
                    uncertainty_error = torch.where(
                        visible, pair_error, pair_error.detach())
                    uncertainty_nll = (
                        uncertainty_error * (-uncertainty).exp() + uncertainty
                    )
                    uncertainty_losses.append(
                        self._masked_mean(uncertainty_nll, supervised))
                    visibility_losses.append(self._balanced_focal_bce(
                        pair_visibility_logit.squeeze(1), target, supervised))
                else:
                    pair_l1_losses.append(
                        self._masked_mean(pair_error, valid))
                    uncertainty_nll = (
                        pair_error * (-uncertainty).exp() + uncertainty
                    )
                    uncertainty_losses.append(
                        self._masked_mean(uncertainty_nll, valid))

                if (hypothesis_logit is not None and
                        self.hypothesis_visibility_weight > 0.0):
                    hypothesis_gt_logit = self._hypothesis_logit_at_gt(
                        hypothesis_logit, depth_hypotheses, gt_ds)
                    hypothesis_visibility_losses.append(
                        self._balanced_focal_bce(
                            hypothesis_gt_logit, target, supervised))
                    if supervised.any():
                        aggregate_hypothesis_predictions.append(
                            torch.sigmoid(hypothesis_gt_logit)[supervised].detach())
                        aggregate_hypothesis_targets.append(
                            target[supervised].detach())

                if needs_pair_visibility:
                    visible_count += int(visible.sum().item())
                    occluded_count += int(occluded.sum().item())
                    supervised_count += int(supervised.sum().item())
                    possible_count += supervised.numel()
                if self.occlusion_aware_supervision and supervised.any():
                    aggregate_predictions.append(
                        torch.sigmoid(pair_visibility_logit.squeeze(1))[supervised].detach())
                    aggregate_targets.append(target[supervised].detach())

            if not pair_l1_losses:
                raise ValueError("at least one source view is required")

            pair_l1 = sum(pair_l1_losses) / len(pair_l1_losses)
            uncertainty_loss = sum(uncertainty_losses) / len(uncertainty_losses)
            visibility_loss = (
                sum(visibility_losses) / len(visibility_losses)
                if visibility_losses else fused_l1.new_zeros(())
            )
            if hypothesis_visibility_losses:
                hypothesis_visibility_loss = (
                    sum(hypothesis_visibility_losses) /
                    len(hypothesis_visibility_losses))
            else:
                hypothesis_visibility_loss = fused_l1.new_zeros(())
                if self.hypothesis_visibility_weight > 0.0:
                    raise ValueError(
                        "hypothesis visibility loss requires hypothesis fusion")
            stage_loss = (
                fused_l1 +
                self.pair_l1_weight * pair_l1 +
                self.uncertainty_weight * uncertainty_loss +
                self.visibility_weight * visibility_loss +
                self.hypothesis_visibility_weight * hypothesis_visibility_loss
            )
            stage_losses.append(stage_weight * stage_loss)

            scalar_stats['l1_stage{}'.format(stage_number)] = fused_l1
            scalar_stats['pair_l1_stage{}'.format(stage_number)] = pair_l1
            scalar_stats['uncertainty_loss_stage{}'.format(stage_number)] = uncertainty_loss
            if self.occlusion_aware_supervision:
                scalar_stats[
                    'visibility_loss_stage{}'.format(stage_number)
                ] = visibility_loss
            if hypothesis_visibility_losses:
                scalar_stats[
                    'hypothesis_visibility_loss_stage{}'.format(stage_number)
                ] = hypothesis_visibility_loss
            if needs_pair_visibility:
                scalar_stats[
                    'pair_supervised_ratio_stage{}'.format(stage_number)
                ] = est_depth.new_tensor(
                    supervised_count / max(possible_count, 1))
                scalar_stats[
                    'pair_visible_ratio_stage{}'.format(stage_number)
                ] = est_depth.new_tensor(
                    visible_count / max(supervised_count, 1))
                scalar_stats[
                    'pair_occluded_ratio_stage{}'.format(stage_number)
                ] = est_depth.new_tensor(
                    occluded_count / max(supervised_count, 1))
            scalar_stats['range_coverage_stage{}'.format(stage_number)] = (
                self._depth_range_coverage(depth_hypotheses, gt_ds, valid))
            if depth_hypotheses.dim() == 2:
                range_width = (
                    depth_hypotheses[:, -1] - depth_hypotheses[:, 0]
                ).view(-1, 1, 1).expand_as(gt_ds)
            else:
                range_width = depth_hypotheses[:, -1] - depth_hypotheses[:, 0]
            scalar_stats['range_width_intervals_stage{}'.format(stage_number)] = (
                self._masked_mean(
                    range_width / depth_interval.view(-1, 1, 1), valid))

            if stage_idx == 2:
                scalar_stats['less1_stage3'] = (
                    abs_err_scaled[valid] < 1.0).float().mean()
                scalar_stats['less3_stage3'] = (
                    abs_err_scaled[valid] < 3.0).float().mean()

        total_loss = sum(stage_losses)
        scalar_stats['loss'] = total_loss

        if aggregate_predictions:
            prediction = torch.cat(aggregate_predictions)
            target = torch.cat(aggregate_targets) >= 0.5
            predicted_visible = prediction >= 0.5
            scalar_stats['visibility_accuracy'] = (
                predicted_visible == target).float().mean()
            if target.any():
                scalar_stats['visibility_visible_recall'] = (
                    predicted_visible[target].float().mean())
            if (~target).any():
                scalar_stats['visibility_occluded_recall'] = (
                    (~predicted_visible[~target]).float().mean())

        if aggregate_hypothesis_predictions:
            prediction = torch.cat(aggregate_hypothesis_predictions)
            target = torch.cat(aggregate_hypothesis_targets) >= 0.5
            predicted_visible = prediction >= 0.5
            scalar_stats['hypothesis_visibility_accuracy'] = (
                predicted_visible == target).float().mean()
            if target.any():
                scalar_stats['hypothesis_visibility_visible_recall'] = (
                    predicted_visible[target].float().mean())
            if (~target).any():
                scalar_stats['hypothesis_visibility_occluded_recall'] = (
                    (~predicted_visible[~target]).float().mean())

        return total_loss, scalar_stats
