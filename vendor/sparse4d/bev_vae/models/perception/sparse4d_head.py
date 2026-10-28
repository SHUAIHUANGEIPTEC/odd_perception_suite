"""Sparse4D detection with optional recurrent temporal fusion, without MMCV.

BEV-VAE box coordinates and the FP32 geometry/head boundary are preserved.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from bev_vae.models.perception.temporal_memory import TemporalMemory

try:
    from bev_vae.models.ops.deformable_aggregation import (
        CUDA_DEFORMABLE_AGGREGATION_AVAILABLE,
        deformable_aggregation,
        format_feature_maps,
    )
except ImportError:
    CUDA_DEFORMABLE_AGGREGATION_AVAILABLE = False
    deformable_aggregation = None
    format_feature_maps = None


X, Y, Z, L, W, H, SIN_YAW, COS_YAW, VX, VY, VZ = range(11)
CENTERNESS, YAWNESS = 0, 1


def _reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / dist.get_world_size()


def _linear_relu_ln(
    input_dims: int,
    output_dims: int,
    in_loops: int,
    out_loops: int,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    current_dims = input_dims
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(current_dims, output_dims))
            layers.append(nn.ReLU(inplace=True))
            current_dims = output_dims
        layers.append(nn.LayerNorm(output_dims))
    return nn.Sequential(*layers)


class Scale(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class SingleFrameInstanceBank(nn.Module):
    """Learnable Sparse4D anchors and their per-anchor instance features."""

    def __init__(
        self,
        anchor: Union[str, Path, Sequence[Sequence[float]], np.ndarray],
        num_anchor: int,
        embed_dims: int,
        anchor_grad: bool = True,
        feat_grad: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(anchor, (str, Path)):
            anchor_path = Path(anchor).expanduser()
            if not anchor_path.is_file():
                raise FileNotFoundError(
                    f"Sparse4D anchor file does not exist: {anchor_path}. "
                    "Generate it with tools/generate_sparse4d_anchors.py or set "
                    "SPARSE4D_ANCHOR_PATH."
                )
            anchor_array = np.load(anchor_path)
        else:
            anchor_array = np.asarray(anchor, dtype=np.float32)

        if anchor_array.ndim != 2 or anchor_array.shape[1] not in (10, 11):
            raise ValueError(
                "Sparse4D anchors must have shape [N, 10] or [N, 11], got "
                f"{anchor_array.shape}"
            )
        if len(anchor_array) < num_anchor:
            raise ValueError(
                f"Anchor file contains {len(anchor_array)} anchors, fewer than "
                f"the configured num_anchor={num_anchor}"
            )
        anchor_array = anchor_array[:num_anchor]
        if anchor_array.shape[1] == 10:
            anchor_array = np.pad(anchor_array, ((0, 0), (0, 1)))

        anchor_tensor = torch.as_tensor(anchor_array, dtype=torch.float32)
        self.num_anchor = int(num_anchor)
        self.embed_dims = int(embed_dims)
        self.anchor = nn.Parameter(anchor_tensor, requires_grad=anchor_grad)
        self.instance_feature = nn.Parameter(
            torch.zeros(self.num_anchor, self.embed_dims),
            requires_grad=feat_grad,
        )
        # Match Sparse4D's InstanceBank: frozen instance features start from
        # zero; Xavier initialization is only used when the feature itself is
        # configured as a learnable parameter.
        if feat_grad:
            nn.init.xavier_uniform_(self.instance_feature)

    def get(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        instance_feature = self.instance_feature.unsqueeze(0).expand(batch_size, -1, -1)
        anchor = self.anchor.unsqueeze(0).expand(batch_size, -1, -1)
        return instance_feature, anchor


class SparseBox3DEncoder(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        component_dims: Optional[Sequence[int]] = None,
        vel_dims: int = 3,
    ) -> None:
        super().__init__()
        if component_dims is None:
            pos_dims = embed_dims // 2
            size_dims = embed_dims // 8
            yaw_dims = embed_dims // 8
            vel_embed_dims = embed_dims - pos_dims - size_dims - yaw_dims
            component_dims = (pos_dims, size_dims, yaw_dims, vel_embed_dims)
        if len(component_dims) != 4 or sum(component_dims) != embed_dims:
            raise ValueError(
                "anchor component_dims must contain four values summing to "
                f"embed_dims={embed_dims}, got {component_dims}"
            )
        self.vel_dims = int(vel_dims)
        self.pos_fc = _linear_relu_ln(3, component_dims[0], 1, 4)
        self.size_fc = _linear_relu_ln(3, component_dims[1], 1, 4)
        self.yaw_fc = _linear_relu_ln(2, component_dims[2], 1, 4)
        self.vel_fc = _linear_relu_ln(self.vel_dims, component_dims[3], 1, 4)

    def forward(self, box_3d: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.pos_fc(box_3d[..., [X, Y, Z]]),
                self.size_fc(box_3d[..., [L, W, H]]),
                self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]]),
                self.vel_fc(box_3d[..., VX : VX + self.vel_dims]),
            ],
            dim=-1,
        )


class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_learnable_pts: int = 6,
        fix_scale: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        super().__init__()
        if fix_scale is None:
            fix_scale = (
                (0.0, 0.0, 0.0),
                (0.45, 0.0, 0.0),
                (-0.45, 0.0, 0.0),
                (0.0, 0.45, 0.0),
                (0.0, -0.45, 0.0),
                (0.0, 0.0, 0.45),
                (0.0, 0.0, -0.45),
            )
        self.register_buffer("fix_scale", torch.tensor(fix_scale, dtype=torch.float32))
        self.num_learnable_pts = int(num_learnable_pts)
        self.num_pts = len(fix_scale) + self.num_learnable_pts
        self.learnable_fc = (
            nn.Linear(embed_dims, self.num_learnable_pts * 3)
            if self.num_learnable_pts > 0
            else None
        )
        if self.learnable_fc is not None:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            nn.init.zeros_(self.learnable_fc.bias)

    def forward(
        self,
        anchor: torch.Tensor,
        instance_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [L, W, H]].exp()
        key_points = self.fix_scale.to(dtype=anchor.dtype) * size
        key_points = key_points.expand(batch_size, num_anchor, -1, -1)

        if self.learnable_fc is not None and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(batch_size, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)

        rotation = anchor.new_zeros(batch_size, num_anchor, 3, 3)
        rotation[..., 0, 0] = anchor[..., COS_YAW]
        rotation[..., 0, 1] = -anchor[..., SIN_YAW]
        rotation[..., 1, 0] = anchor[..., SIN_YAW]
        rotation[..., 1, 1] = anchor[..., COS_YAW]
        rotation[..., 2, 2] = 1.0
        key_points = torch.matmul(rotation[:, :, None], key_points[..., None]).squeeze(-1)
        return key_points + anchor[..., None, [X, Y, Z]]


class DeformableFeatureAggregation(nn.Module):
    """Sparse4Dv3 multi-camera/multi-level aggregation without MMCV."""

    def __init__(
        self,
        embed_dims: int,
        num_groups: int,
        num_levels: int,
        num_cams: int,
        num_learnable_pts: int = 6,
        fix_scale: Optional[Sequence[Sequence[float]]] = None,
        attn_drop: float = 0.15,
        proj_drop: float = 0.0,
        use_camera_embed: bool = True,
        use_deformable_func: bool = False,
    ) -> None:
        super().__init__()
        if embed_dims % num_groups != 0:
            raise ValueError("embed_dims must be divisible by num_groups")
        if use_deformable_func and not CUDA_DEFORMABLE_AGGREGATION_AVAILABLE:
            raise ImportError(
                "use_deformable_func=True, but the optional CUDA extension is not "
                "available. Build it with `python "
                "bev_vae/models/ops/deformable_aggregation/setup.py build_ext --inplace`."
            )
        self.embed_dims = int(embed_dims)
        self.num_groups = int(num_groups)
        self.group_dims = self.embed_dims // self.num_groups
        self.num_levels = int(num_levels)
        self.num_cams = int(num_cams)
        self.attn_drop = float(attn_drop)
        self.use_deformable_func = bool(use_deformable_func)
        self.kps_generator = SparseBox3DKeyPointsGenerator(
            embed_dims,
            num_learnable_pts=num_learnable_pts,
            fix_scale=fix_scale,
        )
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.camera_encoder = (
            _linear_relu_ln(12, embed_dims, 1, 2) if use_camera_embed else None
        )
        weight_input_dims = embed_dims
        if self.camera_encoder is not None:
            weight_output_dims = num_groups * num_levels * self.num_pts
        else:
            weight_output_dims = num_groups * num_cams * num_levels * self.num_pts
        self.weights_fc = nn.Linear(weight_input_dims, weight_output_dims)
        nn.init.zeros_(self.weights_fc.weight)
        nn.init.zeros_(self.weights_fc.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def build_projection_matrices(img_metas: Dict[str, torch.Tensor]) -> torch.Tensor:
        # .float() alone does not protect matmul from an outer AMP context.
        with torch.autocast(device_type=img_metas["extrinsic"].device.type, enabled=False):
            extrinsic = img_metas["extrinsic"].float()
            intrinsic = img_metas["intrinsic"].float()
            geometric = img_metas["geometric"].float()
            ego_to_camera = torch.linalg.inv(extrinsic)[..., :3, :]
            return geometric @ intrinsic @ ego_to_camera

    @staticmethod
    def project_points(
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: torch.Tensor,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=key_points.device.type, enabled=False):
            points = key_points.float()
            points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            projected = torch.matmul(
                projection_mat.float()[:, :, None, None],
                points_h[:, None, ..., None],
            ).squeeze(-1)
            depth = projected[..., 2]
            in_front = depth > eps
            # Behind-camera points never contribute; avoid division by eps.
            # FP32 above also protects near-plane positive depths and large
            # projected numerators from FP16 overflow before validity masking.
            safe_depth = torch.where(in_front, depth, torch.ones_like(depth))
            points_2d = projected[..., :2] / safe_depth[..., None]
            points_2d = points_2d / image_wh.float()[:, :, None, None]
            valid = (
                in_front
                & torch.isfinite(points_2d).all(dim=-1)
                & (points_2d[..., 0] > 0.0)
                & (points_2d[..., 0] < 1.0)
                & (points_2d[..., 1] > 0.0)
                & (points_2d[..., 1] < 1.0)
            )
            points_2d = torch.where(
                valid[..., None], points_2d, points_2d.new_full((), -2.0)
            )
            return points_2d, valid

    def _get_weights(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        projection_mat: torch.Tensor,
        projection_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                projection_mat[:, :, :3].reshape(batch_size, self.num_cams, 12)
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = self.weights_fc(feature).reshape(
            batch_size,
            num_anchor,
            self.num_cams,
            self.num_levels,
            self.num_pts,
            self.num_groups,
        )
        flat_weights = weights.flatten(2, 4)
        if projection_valid is None:
            flat_weights = flat_weights.softmax(dim=2)
        else:
            # A key point is either geometrically visible in a camera or it
            # cannot contribute at any feature level.  The CUDA kernel skips
            # invalid coordinates, but an unmasked softmax still assigns most
            # of the cold-start attention mass to those skipped entries.  In a
            # single-frame model that unnecessarily weakens the only visual
            # signal available to the zero-initialized instance features.
            valid = projection_valid.permute(0, 2, 1, 3)
            valid = valid[:, :, :, None, :, None].expand_as(weights)
            flat_valid = valid.flatten(2, 4)
            flat_weights = flat_weights.float().masked_fill(
                ~flat_valid, torch.finfo(torch.float32).min
            ).softmax(dim=2)
            flat_weights = flat_weights * flat_valid.to(flat_weights.dtype)
            flat_weights = flat_weights / flat_weights.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-12)
            flat_weights = flat_weights.to(dtype=weights.dtype)
        weights = flat_weights.reshape_as(weights)
        if self.training and self.attn_drop > 0:
            keep = torch.rand(
                batch_size,
                num_anchor,
                self.num_cams,
                1,
                self.num_pts,
                1,
                device=weights.device,
            ) > self.attn_drop
            weights = weights * keep.to(weights.dtype) / (1.0 - self.attn_drop)
        return weights

    @staticmethod
    def feature_sampling(
        feature_maps: Sequence[torch.Tensor],
        points_2d: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_cams, num_anchor, num_pts = points_2d.shape[:4]
        grid = (points_2d * 2.0 - 1.0).flatten(0, 1)
        sampled_levels = []
        for feature_map in feature_maps:
            level_grid = grid.to(dtype=feature_map.dtype)
            sampled = F.grid_sample(
                feature_map.flatten(0, 1),
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(
                batch_size,
                num_cams,
                feature_map.shape[2],
                num_anchor,
                num_pts,
            ).permute(0, 3, 1, 4, 2)
            sampled_levels.append(sampled)
        return torch.stack(sampled_levels, dim=3)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: Sequence[torch.Tensor],
        img_metas: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if len(feature_maps) != self.num_levels:
            raise ValueError(
                f"Sparse4D expected {self.num_levels} feature levels, got {len(feature_maps)}"
            )
        if feature_maps[0].shape[1] != self.num_cams:
            raise ValueError(
                f"Sparse4D expected {self.num_cams} cameras, got {feature_maps[0].shape[1]}"
            )
        expected_prefix = (
            feature_maps[0].shape[0],
            self.num_cams,
            self.embed_dims,
        )
        for level_idx, feature_map in enumerate(feature_maps):
            if feature_map.ndim != 5 or feature_map.shape[:3] != expected_prefix:
                raise ValueError(
                    "Each Sparse4D feature level must have shape "
                    f"[B, {self.num_cams}, {self.embed_dims}, H, W]; level "
                    f"{level_idx} has {tuple(feature_map.shape)}"
                )
        projection_mat = self.build_projection_matrices(img_metas)
        if projection_mat.shape[:2] != expected_prefix[:2]:
            raise ValueError(
                "Camera calibration batch/camera dimensions do not match the "
                f"feature maps: {tuple(projection_mat.shape[:2])} versus "
                f"{expected_prefix[:2]}"
            )
        image_height, image_width = img_metas["img_shape"]
        image_wh = projection_mat.new_tensor([image_width, image_height])
        image_wh = image_wh.reshape(1, 1, 2).expand(
            projection_mat.shape[0], self.num_cams, -1
        )
        key_points = self.kps_generator(anchor, instance_feature)
        points_2d, projection_valid = self.project_points(
            key_points, projection_mat, image_wh
        )
        weights = self._get_weights(
            instance_feature,
            anchor_embed,
            projection_mat,
            projection_valid=projection_valid,
        )

        if self.use_deformable_func:
            formatted = format_feature_maps(feature_maps)
            cuda_weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous()
            sampling_location = points_2d.permute(0, 2, 3, 1, 4).contiguous()
            features = deformable_aggregation(
                formatted[0],
                formatted[1],
                formatted[2],
                sampling_location,
                cuda_weights,
            )
            features = features.to(dtype=instance_feature.dtype)
        else:
            features = self.feature_sampling(feature_maps, points_2d)
            features = features.reshape(
                *features.shape[:-1], self.num_groups, self.group_dims
            )
            features = (features * weights[..., None]).sum(dim=(2, 3, 4))
            features = features.reshape(
                instance_feature.shape[0], instance_feature.shape[1], self.embed_dims
            )

        # Keep these detached diagnostics out of the optimization graph.  A
        # numerically correct CUDA operator can still return zero features when
        # the camera/ego projection convention is wrong, which previously made
        # that failure indistinguishable from an optimization problem.
        visible_anchor = (
            projection_valid.permute(0, 2, 1, 3).flatten(2).any(dim=-1)
        )
        self.last_diagnostics = {
            "projection_valid_ratio": projection_valid.float().mean().detach(),
            "visible_anchor_ratio": visible_anchor.float().mean().detach(),
            "sampled_feature_abs_mean": features.float().abs().mean().detach(),
        }

        features = self.proj_drop(self.output_proj(features))
        return torch.cat([features, instance_feature], dim=-1)


class AsymmetricFFN(nn.Module):
    def __init__(self, embed_dims: int, dropout: float) -> None:
        super().__init__()
        input_dims = embed_dims * 2
        self.pre_norm = nn.LayerNorm(input_dims)
        self.layers = nn.Sequential(
            nn.Linear(input_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(dropout),
        )
        self.identity_fc = nn.Linear(input_dims, embed_dims)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.pre_norm(value)
        return self.identity_fc(value) + self.layers(value)


class SparseBox3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_classes: int,
        output_dims: int = 11,
        with_quality_estimation: bool = True,
    ) -> None:
        super().__init__()
        self.output_dims = int(output_dims)
        self.layers = nn.Sequential(
            _linear_relu_ln(embed_dims, embed_dims, 2, 2),
            nn.Linear(embed_dims, output_dims),
            Scale(output_dims),
        )
        self.cls_layers = nn.Sequential(
            _linear_relu_ln(embed_dims, embed_dims, 1, 2),
            nn.Linear(embed_dims, num_classes),
        )
        self.quality_layers = (
            nn.Sequential(
                _linear_relu_ln(embed_dims, embed_dims, 1, 2),
                nn.Linear(embed_dims, 2),
            )
            if with_quality_estimation
            else None
        )
        prior_prob = 0.01
        nn.init.constant_(
            self.cls_layers[-1].bias,
            -np.log((1.0 - prior_prob) / prior_prob),
        )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        refined_state = [X, Y, Z, L, W, H, SIN_YAW, COS_YAW]
        interval = torch.as_tensor(time_interval, dtype=output.dtype, device=output.device)
        if interval.ndim == 1:
            interval = interval[:, None, None]
        output = torch.cat(
            [
                output[..., :8] + anchor[..., refined_state],
                output[..., 8:] / interval + anchor[..., 8:],
            ],
            dim=-1,
        )
        classification = self.cls_layers(instance_feature)
        quality = self.quality_layers(feature) if self.quality_layers is not None else None
        return output, classification, quality


class DecoupledSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        dropout: float,
        decouple_attn: bool,
    ) -> None:
        super().__init__()
        self.decouple_attn = bool(decouple_attn)
        attention_dims = embed_dims * 2 if self.decouple_attn else embed_dims
        self.attention = nn.MultiheadAttention(
            attention_dims,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.value_proj = (
            nn.Linear(embed_dims, attention_dims, bias=False)
            if self.decouple_attn
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(attention_dims, embed_dims, bias=False)
            if self.decouple_attn
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.decouple_attn:
            query = torch.cat([instance_feature, anchor_embed], dim=-1)
        else:
            query = instance_feature + anchor_embed
        value = self.value_proj(instance_feature)
        attended = self.attention(
            query,
            query,
            value,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        return self.output_proj(query + self.dropout(attended))


class DecoupledTemporalAttention(DecoupledSelfAttention):
    def forward(self, instance_feature, anchor_embed, history_feature, history_embed, valid):
        if self.decouple_attn:
            query = torch.cat((instance_feature, anchor_embed), -1)
            key = torch.cat((history_feature, history_embed), -1)
        else:
            query = instance_feature + anchor_embed
            key = history_feature + history_embed
        attended = self.attention(query, key, self.value_proj(history_feature), need_weights=False)[0]
        output = self.output_proj(query + self.dropout(attended))
        # Cold-start slots cannot attend to another scene, even in a mixed batch.
        return torch.where(valid[:, None, None], output, instance_feature)


class Sparse4DDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_classes: int,
        num_groups: int,
        num_levels: int,
        num_cams: int,
        dropout: float,
        attn_drop: float,
        num_learnable_pts: int,
        fix_scale: Optional[Sequence[Sequence[float]]],
        decouple_attn: bool,
        with_self_attention: bool,
        with_quality_estimation: bool,
        use_deformable_func: bool,
        with_temporal_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_attention = (
            DecoupledSelfAttention(embed_dims, num_heads, dropout, decouple_attn)
            if with_self_attention
            else None
        )
        self.temporal_attention = (
            DecoupledTemporalAttention(embed_dims, num_heads, dropout, decouple_attn)
            if with_temporal_attention else None)
        self.attention_norm = nn.LayerNorm(embed_dims)
        self.deformable = DeformableFeatureAggregation(
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            num_learnable_pts=num_learnable_pts,
            fix_scale=fix_scale,
            attn_drop=attn_drop,
            use_camera_embed=True,
            use_deformable_func=use_deformable_func,
        )
        self.ffn = AsymmetricFFN(embed_dims, dropout)
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.refine = SparseBox3DRefinementModule(
            embed_dims,
            num_classes,
            with_quality_estimation=with_quality_estimation,
        )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: Sequence[torch.Tensor],
        img_metas: Dict[str, torch.Tensor],
        time_interval: float,
        attn_mask: Optional[torch.Tensor],
        temporal_context=None,
        temporal_embed=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.temporal_attention is not None and temporal_context is not None:
            instance_feature = self.temporal_attention(instance_feature, anchor_embed,
                temporal_context["feature"], temporal_embed, temporal_context["valid"])
        if self.self_attention is not None:
            instance_feature = self.self_attention(
                instance_feature, anchor_embed, attn_mask
            )
            instance_feature = self.attention_norm(instance_feature)
        instance_feature = self.deformable(
            instance_feature,
            anchor,
            anchor_embed,
            feature_maps,
            img_metas,
        )
        instance_feature = self.ffn_norm(self.ffn(instance_feature))
        anchor, classification, quality = self.refine(
            instance_feature,
            anchor,
            anchor_embed,
            time_interval,
        )
        return instance_feature, anchor, classification, quality


class SparseBox3DTarget:
    """Hungarian target assignment for BEV-VAE's encoded padded boxes."""

    def __init__(
        self,
        num_classes: int,
        cls_weight: float,
        box_weight: float,
        matching_reg_weights: Sequence[float],
        class_wise_reg_weights: Optional[Dict[Union[int, str], Sequence[float]]],
        num_dn_groups: int,
        dn_noise_scale: Sequence[float],
        max_dn_gt: int,
        add_neg_dn: bool,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        self.num_classes = int(num_classes)
        self.cls_weight = float(cls_weight)
        self.box_weight = float(box_weight)
        self.matching_reg_weights = tuple(float(x) for x in matching_reg_weights)
        self.class_wise_reg_weights = {
            int(key): tuple(float(x) for x in value)
            for key, value in (class_wise_reg_weights or {}).items()
        }
        self.num_dn_groups = int(num_dn_groups)
        self.dn_noise_scale = tuple(float(x) for x in dn_noise_scale)
        self.max_dn_gt = int(max_dn_gt)
        self.add_neg_dn = bool(add_neg_dn)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    @staticmethod
    def split_gt(
        gt_bboxes: torch.Tensor,
        state_dims: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        labels: List[torch.Tensor] = []
        boxes: List[torch.Tensor] = []
        valid = gt_bboxes[..., -1] != -1
        for batch_idx in range(gt_bboxes.shape[0]):
            batch_boxes = gt_bboxes[batch_idx, valid[batch_idx]]
            labels.append(batch_boxes[..., -1].long())
            encoded = batch_boxes[..., :-1]
            if encoded.shape[-1] < state_dims:
                encoded = F.pad(encoded, (0, state_dims - encoded.shape[-1]))
            boxes.append(encoded)
        return labels, boxes

    def instance_weights(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.isfinite(boxes).to(dtype=boxes.dtype)
        for class_id, class_weights in self.class_wise_reg_weights.items():
            class_weight = boxes.new_tensor(class_weights)
            weights[..., : len(class_weights)] = torch.where(
                (labels == class_id)[..., None],
                class_weight,
                weights[..., : len(class_weights)],
            )
        return weights

    def _cls_cost(
        self,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if cls_target.numel() == 0:
            return None
        probability = cls_pred.float().sigmoid().clamp(1e-8, 1.0 - 1e-8)
        neg_cost = -torch.log1p(-probability) * (1 - self.alpha) * probability.pow(self.gamma)
        pos_cost = -probability.log() * self.alpha * (1 - probability).pow(self.gamma)
        return (pos_cost[:, cls_target] - neg_cost[:, cls_target]) * self.cls_weight

    def sample(
        self,
        cls_pred: torch.Tensor,
        box_pred: torch.Tensor,
        gt_bboxes: torch.Tensor,
        return_matched_indices: bool = False,
    ):
        """Optionally append [B, N] original padded GT row indices (-1 unmatched)."""
        batch_size, num_pred, num_classes = cls_pred.shape
        labels_list, boxes_list = self.split_gt(gt_bboxes, box_pred.shape[-1])
        cls_target = cls_pred.new_full(
            (batch_size, num_pred), num_classes, dtype=torch.long
        )
        box_target = box_pred.new_zeros(box_pred.shape)
        reg_weight = box_pred.new_zeros(box_pred.shape)
        matching_weights = box_pred.new_tensor(self.matching_reg_weights)
        if return_matched_indices:
            matched_indices = cls_pred.new_full((batch_size, num_pred), -1, dtype=torch.long)

        for batch_idx in range(batch_size):
            labels = labels_list[batch_idx].to(cls_pred.device)
            boxes = boxes_list[batch_idx].to(box_pred.device, box_pred.dtype)
            if labels.numel() == 0:
                continue
            instance_weight = self.instance_weights(labels, boxes)
            cls_cost = self._cls_cost(cls_pred[batch_idx], labels)
            box_cost = (
                torch.abs(
                    box_pred[batch_idx, :, None, : len(matching_weights)]
                    - boxes[None, :, : len(matching_weights)]
                )
                * instance_weight[None, :, : len(matching_weights)]
                * matching_weights
            ).sum(dim=-1) * self.box_weight
            cost = torch.nan_to_num(cls_cost + box_cost, nan=1e8, posinf=1e8, neginf=-1e8)
            pred_indices, target_indices = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )
            pred_indices = torch.as_tensor(pred_indices, device=cls_pred.device)
            target_indices = torch.as_tensor(target_indices, device=cls_pred.device)
            cls_target[batch_idx, pred_indices] = labels[target_indices]
            box_target[batch_idx, pred_indices] = boxes[target_indices]
            reg_weight[batch_idx, pred_indices] = instance_weight[target_indices]
            if return_matched_indices:
                original_rows = (gt_bboxes[batch_idx, ..., -1] != -1).nonzero(as_tuple=False).flatten()
                matched_indices[batch_idx, pred_indices] = original_rows.to(cls_pred.device)[target_indices]
        if return_matched_indices:
            return cls_target, box_target, reg_weight, matched_indices
        return cls_target, box_target, reg_weight

    def get_dn_anchors(
        self,
        gt_bboxes: torch.Tensor,
        state_dims: int,
        synchronize: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.num_dn_groups <= 0:
            return None
        labels_list, boxes_list = self.split_gt(gt_bboxes, state_dims)
        labels_list = [labels[: self.max_dn_gt] for labels in labels_list]
        boxes_list = [boxes[: self.max_dn_gt] for boxes in boxes_list]
        max_gt = max((len(labels) for labels in labels_list), default=0)
        if synchronize and dist.is_available() and dist.is_initialized():
            # A rank with an empty scene must still participate in DN loss
            # normalization collectives executed by ranks with nonempty scenes.
            global_max = gt_bboxes.new_tensor(max_gt, dtype=torch.long)
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
            max_gt = int(global_max.item())
        if max_gt == 0:
            return None

        batch_size = gt_bboxes.shape[0]
        device = gt_bboxes.device
        boxes = gt_bboxes.new_zeros(batch_size, max_gt, state_dims)
        labels = torch.full(
            (batch_size, max_gt),
            self.num_classes,
            dtype=torch.long,
            device=device,
        )
        valid = torch.zeros(batch_size, max_gt, dtype=torch.bool, device=device)
        for batch_idx, (batch_labels, batch_boxes) in enumerate(zip(labels_list, boxes_list)):
            count = len(batch_labels)
            if count == 0:
                continue
            boxes[batch_idx, :count] = batch_boxes.to(device=device, dtype=boxes.dtype)
            labels[batch_idx, :count] = batch_labels.to(device=device)
            valid[batch_idx, :count] = True

        boxes = boxes[:, None].expand(-1, self.num_dn_groups, -1, -1)
        labels = labels[:, None].expand(-1, self.num_dn_groups, -1)
        valid = valid[:, None].expand(-1, self.num_dn_groups, -1)
        noise_scale = boxes.new_zeros(state_dims)
        configured_noise = boxes.new_tensor(self.dn_noise_scale)
        noise_scale[: len(configured_noise)] = configured_noise
        positive_noise = (torch.rand_like(boxes) * 2.0 - 1.0) * noise_scale
        dn_anchor = boxes + positive_noise

        if self.add_neg_dn:
            negative_noise = (torch.rand_like(boxes) + 1.0) * noise_scale
            negative_noise *= torch.where(
                torch.rand_like(boxes) > 0.5,
                boxes.new_tensor(1.0),
                boxes.new_tensor(-1.0),
            )
            dn_anchor = torch.cat([dn_anchor, boxes + negative_noise], dim=2)

        # Match the official Sparse4D DN target construction: after adding
        # noise, reassign the anchors to GT boxes by box cost.  Directly
        # retaining the source GT is incorrect when a noisy anchor moves
        # closer to another object.  Unassigned anchors are explicit
        # background samples; padded anchors remain invalid.
        anchors_per_group = dn_anchor.shape[2]
        dn_target = torch.zeros_like(dn_anchor)
        dn_labels = torch.full(
            dn_anchor.shape[:-1],
            self.num_classes,
            dtype=torch.long,
            device=device,
        )
        dn_valid = torch.zeros_like(dn_labels, dtype=torch.bool)
        matching_weights = dn_anchor.new_tensor(self.matching_reg_weights)
        matching_dims = min(len(matching_weights), state_dims)
        for batch_idx in range(batch_size):
            for group_idx in range(self.num_dn_groups):
                cost = (
                    torch.abs(
                        dn_anchor[batch_idx, group_idx, :, None, :matching_dims]
                        - boxes[batch_idx, group_idx, None, :, :matching_dims]
                    )
                    * matching_weights[:matching_dims]
                ).sum(dim=-1) * self.box_weight
                cost = torch.nan_to_num(
                    cost, nan=1e8, posinf=1e8, neginf=-1e8
                )
                anchor_indices, target_indices = linear_sum_assignment(
                    cost.detach().cpu().numpy()
                )
                anchor_indices = torch.as_tensor(anchor_indices, device=device)
                target_indices = torch.as_tensor(target_indices, device=device)
                target_valid = valid[batch_idx, group_idx, target_indices]
                dn_target[batch_idx, group_idx, anchor_indices] = boxes[
                    batch_idx, group_idx, target_indices
                ]
                dn_labels[batch_idx, group_idx, anchor_indices] = torch.where(
                    target_valid,
                    labels[batch_idx, group_idx, target_indices],
                    labels.new_tensor(self.num_classes),
                )
                dn_valid[batch_idx, group_idx, anchor_indices] = target_valid

        if self.add_neg_dn:
            # The official implementation also trains every non-padding DN
            # slot as either a matched positive or an explicit background.
            source_valid = torch.cat([valid, valid], dim=2)
            dn_valid = dn_valid | source_valid

        total_dn = self.num_dn_groups * anchors_per_group
        attention_mask = torch.ones(total_dn, total_dn, dtype=torch.bool, device=device)
        for group_idx in range(self.num_dn_groups):
            start = group_idx * anchors_per_group
            attention_mask[start : start + anchors_per_group, start : start + anchors_per_group] = False
        return {
            "anchor": dn_anchor.flatten(1, 2),
            "reg_target": dn_target.flatten(1, 2),
            "cls_target": dn_labels.flatten(1, 2),
            "valid_mask": dn_valid.flatten(1, 2),
            "attention_mask": attention_mask,
        }


class Sparse4DHead(nn.Module):
    """Sparse4Dv3 spatial head, optionally with v2-style recurrent fusion."""

    def __init__(
        self,
        num_classes: int,
        anchor: Union[str, Sequence[Sequence[float]]],
        pc: Sequence[float],
        num_anchor: int = 900,
        embed_dims: int = 256,
        num_decoder: int = 6,
        num_heads: int = 8,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_learnable_pts: int = 6,
        fix_scale: Optional[Sequence[Sequence[float]]] = None,
        dropout: float = 0.1,
        attn_drop: float = 0.15,
        decouple_attn: bool = True,
        with_quality_estimation: bool = True,
        default_time_interval: float = 0.5,
        cls_threshold_to_reg: float = -1.0,
        cls_loss_weight: float = 2.0,
        box_loss_weight: float = 0.25,
        matching_reg_weights: Optional[Sequence[float]] = None,
        reg_weights: Optional[Sequence[float]] = None,
        class_wise_reg_weights: Optional[Dict[Union[int, str], Sequence[float]]] = None,
        barrier_class_id: int = 0,
        num_dn_groups: int = 5,
        dn_noise_scale: Optional[Sequence[float]] = None,
        max_dn_gt: int = 32,
        add_neg_dn: bool = True,
        anchor_grad: bool = True,
        feat_grad: bool = False,
        use_deformable_func: bool = False,
        score_threshold: Optional[float] = None,
        temporal: bool = False,
        num_temp_instances: int = 600,
        num_single_frame_decoder: int = 1,
        confidence_decay: float = 0.6,
        max_time_interval: float = 2.0,
        min_time_interval: float = 0.01,
        tracking: bool = False,
        tracking_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        if matching_reg_weights is None:
            matching_reg_weights = [2.0] * 3 + [0.5] * 3 + [0.0] * 4
        if reg_weights is None:
            reg_weights = [2.0] * 3 + [1.0] * 7
        if dn_noise_scale is None:
            dn_noise_scale = [2.0] * 3 + [0.5] * 7
        if len(matching_reg_weights) != 10 or len(reg_weights) != 10:
            raise ValueError("Sparse4D matching_reg_weights and reg_weights must have length 10")

        self.num_classes = int(num_classes)
        self.num_anchor = int(num_anchor)
        self.embed_dims = int(embed_dims)
        self.num_decoder = int(num_decoder)
        self.pc = tuple(float(x) for x in pc)
        self.default_time_interval = float(default_time_interval)
        self.cls_threshold_to_reg = float(cls_threshold_to_reg)
        self.cls_loss_weight = float(cls_loss_weight)
        self.box_loss_weight = float(box_loss_weight)
        self.barrier_class_id = int(barrier_class_id)
        self.score_threshold = score_threshold
        self.temporal = bool(temporal)
        self.tracking = bool(tracking)
        self.tracking_threshold = float(tracking_threshold)
        if self.tracking and not self.temporal:
            raise ValueError("Tracking requires temporal Sparse4D")
        if not 0 <= self.tracking_threshold <= 1:
            raise ValueError("tracking_threshold must be in [0, 1]")
        self.num_single_frame_decoder = int(num_single_frame_decoder)
        if temporal and not (0 < num_temp_instances < num_anchor and 1 <= num_single_frame_decoder < num_decoder):
            raise ValueError("Temporal Sparse4D needs 0<K<N and at least one single-frame and temporal decoder")
        if temporal and not (0 <= confidence_decay <= 1 and max_time_interval > 0 and default_time_interval > 0):
            raise ValueError("Invalid temporal decay or time intervals")
        if temporal and not (0 < min_time_interval <= default_time_interval <= max_time_interval):
            raise ValueError("Temporal intervals must satisfy 0 < min <= default <= max")
        self.temporal_memory = TemporalMemory(num_temp_instances, confidence_decay,
            max_time_interval, default_time_interval, min_time_interval) if temporal else None
        self.register_buffer("reg_weights", torch.tensor(reg_weights, dtype=torch.float32))

        self.instance_bank = SingleFrameInstanceBank(
            anchor,
            num_anchor,
            embed_dims,
            anchor_grad=anchor_grad,
            feat_grad=feat_grad,
        )
        self.anchor_encoder = SparseBox3DEncoder(embed_dims)
        self.layers = nn.ModuleList(
            [
                Sparse4DDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_classes=num_classes,
                    num_groups=num_groups,
                    num_levels=num_levels,
                    num_cams=num_cams,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    num_learnable_pts=num_learnable_pts,
                    fix_scale=fix_scale,
                    decouple_attn=decouple_attn,
                    with_self_attention=decoder_idx > 0,
                    with_temporal_attention=temporal and decoder_idx >= num_single_frame_decoder,
                    with_quality_estimation=with_quality_estimation,
                    use_deformable_func=use_deformable_func,
                )
                for decoder_idx in range(num_decoder)
            ]
        )
        # Sparse4D applies Xavier initialization to all decoder operations
        # except the refinement module, then lets deformable aggregation reset
        # its attention weights to zero (uniform after softmax).
        for layer in self.layers:
            decoder_operations = [layer.deformable, layer.ffn]
            if layer.self_attention is not None:
                decoder_operations.append(layer.self_attention)
            if layer.temporal_attention is not None:
                decoder_operations.append(layer.temporal_attention)
            for operation in decoder_operations:
                for parameter in operation.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)
            nn.init.zeros_(layer.deformable.weights_fc.weight)
            nn.init.zeros_(layer.deformable.weights_fc.bias)
            nn.init.xavier_uniform_(layer.deformable.output_proj.weight)
            nn.init.zeros_(layer.deformable.output_proj.bias)
            if layer.deformable.kps_generator.learnable_fc is not None:
                nn.init.xavier_uniform_(
                    layer.deformable.kps_generator.learnable_fc.weight
                )
                nn.init.zeros_(layer.deformable.kps_generator.learnable_fc.bias)
        self.target = SparseBox3DTarget(
            num_classes=num_classes,
            cls_weight=cls_loss_weight,
            box_weight=box_loss_weight,
            matching_reg_weights=matching_reg_weights,
            class_wise_reg_weights=class_wise_reg_weights,
            num_dn_groups=num_dn_groups,
            dn_noise_scale=dn_noise_scale,
            max_dn_gt=max_dn_gt,
            add_neg_dn=add_neg_dn,
        )

    def reset_temporal(self, scope=None):
        if self.temporal_memory is not None:
            self.temporal_memory.reset(scope)

    def _apply(self, fn):
        self.reset_temporal()
        return super()._apply(fn)

    def _build_attention_mask(
        self,
        num_free: int,
        dn_attention_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        num_dn = dn_attention_mask.shape[0]
        attention_mask = torch.ones(
            num_free + num_dn,
            num_free + num_dn,
            dtype=torch.bool,
            device=device,
        )
        attention_mask[:num_free, :num_free] = False
        attention_mask[num_free:, num_free:] = dn_attention_mask
        return attention_mask

    def get_model_outs(
        self,
        feature_maps: Sequence[torch.Tensor],
        img_metas: Dict[str, torch.Tensor],
        include_dn: bool,
        return_motion_features: bool = False,
    ) -> Dict[str, Any]:
        # Match Sparse4D's FP32 head boundary while the image backbone/FPN
        # retain mixed precision. The cast preserves gradients to the FPN.
        with torch.autocast(device_type=feature_maps[0].device.type, enabled=False):
            if return_motion_features:
                return self._get_model_outs_fp32(
                    [feature.float() for feature in feature_maps], img_metas, include_dn,
                    return_motion_features=True,
                )
            return self._get_model_outs_fp32(
                [feature.float() for feature in feature_maps], img_metas, include_dn
            )

    def _get_model_outs_fp32(
        self,
        feature_maps: Sequence[torch.Tensor],
        img_metas: Dict[str, torch.Tensor],
        include_dn: bool,
        return_motion_features: bool = False,
    ) -> Dict[str, Any]:
        batch_size = feature_maps[0].shape[0]
        instance_feature, anchor = self.instance_bank.get(batch_size)
        num_free = anchor.shape[1]
        temporal_context = self.temporal_memory.prepare(img_metas, instance_feature, anchor) if self.temporal else None
        interval = temporal_context["interval"] if self.temporal else self.default_time_interval
        temporal_embed = self.anchor_encoder(temporal_context["anchor"]) if self.temporal else None
        dn_metas = None
        attention_mask = None
        if include_dn and "gt_bboxes" in img_metas:
            dn_metas = self.target.get_dn_anchors(
                img_metas["gt_bboxes"], anchor.shape[-1], synchronize=self.temporal)
        if dn_metas is not None:
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size,
                        dn_metas["anchor"].shape[1],
                        self.embed_dims,
                    ),
                ],
                dim=1,
            )
            anchor = torch.cat(
                [anchor, dn_metas["anchor"].to(device=anchor.device, dtype=anchor.dtype)],
                dim=1,
            )
            attention_mask = self._build_attention_mask(
                num_free, dn_metas["attention_mask"], anchor.device
            )

        predictions = []
        classifications = []
        qualities = []
        diagnostics: Dict[str, torch.Tensor] = {}
        anchor_embed = self.anchor_encoder(anchor)
        for decoder_idx, layer in enumerate(self.layers):
            instance_feature, anchor, classification, quality = layer(
                instance_feature,
                anchor,
                anchor_embed,
                feature_maps,
                img_metas,
                interval,
                attention_mask,
                temporal_context,
                temporal_embed,
            )
            if self.temporal:
                # This dataset supervises only vx/vy (10 box dimensions).
                # Do not recurrently propagate an unsupervised vertical speed.
                anchor = torch.cat((anchor[..., :VZ], torch.zeros_like(anchor[..., VZ:])), -1)
            predictions.append(anchor)
            classifications.append(classification)
            qualities.append(quality)
            if return_motion_features and decoder_idx + 1 == self.num_decoder:
                # Capture before a possible temporal-bank reorder; preserve
                # alignment with the final prediction and exclude DN queries.
                motion_features = instance_feature[:, :num_free]
            for name, value in layer.deformable.last_diagnostics.items():
                diagnostics[f"{name}_{decoder_idx}"] = value
            if self.temporal and decoder_idx + 1 == self.num_single_frame_decoder:
                instance_feature, anchor = self.temporal_memory.update(
                    instance_feature, anchor, classification, temporal_context, num_free)
            if decoder_idx + 1 < self.num_decoder:
                anchor_embed = self.anchor_encoder(anchor)
            if self.temporal and decoder_idx + 1 > self.num_single_frame_decoder:
                # History values stay detached; keys follow the refined propagated anchors.
                temporal_embed = anchor_embed[:, :self.temporal_memory.num_instances]

        instance_ids = None
        if self.tracking and not self.training:
            instance_ids = self.temporal_memory.assign_instance_ids(
                classifications[-1][:, :num_free], temporal_context,
                self.tracking_threshold, img_metas.get("temporal_read_only", False))
        if self.temporal:
            diagnostics["temporal_valid_ratio"] = temporal_context["valid"].float().mean().detach()
            diagnostics["temporal_time_interval"] = interval.mean().detach()
            if not img_metas.get("temporal_read_only", False):
                self.temporal_memory.cache(instance_feature[:, :num_free], anchor[:, :num_free],
                    classifications[-1][:, :num_free], temporal_context, instance_ids)
        output: Dict[str, Any] = {"instance_ids": instance_ids}
        if dn_metas is not None:
            output.update(
                {
                    "dn_prediction": [prediction[:, num_free:] for prediction in predictions],
                    "dn_classification": [classification[:, num_free:] for classification in classifications],
                    "dn_reg_target": dn_metas["reg_target"],
                    "dn_cls_target": dn_metas["cls_target"],
                    "dn_valid_mask": dn_metas["valid_mask"],
                }
            )
            predictions = [prediction[:, :num_free] for prediction in predictions]
            classifications = [classification[:, :num_free] for classification in classifications]
            qualities = [quality[:, :num_free] if quality is not None else None for quality in qualities]
        output.update(
            {
                "prediction": predictions,
                "classification": classifications,
                "quality": qualities,
                "diagnostics": diagnostics,
            }
        )
        if return_motion_features:
            output["instance_features"] = motion_features
        return output

    @staticmethod
    def _sigmoid_focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        avg_factor: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        one_hot = logits.new_zeros(logits.shape)
        positive = (targets >= 0) & (targets < num_classes)
        if positive.any():
            one_hot[positive, targets[positive]] = 1.0
        probability = logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
        p_t = probability * one_hot + (1.0 - probability) * (1.0 - one_hot)
        alpha_t = alpha * one_hot + (1.0 - alpha) * (1.0 - one_hot)
        return (ce_loss * alpha_t * (1.0 - p_t).pow(gamma)).sum() / avg_factor

    @staticmethod
    def _gaussian_focal_loss(
        probability: torch.Tensor,
        target: torch.Tensor,
        avg_factor: torch.Tensor,
        alpha: float = 2.0,
        gamma: float = 4.0,
    ) -> torch.Tensor:
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        positive = target.eq(1.0).to(probability.dtype)
        negative = (target < 1.0).to(probability.dtype)
        positive_loss = -probability.log() * (1.0 - probability).pow(alpha) * positive
        negative_loss = (
            -(1.0 - probability).log()
            * probability.pow(alpha)
            * (1.0 - target).pow(gamma)
            * negative
        )
        return (positive_loss + negative_loss).sum() / avg_factor

    def _box_and_quality_loss(
        self,
        box: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        cls_target: torch.Tensor,
        quality: Optional[torch.Tensor],
        avg_factor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        box = box[..., : len(self.reg_weights)]
        target = target[..., : len(self.reg_weights)].clone()
        weight = weight[..., : len(self.reg_weights)] * self.reg_weights

        reverse = (
            cls_target == self.barrier_class_id
        ) & (
            F.cosine_similarity(
                target[..., [SIN_YAW, COS_YAW]],
                box[..., [SIN_YAW, COS_YAW]],
                dim=-1,
            ) < 0
        )
        target[..., [SIN_YAW, COS_YAW]] = torch.where(
            reverse[..., None],
            -target[..., [SIN_YAW, COS_YAW]],
            target[..., [SIN_YAW, COS_YAW]],
        )
        finite = torch.isfinite(target) & torch.isfinite(box)
        safe_target = torch.where(finite, target, box.detach())
        safe_weight = weight * finite.to(weight.dtype)
        output = {
            "loss_box": (
                torch.abs(box - safe_target) * safe_weight
            ).sum() / avg_factor * self.box_loss_weight
        }
        if quality is not None:
            centerness_target = torch.exp(
                -torch.norm(
                    safe_target[..., [X, Y, Z]] - box[..., [X, Y, Z]],
                    p=2,
                    dim=-1,
                )
            )
            output["loss_cns"] = F.binary_cross_entropy_with_logits(
                quality[..., CENTERNESS], centerness_target, reduction="sum"
            ) / avg_factor
            yawness_target = (
                F.cosine_similarity(
                    safe_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                ) > 0
            ).to(box.dtype)
            output["loss_yns"] = self._gaussian_focal_loss(
                quality[..., YAWNESS].sigmoid(), yawness_target, avg_factor
            )
        return output

    def loss(
        self,
        model_outs: Dict[str, Any],
        gt_bboxes: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Union[
        Dict[str, torch.Tensor],
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        with torch.autocast(device_type=gt_bboxes.device.type, enabled=False):
            fp32_outs = dict(model_outs)
            for key in (
                "classification", "prediction", "quality",
                "dn_classification", "dn_prediction",
            ):
                if key in fp32_outs:
                    fp32_outs[key] = [
                        value.float() if value is not None else None
                        for value in fp32_outs[key]
                    ]
            if "dn_reg_target" in fp32_outs:
                fp32_outs["dn_reg_target"] = fp32_outs["dn_reg_target"].float()
            return self._loss_fp32(fp32_outs, gt_bboxes.float(), return_diagnostics)

    def _loss_fp32(
        self,
        model_outs: Dict[str, Any],
        gt_bboxes: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        output: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, torch.Tensor] = {}
        for decoder_idx, (classification, prediction, quality) in enumerate(
            zip(
                model_outs["classification"],
                model_outs["prediction"],
                model_outs["quality"],
            )
        ):
            cls_target, reg_target, reg_weight = self.target.sample(
                classification, prediction, gt_bboxes
            )
            positive_mask = reg_weight[..., :10].sum(dim=-1) > 0
            num_pos = _reduce_mean(positive_mask.sum().to(prediction.dtype)).clamp_min(1.0)
            diagnostics[f"positive_match_count_{decoder_idx}"] = num_pos.detach()
            positive_score = classification.sigmoid().amax(dim=-1)[positive_mask]
            diagnostics[f"positive_cls_score_{decoder_idx}"] = (
                positive_score.mean().detach()
                if positive_score.numel() > 0
                else classification.new_zeros(())
            )
            output[f"loss_cls_{decoder_idx}"] = self._sigmoid_focal_loss(
                classification.flatten(0, 1),
                cls_target.flatten(),
                self.num_classes,
                num_pos,
            ) * self.cls_loss_weight

            reg_mask = positive_mask
            if self.cls_threshold_to_reg > 0:
                reg_mask = reg_mask & (
                    classification.sigmoid().amax(dim=-1) > self.cls_threshold_to_reg
                )
            diagnostics[f"reg_active_ratio_{decoder_idx}"] = (
                reg_mask.sum().to(dtype=prediction.dtype)
                / positive_mask.sum().clamp_min(1).to(dtype=prediction.dtype)
            ).detach()
            if reg_mask.any():
                selected_quality = quality[reg_mask] if quality is not None else None
                reg_losses = self._box_and_quality_loss(
                    prediction[reg_mask],
                    reg_target[reg_mask],
                    reg_weight[reg_mask],
                    cls_target[reg_mask],
                    selected_quality,
                    num_pos,
                )
            else:
                zero = prediction.sum() * 0.0
                reg_losses = {"loss_box": zero}
                if quality is not None:
                    reg_losses.update({"loss_cns": zero, "loss_yns": zero})
            for name, value in reg_losses.items():
                output[f"{name}_{decoder_idx}"] = value

        if "dn_prediction" not in model_outs:
            if return_diagnostics:
                return output, diagnostics
            return output

        dn_valid = model_outs["dn_valid_mask"].flatten()
        dn_cls_target = model_outs["dn_cls_target"].flatten()[dn_valid]
        dn_reg_target = model_outs["dn_reg_target"].flatten(0, 1)[dn_valid]
        dn_positive = dn_cls_target < self.num_classes
        # The original Sparse4D implementation normalizes both DN
        # classification and regression by all valid DN items (positive and
        # explicitly generated negative anchors).
        num_dn = _reduce_mean(dn_valid.sum().to(gt_bboxes.dtype)).clamp_min(1.0)
        for decoder_idx, (classification, prediction) in enumerate(
            zip(model_outs["dn_classification"], model_outs["dn_prediction"])
        ):
            classification = classification.flatten(0, 1)[dn_valid]
            prediction = prediction.flatten(0, 1)[dn_valid]
            output[f"loss_cls_dn_{decoder_idx}"] = self._sigmoid_focal_loss(
                classification,
                dn_cls_target,
                self.num_classes,
                num_dn,
            ) * self.cls_loss_weight
            if dn_positive.any():
                positive_target = dn_reg_target[dn_positive]
                positive_prediction = prediction[dn_positive]
                output[f"loss_box_dn_{decoder_idx}"] = (
                    torch.abs(
                        positive_prediction[..., :10] - positive_target[..., :10]
                    )
                    * self.reg_weights
                ).sum() / num_dn * self.box_loss_weight
            else:
                output[f"loss_box_dn_{decoder_idx}"] = prediction.sum() * 0.0
        if return_diagnostics:
            return output, diagnostics
        return output

    def forward(
        self,
        feature_maps: Sequence[torch.Tensor],
        img_metas: Dict[str, torch.Tensor],
        split: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if "gt_bboxes" not in img_metas:
            raise KeyError("Sparse4D training/evaluation loss requires img_metas['gt_bboxes']")
        img_metas = dict(img_metas, temporal_scope=split)
        model_outs = self.get_model_outs(
            feature_maps,
            img_metas,
            include_dn=self.training,
        )
        losses, loss_diagnostics = self.loss(
            model_outs,
            img_metas["gt_bboxes"],
            return_diagnostics=True,
        )
        total_loss = sum(losses.values())
        log = {f"{split}/{name}": value.detach().item() for name, value in losses.items()}
        diagnostics = {**model_outs.get("diagnostics", {}), **loss_diagnostics}
        log.update(
            {
                f"{split}/{name}": value.detach().item()
                for name, value in diagnostics.items()
            }
        )
        log[f"{split}/loss"] = total_loss.detach().item()
        # Validation has no DN branch; expose comparable detection-only loss.
        log[f"{split}/loss_detection"] = sum(
            value.detach().item() for name, value in losses.items() if "_dn_" not in name)
        log[f"{split}/loss_denoising"] = sum(
            value.detach().item() for name, value in losses.items() if "_dn_" in name)
        return total_loss, log

    @torch.no_grad()
    def get_bboxes(
        self,
        feature_maps: Sequence[torch.Tensor],
        img_metas: Dict[str, torch.Tensor],
        score_thr: float = 0.0,
        max_num: int = 300,
    ) -> torch.Tensor:
        model_outs = self.get_model_outs(feature_maps, img_metas, include_dn=False)
        return self._decode_bboxes(model_outs, score_thr, max_num)

    @torch.no_grad()
    def get_tracks(self, feature_maps, img_metas, score_thr=0.0, max_num=300):
        if self.training or not self.tracking:
            raise ValueError("get_tracks requires eval mode and tracking=True")
        outs = self.get_model_outs(feature_maps, img_metas, include_dn=False)
        return self._decode_bboxes(outs, score_thr, max_num, tracking=True)

    def _decode_bboxes(self, model_outs, score_thr, max_num, tracking=False):
        classification = model_outs["classification"][-1].sigmoid()
        batch_size, num_query, num_classes = classification.shape
        if tracking:
            # One class per query: flattening classes would duplicate a track ID.
            confidence, all_labels = classification.max(-1)
            max_num = min(int(max_num), num_query)
            scores, query_indices = confidence.topk(max_num, dim=1)
            labels = all_labels.gather(1, query_indices)
        else:
            max_num = min(int(max_num), num_query * num_classes)
            scores, flat_indices = classification.flatten(1).topk(max_num, dim=1)
            labels = flat_indices.remainder(num_classes)
            query_indices = torch.div(flat_indices, num_classes, rounding_mode="floor")
        quality = model_outs["quality"][-1]
        if quality is not None:
            centerness = torch.gather(
                quality[..., CENTERNESS], 1, query_indices
            ).sigmoid()
            scores = scores * centerness
            scores, rerank = scores.sort(dim=-1, descending=True)
            labels = torch.gather(labels, 1, rerank)
            query_indices = torch.gather(query_indices, 1, rerank)
        prediction = model_outs["prediction"][-1][..., :10]
        boxes = torch.gather(
            prediction,
            1,
            query_indices[..., None].expand(-1, -1, prediction.shape[-1]),
        )

        threshold = self.score_threshold if self.score_threshold is not None else score_thr
        pc = boxes.new_tensor(self.pc)
        valid = (
            (boxes[..., :3] >= pc[:3]).all(dim=-1)
            & (boxes[..., :3] <= pc[3:]).all(dim=-1)
            & torch.isfinite(boxes).all(dim=-1)
            & torch.isfinite(scores)
            & (scores > float(threshold))
        )
        output = boxes.new_zeros(batch_size, max_num, 12)
        output[..., -2:] = -1.0
        output[..., :10][valid] = boxes[valid]
        output[..., 10][valid] = labels[valid].to(output.dtype)
        output[..., 11][valid] = scores[valid]
        if tracking:
            ids = model_outs["instance_ids"].gather(1, query_indices)
            valid &= ids >= 0
            output[..., 10:][~valid] = -1
            return {"bboxes": output, "instance_ids": ids.masked_fill(~valid, -1)}
        return output
