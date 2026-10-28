"""Optional Sparse4D deformable aggregation CUDA backend.

Importing this package never requires a compiled extension.  Callers can use
``CUDA_DEFORMABLE_AGGREGATION_AVAILABLE`` to select the CUDA kernel or the
PyTorch reference implementation.
"""

from typing import Sequence

import torch

try:
    from .function import DeformableAggregationFunction

    CUDA_DEFORMABLE_AGGREGATION_AVAILABLE = True
except (ImportError, OSError):
    DeformableAggregationFunction = None
    CUDA_DEFORMABLE_AGGREGATION_AVAILABLE = False


def deformable_aggregation(
    feature_maps: torch.Tensor,
    spatial_shape: torch.Tensor,
    scale_start_index: torch.Tensor,
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if DeformableAggregationFunction is None:
        raise RuntimeError(
            "The Sparse4D deformable aggregation CUDA extension has not been built"
        )
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def format_feature_maps(feature_maps: Sequence[torch.Tensor]):
    """Format [B,V,C,H,W] levels for the original Sparse4D CUDA kernel."""
    batch_size, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []
    flattened = []
    for feature_map in feature_maps:
        if feature_map.shape[:3] != (
            batch_size,
            num_cams,
            feature_maps[0].shape[2],
        ):
            raise ValueError("All Sparse4D feature levels must share B, V and C")
        spatial_shape.append(feature_map.shape[-2:])
        flattened.append(
            feature_map.reshape(
                batch_size, num_cams, feature_map.shape[2], -1
            )
        )
    flattened = torch.cat(flattened, dim=-1)
    flattened = flattened.permute(0, 1, 3, 2).flatten(1, 2).contiguous()
    spatial_shape = torch.tensor(
        [spatial_shape] * num_cams,
        dtype=torch.int32,
        device=flattened.device,
    )
    scale_size = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_size.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [scale_start_index.new_zeros(1), scale_start_index[:-1]], dim=0
    ).reshape(num_cams, -1)
    return flattened, spatial_shape, scale_start_index


__all__ = [
    "CUDA_DEFORMABLE_AGGREGATION_AVAILABLE",
    "deformable_aggregation",
    "format_feature_maps",
]
