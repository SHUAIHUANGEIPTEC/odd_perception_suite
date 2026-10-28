import torch
from torch.autograd.function import Function, once_differentiable

from . import deformable_aggregation_ext


class DeformableAggregationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        feature_maps = feature_maps.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            feature_maps,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            feature_maps,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            feature_maps,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        grad_feature_maps = torch.zeros_like(feature_maps)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_ext.deformable_aggregation_backward(
            feature_maps,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous().float(),
            grad_feature_maps,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_feature_maps,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )
