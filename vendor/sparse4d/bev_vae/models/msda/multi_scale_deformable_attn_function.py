import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
# from mmcv.utils import ext_loader
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
import os
from torch.utils.cpp_extension import load

# 获取当前脚本所在目录，确保能找到 cpp 和 cu 文件
_src_path = os.path.dirname(os.path.abspath(__file__))

def _load_extension():
    return load(
        name="ms_deform_attn_ext",
        sources=[
            os.path.join(_src_path, "ms_deform_attn.cpp"),
            os.path.join(_src_path, "ms_deform_attn_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        # Let PyTorch translate TORCH_CUDA_ARCH_LIST for the installed toolkit.
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


class _LazyExtension:
    """Compile on first CUDA use, so CPU inspection/tests need no nvcc."""
    module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = _load_extension()
        return getattr(self.module, name)


ext_module = _LazyExtension()


def multi_scale_deformable_attn_reference(value, shapes, locations, weights):
    """Differentiable CPU reference for the 2D operator (align_corners=False)."""
    from torch.nn import functional as F
    b, _, heads, channels = value.shape
    queries, points = locations.shape[1], locations.shape[4]
    outputs = []
    offset = 0
    for level, (h, w) in enumerate(shapes.tolist()):
        feature = value[:, offset:offset+h*w].permute(0, 2, 3, 1)
        feature = feature.reshape(b*heads, channels, h, w)
        grid = locations[:, :, :, level].permute(0, 2, 1, 3, 4)
        grid = grid.reshape(b*heads, queries, points, 2)
        sampled = F.grid_sample(feature, grid*2-1, align_corners=False)
        sampled = sampled.reshape(b, heads, channels, queries, points).permute(0, 3, 1, 4, 2)
        outputs.append((sampled * weights[:, :, :, level, :, None]).sum(3))
        offset += h*w
    return torch.stack(outputs).sum(0).reshape(b, queries, heads*channels)


class MultiScaleDeformableAttnFunction_fp16(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttnFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None

class MultiScaleDeformableAttnFunction3D_fp16(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        # 注意这里调用的是 ext_module 中即将绑定的 3d_forward
        output = ext_module.ms_deform_attn_3d_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_3d_backward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output.contiguous(),
            grad_value, grad_sampling_loc, grad_attn_weight, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
    

class MultiScaleDeformableAttnFunction3D_fp32(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        # 注意这里调用的是 ext_module 中即将绑定的 3d_forward
        output = ext_module.ms_deform_attn_3d_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_3d_backward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output.contiguous(),
            grad_value, grad_sampling_loc, grad_attn_weight, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
