#include <torch/extension.h>

// 声明在前向传播 CUDA 文件中定义的函数
at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step);

// 声明在反向传播 CUDA 文件中定义的函数
void ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, 
    const at::Tensor &grad_output,
    at::Tensor &grad_value, 
    at::Tensor &grad_sampling_loc,
    at::Tensor &grad_attn_weight, 
    const int im2col_step);

// 新增 3D 声明
at::Tensor ms_deform_attn_3d_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes, 
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc, 
    const at::Tensor &attn_weight, 
    const int im2col_step);

void ms_deform_attn_3d_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes, 
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc, 
    const at::Tensor &attn_weight, 
    const at::Tensor &grad_output,
    at::Tensor &grad_value, 
    at::Tensor &grad_sampling_loc, 
    at::Tensor &grad_attn_weight, 
    const int im2col_step);

    
// 使用 PyBind11 将 C++ 函数绑定为 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_cuda_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_cuda_backward, "ms_deform_attn_backward");
  m.def("ms_deform_attn_3d_forward", &ms_deform_attn_3d_cuda_forward, "ms_deform_attn_3d_forward");
  m.def("ms_deform_attn_3d_backward", &ms_deform_attn_3d_cuda_backward, "ms_deform_attn_3d_backward");
}
