#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void deformable_aggregation(
    float* output,
    const float* feature_maps,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sampling_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups);

void deformable_aggregation_grad(
    const float* feature_maps,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sampling_location,
    const float* weights,
    const float* grad_output,
    float* grad_feature_maps,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups);

at::Tensor deformable_aggregation_forward(
    const at::Tensor& feature_maps,
    const at::Tensor& spatial_shape,
    const at::Tensor& scale_start_index,
    const at::Tensor& sampling_location,
    const at::Tensor& weights) {
  TORCH_CHECK(feature_maps.is_cuda(), "feature_maps must be a CUDA tensor");
  at::DeviceGuard guard(feature_maps.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(feature_maps));
  const int batch_size = feature_maps.size(0);
  const int num_feat = feature_maps.size(1);
  const int num_embeds = feature_maps.size(2);
  const int num_cams = spatial_shape.size(0);
  const int num_scale = spatial_shape.size(1);
  const int num_anchors = sampling_location.size(1);
  const int num_pts = sampling_location.size(2);
  const int num_groups = weights.size(5);
  auto output = at::zeros(
      {batch_size, num_anchors, num_embeds}, feature_maps.options());
  deformable_aggregation(
      output.data_ptr<float>(),
      feature_maps.data_ptr<float>(),
      spatial_shape.data_ptr<int>(),
      scale_start_index.data_ptr<int>(),
      sampling_location.data_ptr<float>(),
      weights.data_ptr<float>(),
      batch_size,
      num_cams,
      num_feat,
      num_embeds,
      num_scale,
      num_anchors,
      num_pts,
      num_groups);
  return output;
}

void deformable_aggregation_backward(
    const at::Tensor& feature_maps,
    const at::Tensor& spatial_shape,
    const at::Tensor& scale_start_index,
    const at::Tensor& sampling_location,
    const at::Tensor& weights,
    const at::Tensor& grad_output,
    at::Tensor& grad_feature_maps,
    at::Tensor& grad_sampling_location,
    at::Tensor& grad_weights) {
  TORCH_CHECK(feature_maps.is_cuda(), "feature_maps must be a CUDA tensor");
  at::DeviceGuard guard(feature_maps.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(feature_maps));
  deformable_aggregation_grad(
      feature_maps.data_ptr<float>(),
      spatial_shape.data_ptr<int>(),
      scale_start_index.data_ptr<int>(),
      sampling_location.data_ptr<float>(),
      weights.data_ptr<float>(),
      grad_output.data_ptr<float>(),
      grad_feature_maps.data_ptr<float>(),
      grad_sampling_location.data_ptr<float>(),
      grad_weights.data_ptr<float>(),
      feature_maps.size(0),
      spatial_shape.size(0),
      feature_maps.size(1),
      feature_maps.size(2),
      spatial_shape.size(1),
      sampling_location.size(1),
      sampling_location.size(2),
      weights.size(5));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "deformable_aggregation_forward",
      &deformable_aggregation_forward,
      "Sparse4D deformable aggregation forward");
  module.def(
      "deformable_aggregation_backward",
      &deformable_aggregation_backward,
      "Sparse4D deformable aggregation backward");
}
