#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float bilinear_sample(
    const float* data,
    int height,
    int width,
    int channels,
    float h,
    float w,
    int base) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1.0f - lh;
  const float hw = 1.0f - lw;
  const int h_stride = width * channels;
  const int w_stride = channels;
  float value = 0.0f;
  if (h_low >= 0 && w_low >= 0) {
    value += hh * hw * data[base + h_low * h_stride + w_low * w_stride];
  }
  if (h_low >= 0 && w_high < width) {
    value += hh * lw * data[base + h_low * h_stride + w_high * w_stride];
  }
  if (h_high < height && w_low >= 0) {
    value += lh * hw * data[base + h_high * h_stride + w_low * w_stride];
  }
  if (h_high < height && w_high < width) {
    value += lh * lw * data[base + h_high * h_stride + w_high * w_stride];
  }
  return value;
}

__device__ void bilinear_sample_grad(
    const float* data,
    float weight,
    int height,
    int width,
    int channels,
    float h,
    float w,
    int base,
    float grad_output,
    float* grad_data,
    float* grad_location,
    float* grad_weight) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1.0f - lh;
  const float hw = 1.0f - lw;
  const int h_stride = width * channels;
  const int w_stride = channels;
  const float grad_sample = grad_output * weight;
  float grad_h = 0.0f;
  float grad_w = 0.0f;
  float value = 0.0f;

  if (h_low >= 0 && w_low >= 0) {
    const int ptr = base + h_low * h_stride + w_low * w_stride;
    const float v = data[ptr];
    value += hh * hw * v;
    grad_h -= hw * v;
    grad_w -= hh * v;
    atomicAdd(grad_data + ptr, hh * hw * grad_sample);
  }
  if (h_low >= 0 && w_high < width) {
    const int ptr = base + h_low * h_stride + w_high * w_stride;
    const float v = data[ptr];
    value += hh * lw * v;
    grad_h -= lw * v;
    grad_w += hh * v;
    atomicAdd(grad_data + ptr, hh * lw * grad_sample);
  }
  if (h_high < height && w_low >= 0) {
    const int ptr = base + h_high * h_stride + w_low * w_stride;
    const float v = data[ptr];
    value += lh * hw * v;
    grad_h += hw * v;
    grad_w -= lh * v;
    atomicAdd(grad_data + ptr, lh * hw * grad_sample);
  }
  if (h_high < height && w_high < width) {
    const int ptr = base + h_high * h_stride + w_high * w_stride;
    const float v = data[ptr];
    value += lh * lw * v;
    grad_h += lw * v;
    grad_w += lh * v;
    atomicAdd(grad_data + ptr, lh * lw * grad_sample);
  }
  atomicAdd(grad_weight, grad_output * value);
  atomicAdd(grad_location, width * grad_w * grad_sample);
  atomicAdd(grad_location + 1, height * grad_h * grad_sample);
}

__global__ void deformable_aggregation_kernel(
    int kernels,
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
    int num_groups) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kernels) {
    return;
  }
  const float weight = weights[index / (num_embeds / num_groups)];
  const int channel = index % num_embeds;
  index /= num_embeds;
  const int scale = index % num_scale;
  index /= num_scale;
  const int camera = index % num_cams;
  index /= num_cams;
  const int point = index % num_pts;
  index /= num_pts;
  const int anchor_local = index % num_anchors;
  index /= num_anchors;
  const int batch = index % batch_size;
  const int anchor = batch * num_anchors + anchor_local;
  const int location_offset =
      ((anchor * num_pts + point) * num_cams + camera) * 2;
  const float location_w = sampling_location[location_offset];
  const float location_h = sampling_location[location_offset + 1];
  if (location_w <= 0.0f || location_w >= 1.0f ||
      location_h <= 0.0f || location_h >= 1.0f) {
    return;
  }
  const int camera_scale = camera * num_scale + scale;
  const int value_offset =
      (batch * num_feat + scale_start_index[camera_scale]) * num_embeds + channel;
  const int height = spatial_shape[camera_scale * 2];
  const int width = spatial_shape[camera_scale * 2 + 1];
  const float h = location_h * height - 0.5f;
  const float w = location_w * width - 0.5f;
  atomicAdd(
      output + anchor * num_embeds + channel,
      bilinear_sample(
          feature_maps, height, width, num_embeds, h, w, value_offset) * weight);
}

__global__ void deformable_aggregation_grad_kernel(
    int kernels,
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
    int num_groups) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kernels) {
    return;
  }
  const int weight_offset = index / (num_embeds / num_groups);
  const int channel = index % num_embeds;
  index /= num_embeds;
  const int scale = index % num_scale;
  index /= num_scale;
  const int camera = index % num_cams;
  index /= num_cams;
  const int point = index % num_pts;
  index /= num_pts;
  const int anchor_local = index % num_anchors;
  index /= num_anchors;
  const int batch = index % batch_size;
  const int anchor = batch * num_anchors + anchor_local;
  const int location_offset =
      ((anchor * num_pts + point) * num_cams + camera) * 2;
  const float location_w = sampling_location[location_offset];
  const float location_h = sampling_location[location_offset + 1];
  if (location_w <= 0.0f || location_w >= 1.0f ||
      location_h <= 0.0f || location_h >= 1.0f) {
    return;
  }
  const int camera_scale = camera * num_scale + scale;
  const int value_offset =
      (batch * num_feat + scale_start_index[camera_scale]) * num_embeds + channel;
  const int height = spatial_shape[camera_scale * 2];
  const int width = spatial_shape[camera_scale * 2 + 1];
  const float h = location_h * height - 0.5f;
  const float w = location_w * width - 0.5f;
  bilinear_sample_grad(
      feature_maps,
      weights[weight_offset],
      height,
      width,
      num_embeds,
      h,
      w,
      value_offset,
      grad_output[anchor * num_embeds + channel],
      grad_feature_maps,
      grad_sampling_location + location_offset,
      grad_weights + weight_offset);
}

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
    int num_groups) {
  const int kernels = batch_size * num_pts * num_embeds * num_anchors *
                      num_cams * num_scale;
  deformable_aggregation_kernel<<<(kernels + 127) / 128, 128>>>(
      kernels,
      output,
      feature_maps,
      spatial_shape,
      scale_start_index,
      sampling_location,
      weights,
      batch_size,
      num_cams,
      num_feat,
      num_embeds,
      num_scale,
      num_anchors,
      num_pts,
      num_groups);
}

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
    int num_groups) {
  const int kernels = batch_size * num_pts * num_embeds * num_anchors *
                      num_cams * num_scale;
  deformable_aggregation_grad_kernel<<<(kernels + 127) / 128, 128>>>(
      kernels,
      feature_maps,
      spatial_shape,
      scale_start_index,
      sampling_location,
      weights,
      grad_output,
      grad_feature_maps,
      grad_sampling_location,
      grad_weights,
      batch_size,
      num_cams,
      num_feat,
      num_embeds,
      num_scale,
      num_anchors,
      num_pts,
      num_groups);
}
