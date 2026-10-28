#ifndef DEFORM_ATTN_CUDA_KERNEL
#define DEFORM_ATTN_CUDA_KERNEL

// #include "common_cuda_helper.hpp"
// #include "pytorch_cuda_helper.hpp"

// 替换为这 3 个核心宏定义
#define THREADS_PER_BLOCK 512

#define GET_BLOCKS(N, T) (((N) + (T) - 1) / (T))

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                spatial_w, num_heads, channels,
                                                h_im, w_im, m_col, c_col) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  unsigned int tid = threadIdx.x;
  const int qid_stride = num_heads * channels;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int _tid = 1; _tid < blockSize; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[_tid];
            sid += 2;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *grad_attn_weight_out = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int _tid = 1; _tid < blockDim.x; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[_tid];
            sid += 2;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *grad_attn_weight_out = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          atomicAdd(grad_sampling_loc_out, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc_out + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight_out, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear_gm(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              grad_sampling_loc_out, grad_attn_weight_out);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}


// =========================================================================================
// ========================== 3D Deformable Attention Kernels ==============================
// =========================================================================================

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_trilinear(
    const scalar_t *&bottom_data, const int &depth, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &d, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c) {
  
  const int d_low = floorf(d); const int h_low = floorf(h); const int w_low = floorf(w);
  const int d_high = d_low + 1; const int h_high = h_low + 1; const int w_high = w_low + 1;

  const scalar_t ld = d - d_low; const scalar_t lh = h - h_low; const scalar_t lw = w - w_low;
  const scalar_t hd = 1 - ld, hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int d_stride = height * h_stride;
  const int base_ptr = m * channels + c;

  scalar_t val = 0;
  for (int i = 0; i < 2; ++i) {
    int cur_d = (i == 0) ? d_low : d_high;
    scalar_t wd = (i == 0) ? hd : ld;
    if (cur_d < 0 || cur_d >= depth) continue;
    
    for (int j = 0; j < 2; ++j) {
      int cur_h = (j == 0) ? h_low : h_high;
      scalar_t wh = (j == 0) ? hh : lh;
      if (cur_h < 0 || cur_h >= height) continue;
      
      for (int k = 0; k < 2; ++k) {
        int cur_w = (k == 0) ? w_low : w_high;
        scalar_t ww = (k == 0) ? hw : lw;
        if (cur_w < 0 || cur_w >= width) continue;

        int ptr = cur_d * d_stride + cur_h * h_stride + cur_w * w_stride + base_ptr;
        val += wd * wh * ww * bottom_data[ptr];
      }
    }
  }
  return val;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_trilinear_shm(
    const scalar_t *&bottom_data, const int &depth, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &d, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  
  const int d_low = floorf(d); const int h_low = floorf(h); const int w_low = floorf(w);
  const int d_high = d_low + 1; const int h_high = h_low + 1; const int w_high = w_low + 1;

  const scalar_t ld = d - d_low; const scalar_t lh = h - h_low; const scalar_t lw = w - w_low;
  const scalar_t hd = 1 - ld, hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int d_stride = height * h_stride;
  const int base_ptr = m * channels + c;

  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_d_weight = 0, grad_h_weight = 0, grad_w_weight = 0, total_val = 0;

  for (int i = 0; i < 2; ++i) {
    int cur_d = (i == 0) ? d_low : d_high;
    scalar_t wd = (i == 0) ? hd : ld;
    if (cur_d < 0 || cur_d >= depth) continue;
    for (int j = 0; j < 2; ++j) {
      int cur_h = (j == 0) ? h_low : h_high;
      scalar_t wh = (j == 0) ? hh : lh;
      if (cur_h < 0 || cur_h >= height) continue;
      for (int k = 0; k < 2; ++k) {
        int cur_w = (k == 0) ? w_low : w_high;
        scalar_t ww = (k == 0) ? hw : lw;
        if (cur_w < 0 || cur_w >= width) continue;

        int ptr = cur_d * d_stride + cur_h * h_stride + cur_w * w_stride + base_ptr;
        scalar_t v = bottom_data[ptr];
        
        grad_d_weight += (i == 0 ? -1 : 1) * wh * ww * v;
        grad_h_weight += wd * (j == 0 ? -1 : 1) * ww * v;
        grad_w_weight += wd * wh * (k == 0 ? -1 : 1) * v;
        total_val += wd * wh * ww * v;
        
        // 特征梯度的写入由于分散在三维空间，依然保留 atomicAdd
        atomicAdd(grad_value + ptr, wd * wh * ww * top_grad_value);
      }
    }
  }
  // 核心优化：直接赋值给属于该线程的局部共享内存指针！
  *grad_attn_weight = top_grad * total_val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
  *(grad_sampling_loc + 2) = depth * grad_d_weight * top_grad_value;
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_3d_gpu_kernel(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels; _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads; _temp /= num_heads; _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3; 
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_ptr = l_col * 3; 
      const int spatial_d = data_spatial_shapes[spatial_ptr];
      const int spatial_h = data_spatial_shapes[spatial_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_ptr + 2];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (d_im > -1 && w_im > -1 && h_im > -1 && d_im < spatial_d && h_im < spatial_h && w_im < spatial_w) {
          col += ms_deform_attn_im2col_trilinear(data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im, m_col, c_col) * weight;
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_3d_gpu_kernel_shm_reduce(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  
  // 动态分配的共享内存
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  // 3D 偏移量有 xyz 三个，因此偏移量内存占用是 3 倍
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 3 * blockDim.x;
  
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels; _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads; _temp /= num_heads; _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr * 3;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out = grad_sampling_loc + (grad_sampling_ptr * 3);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_ptr = l_col * 3;
      const int spatial_d = data_spatial_shapes[spatial_ptr];
      const int spatial_h = data_spatial_shapes[spatial_ptr + 1];
      const int spatial_w = data_spatial_shapes[spatial_ptr + 2];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      scalar_t *grad_value_ptr = grad_value + (data_value_ptr_init_offset + level_start_id * qid_stride);

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t loc_d = data_sampling_loc[data_loc_w_ptr + 2];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t d_im = loc_d * spatial_d - 0.5;
        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        // 初始化局部共享内存
        *(cache_grad_sampling_loc + (tid * 3)) = 0;
        *(cache_grad_sampling_loc + (tid * 3) + 1) = 0;
        *(cache_grad_sampling_loc + (tid * 3) + 2) = 0;
        *(cache_grad_attn_weight + tid) = 0;

        if (d_im > -1 && w_im > -1 && h_im > -1 && d_im < spatial_d && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_trilinear_shm(
              data_value_ptr, spatial_d, spatial_h, spatial_w, num_heads, channels, d_im, h_im, w_im,
              m_col, c_col, top_grad, weight, grad_value_ptr, 
              cache_grad_sampling_loc + (tid * 3), 
              cache_grad_attn_weight + tid);
        }
        // 等待同一 Block 里的 32 个通道全部写完局部内存
        __syncthreads();

        // 让 0 号线程收尾，完成局部累加并写入一次全局内存！
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0];
          scalar_t _grad_h = cache_grad_sampling_loc[1];
          scalar_t _grad_d = cache_grad_sampling_loc[2];
          scalar_t _grad_a = cache_grad_attn_weight[0];
          
          int sid = 3;
          for (unsigned int _tid = 1; _tid < blockDim.x; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_d += cache_grad_sampling_loc[sid + 2];
            _grad_a += cache_grad_attn_weight[_tid];
            sid += 3;
          }

          // 直接使用等号赋值，彻底干掉冗余的 atomicAdd！
          // 因为每个 Block 独占这部分显存，直接原位累加即可保证绝对安全。
          grad_sampling_loc_out[0] = _grad_w;
          grad_sampling_loc_out[1] = _grad_h;
          grad_sampling_loc_out[2] = _grad_d;
          *grad_attn_weight_out = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 3;
        grad_attn_weight_out += 1;
        grad_sampling_loc_out += 3;
      }
    }
  }
}


#endif  // DEFORM_ATTN_CUDA_KERNEL
