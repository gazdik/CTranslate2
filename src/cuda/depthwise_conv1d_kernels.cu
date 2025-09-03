#include "ctranslate2/ops/depthwise_conv1d.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template<typename T>
    __global__ void depthwise_conv1d_kernel(const T* input,
                                             const T* weight,
                                             const T* bias,
                                             T* output,
                                             int batch_size,
                                             int input_length,
                                             int output_length,
                                             int channels,
                                             int kernel_size,
                                             int stride,
                                             int padding,
                                             int dilation) {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      const int total_elements = batch_size * output_length * channels;
      
      if (tid >= total_elements) return;
      
      const int c = tid % channels;
      const int o = (tid / channels) % output_length;
      const int b = tid / (output_length * channels);
      
      T sum = bias ? bias[c] : T(0);
      
      for (int k = 0; k < kernel_size; ++k) {
        const int input_pos = o * stride - padding + k * dilation;
        if (input_pos >= 0 && input_pos < input_length) {
          const int input_idx = b * input_length * channels + input_pos * channels + c;
          const int weight_idx = k * channels + c;
          sum += input[input_idx] * weight[weight_idx];
        }
      }
      
      output[tid] = sum;
    }

    template<>
    void DepthwiseConv1D::compute<Device::CUDA, float>(const StorageView& input,
                                                        const StorageView& weight,
                                                        const StorageView* bias,
                                                        StorageView& output,
                                                        const StorageView* qscale) const {
      const dim_t batch_size = input.dim(0);
      const dim_t input_length = input.dim(1);
      const dim_t output_length = output.dim(1);
      const dim_t channels = input.dim(2);
      const dim_t kernel_size = weight.dim(0);

      const float* input_data = input.data<float>();
      const float* weight_data = weight.data<float>();
      const float* bias_data = bias ? bias->data<float>() : nullptr;
      float* output_data = output.data<float>();

      const dim_t total_elements = batch_size * output_length * channels;
      const dim_t block_size = 256;
      const dim_t grid_size = (total_elements + block_size - 1) / block_size;

      depthwise_conv1d_kernel<<<grid_size, block_size, 0, cuda::get_cuda_stream()>>>(
        input_data, weight_data, bias_data, output_data,
        batch_size, input_length, output_length, channels, kernel_size,
        _stride, _padding, _dilation);
    }

    template<>
    void DepthwiseConv1D::compute<Device::CUDA, float16>(const StorageView& input,
                                                          const StorageView& weight,
                                                          const StorageView* bias,
                                                          StorageView& output,
                                                          const StorageView* qscale) const {
      const dim_t batch_size = input.dim(0);
      const dim_t input_length = input.dim(1);
      const dim_t output_length = output.dim(1);
      const dim_t channels = input.dim(2);
      const dim_t kernel_size = weight.dim(0);

      const float16* input_data = input.data<float16>();
      const float16* weight_data = weight.data<float16>();
      const float16* bias_data = bias ? bias->data<float16>() : nullptr;
      float16* output_data = output.data<float16>();

      const dim_t total_elements = batch_size * output_length * channels;
      const dim_t block_size = 256;
      const dim_t grid_size = (total_elements + block_size - 1) / block_size;

      depthwise_conv1d_kernel<<<grid_size, block_size, 0, cuda::get_cuda_stream()>>>(
        input_data, weight_data, bias_data, output_data,
        batch_size, input_length, output_length, channels, kernel_size,
        _stride, _padding, _dilation);
    }

  }
}
