#include "ctranslate2/ops/depthwise_conv1d.h"

#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/activation.h"

namespace ctranslate2 {
  namespace ops {

    DepthwiseConv1D::DepthwiseConv1D(const dim_t stride,
                                     const dim_t padding,
                                     const dim_t dilation)
      : _stride(stride)
      , _padding(padding)
      , _dilation(dilation) {
    }

    void DepthwiseConv1D::operator()(const StorageView& input,
                                     const StorageView& weight,
                                     const StorageView& bias,
                                     StorageView& output,
                                     const StorageView* qscale) const {
      operator()(input, weight, &bias, output, qscale);
    }

    void DepthwiseConv1D::operator()(const StorageView& input,
                                     const StorageView& weight,
                                     StorageView& output,
                                     const StorageView* qscale) const {
      operator()(input, weight, nullptr, output, qscale);
    }

    void DepthwiseConv1D::operator()(const StorageView& input,
                                     const StorageView& weight,
                                     const StorageView* bias,
                                     StorageView& output,
                                     const StorageView* qscale) const {
      PROFILE("DepthwiseConv1D");

      const Device device = input.device();
      const DataType dtype = input.dtype();

      const dim_t batch_size = input.dim(0);
      const dim_t input_length = input.dim(1);
      const dim_t input_depth = input.dim(2);
      const dim_t kernel_size = weight.dim(0);
      const dim_t output_depth = weight.dim(1);

      if (input_depth != output_depth) {
        throw std::invalid_argument("DepthwiseConv1D: input depth must equal output depth");
      }

      const dim_t output_length = (input_length + 2 * _padding - _dilation * (kernel_size - 1) - 1) / _stride + 1;

      Shape output_shape = {static_cast<dim_t>(batch_size), static_cast<dim_t>(output_length), static_cast<dim_t>(output_depth)};
      output = StorageView(output_shape, dtype, device);

      if (device == Device::CUDA) {
#ifdef CT2_WITH_CUDA
        compute<Device::CUDA, float>(input, weight, bias, output, qscale);
#else
        throw std::runtime_error("CUDA support is not available");
#endif
      } else {
        compute<Device::CPU, float>(input, weight, bias, output, qscale);
      }
    }

    template <Device D, typename T>
    void DepthwiseConv1D::compute(const StorageView& input,
                                  const StorageView& weight,
                                  const StorageView* bias,
                                  StorageView& output,
                                  const StorageView* /*qscale*/) const {
      const dim_t batch_size = input.dim(0);
      const dim_t input_length = input.dim(1);
      const dim_t input_depth = input.dim(2);
      const dim_t kernel_size = weight.dim(0);
      const dim_t output_length = output.dim(1);

      const T* input_data = input.data<T>();
      const T* weight_data = weight.data<T>();
      const T* bias_data = bias ? bias->data<T>() : nullptr;
      T* output_data = output.data<T>();

      if (D == Device::CPU) {
        // CPU implementation
        #pragma omp parallel for collapse(3)
        for (dim_t b = 0; b < batch_size; ++b) {
          for (dim_t o = 0; o < output_length; ++o) {
            for (dim_t c = 0; c < input_depth; ++c) {
              float sum = bias_data ? float(bias_data[c]) : 0.0f;
              
              for (dim_t k = 0; k < kernel_size; ++k) {
                const dim_t input_pos = o * _stride - _padding + k * _dilation;
                if (input_pos >= 0 && input_pos < input_length) {
                  const dim_t input_idx = b * input_length * input_depth + input_pos * input_depth + c;
                  const dim_t weight_idx = k * input_depth + c;
                  sum += float(input_data[input_idx]) * float(weight_data[weight_idx]);
                }
              }
              
              const dim_t output_idx = b * output_length * input_depth + o * input_depth + c;
              output_data[output_idx] = sum;
            }
          }
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    DepthwiseConv1D::compute<Device::CPU, T>(const StorageView& input, \
                                              const StorageView& weight, \
                                              const StorageView* bias,   \
                                              StorageView& output,       \
                                              const StorageView* qscale) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
