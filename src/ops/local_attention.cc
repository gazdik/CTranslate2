#include "ctranslate2/ops/local_attention.h"

#include <cmath>

namespace ctranslate2 {
  namespace ops {

    LocalAttention::LocalAttention(const dim_t num_heads,
                                   const dim_t window_size,
                                   const bool use_global_token,
                                   const float scale)
      : _num_heads(num_heads)
      , _window_size(window_size)
      , _use_global_token(use_global_token)
      , _scale(scale) {
    }

    void LocalAttention::operator()(const StorageView& queries,
                                    const StorageView& keys,
                                    const StorageView& values,
                                    const StorageView* lengths,
                                    StorageView& output,
                                    StorageView* cached_keys,
                                    StorageView* cached_values,
                                    StorageView* attention_weights) const {
      PROFILE("LocalAttention");

      const Device device = queries.device();
      // const DataType dtype = queries.dtype(); // Unused

      if (device == Device::CUDA) {
#ifdef CT2_WITH_CUDA
        compute<Device::CUDA, float>(queries, keys, values, lengths, output,
                                     cached_keys, cached_values, attention_weights);
#else
        throw std::runtime_error("CUDA support is not available");
#endif
      } else {
        compute<Device::CPU, float>(queries, keys, values, lengths, output,
                                    cached_keys, cached_values, attention_weights);
      }
    }

    template <Device D, typename T>
    void LocalAttention::compute(const StorageView& queries,
                                 const StorageView& keys,
                                 const StorageView& values,
                                 const StorageView* /*lengths*/,
                                 StorageView& output,
                                 StorageView* /*cached_keys*/,
                                 StorageView* /*cached_values*/,
                                 StorageView* /*attention_weights*/) const {
      const dim_t batch_size = queries.dim(0);
      const dim_t seq_len = queries.dim(1);
      const dim_t head_dim = queries.dim(2) / _num_heads;

      const float scale_f = _scale > 0 ? _scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
      const T scale = static_cast<T>(scale_f);

      output = StorageView(queries.shape(), queries.dtype(), queries.device());

      if (D == Device::CPU) {
        // CPU implementation of local attention
        const T* q_data = queries.data<T>();
        const T* k_data = keys.data<T>();
        const T* v_data = values.data<T>();
        T* out_data = output.data<T>();

        #pragma omp parallel for collapse(2)
        for (dim_t b = 0; b < batch_size; ++b) {
          for (dim_t h = 0; h < _num_heads; ++h) {
            for (dim_t i = 0; i < seq_len; ++i) {
              // Determine attention window
              dim_t start_pos = 0;
              dim_t end_pos = seq_len;
              
              if (_window_size > 0 && !_use_global_token) {
                start_pos = std::max(static_cast<dim_t>(0), i - _window_size / 2);
                end_pos = std::min(seq_len, i + _window_size / 2 + 1);
              }

              // Compute attention scores
              std::vector<float> scores(end_pos - start_pos);
              float max_score = std::numeric_limits<float>::lowest();
              
              for (dim_t j = start_pos; j < end_pos; ++j) {
                float score = 0.0f;
                for (dim_t d = 0; d < head_dim; ++d) {
                  const dim_t q_idx = b * seq_len * _num_heads * head_dim + 
                                      i * _num_heads * head_dim + h * head_dim + d;
                  const dim_t k_idx = b * seq_len * _num_heads * head_dim + 
                                      j * _num_heads * head_dim + h * head_dim + d;
                  score += static_cast<float>(q_data[q_idx]) * static_cast<float>(k_data[k_idx]);
                }
                score *= scale_f;
                scores[j - start_pos] = score;
                max_score = std::max(max_score, score);
              }

              // Apply softmax
              float sum_exp = 0.0f;
              for (auto& score : scores) {
                score = std::exp(score - max_score);
                sum_exp += score;
              }
              for (auto& score : scores) {
                score /= sum_exp;
              }

              // Compute weighted sum of values
              for (dim_t d = 0; d < head_dim; ++d) {
                float result = 0.0f;
                for (dim_t j = start_pos; j < end_pos; ++j) {
                  const dim_t v_idx = b * seq_len * _num_heads * head_dim + 
                                      j * _num_heads * head_dim + h * head_dim + d;
                  result += scores[j - start_pos] * static_cast<float>(v_data[v_idx]);
                }
                const dim_t out_idx = b * seq_len * _num_heads * head_dim + 
                                      i * _num_heads * head_dim + h * head_dim + d;
                out_data[out_idx] = static_cast<T>(result);
              }
            }
          }
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LocalAttention::compute<Device::CPU, T>(const StorageView& queries, \
                                             const StorageView& keys,    \
                                             const StorageView& values,  \
                                             const StorageView* lengths, \
                                             StorageView& output,        \
                                             StorageView* cached_keys,   \
                                             StorageView* cached_values, \
                                             StorageView* attention_weights) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
