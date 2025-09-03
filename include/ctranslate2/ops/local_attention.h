#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class LocalAttention : public Op {
    public:
      LocalAttention(dim_t num_heads, 
                     dim_t window_size, 
                     bool use_global_token = false,
                     float scale = 0);

      void operator()(const StorageView& queries,
                      const StorageView& keys,
                      const StorageView& values,
                      const StorageView* lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention_weights = nullptr) const;

    private:
      dim_t _num_heads;
      dim_t _window_size;
      bool _use_global_token;
      float _scale;

      template <Device D, typename T>
      void compute(const StorageView& queries,
                   const StorageView& keys,
                   const StorageView& values,
                   const StorageView* lengths,
                   StorageView& output,
                   StorageView* cached_keys,
                   StorageView* cached_values,
                   StorageView* attention_weights) const;
    };

  }
}
