#pragma once

#include "ctranslate2/layers/attention.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/ops/depthwise_conv1d.h"
#include "ctranslate2/ops/local_attention.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    class DepthwiseSeparableConv1D : public Layer {
    public:
      DepthwiseSeparableConv1D(const models::Model& model,
                               const std::string& scope,
                               dim_t stride = 1,
                               dim_t padding = 0,
                               dim_t dilation = 1);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _pointwise_conv.output_type();
      }

      dim_t output_size() const override {
        return _pointwise_conv.output_size();
      }

    private:
      const ops::DepthwiseConv1D _depthwise_conv;
      const Dense _pointwise_conv;
    };

    class FastConformerSubsampling : public Layer {
    public:
      FastConformerSubsampling(const models::Model& model,
                               const std::string& scope,
                               dim_t subsampling_factor = 8);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _conv3.output_type();
      }

      dim_t output_size() const override {
        return _conv3.output_size();
      }

    private:
      const std::unique_ptr<DepthwiseSeparableConv1D> _conv1;
      const std::unique_ptr<DepthwiseSeparableConv1D> _conv2;
      const Dense _conv3;
      const ops::ActivationType _activation_type;
      dim_t _subsampling_factor;
    };

    class FastConformerEncoderLayer : public Layer {
    public:
      FastConformerEncoderLayer(const models::Model& model,
                                const std::string& scope,
                                dim_t num_heads,
                                dim_t window_size = -1,
                                bool use_global_token = false,
                                bool pre_norm = true,
                                ops::ActivationType activation_type = ops::ActivationType::Swish);

      void operator()(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output,
                      const Padder* padder = nullptr,
                      StorageView* position_bias = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

    private:
      const std::unique_ptr<LayerNorm> _self_attn_layer_norm;
      const std::unique_ptr<LayerNorm> _conv_layer_norm;
      const std::unique_ptr<LayerNorm> _ff_layer_norm;
      
      // Multi-head attention (local or global)
      const std::unique_ptr<AttentionLayer> _self_attention;
      
      // Convolution module
      const std::unique_ptr<LayerNorm> _conv_norm;
      const std::unique_ptr<Dense> _conv_pointwise1;
      const std::unique_ptr<ops::Conv1D> _conv_depthwise;
      const std::unique_ptr<Dense> _conv_pointwise2;
      
      // Feed forward network
      const FeedForwardNetwork _ff;
      
      const bool _pre_norm;
      const ops::ActivationType _activation_type;
    };

    class FastConformerEncoder : public Encoder {
    public:
      FastConformerEncoder(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& features,
                      const StorageView& lengths,
                      StorageView& output) override;

      DataType output_type() const override {
        return _layers.back()->output_type();
      }

      dim_t output_size() const override {
        return _layers.back()->output_size();
      }

    private:
      const std::unique_ptr<FastConformerSubsampling> _subsampling;
      const std::unique_ptr<Dense> _linear;
      const std::unique_ptr<LayerNorm> _layer_norm;
      std::vector<std::unique_ptr<FastConformerEncoderLayer>> _layers;
      const bool _use_relative_position;
    };

  }
}
