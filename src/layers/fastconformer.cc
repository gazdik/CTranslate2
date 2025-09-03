#include "ctranslate2/layers/fastconformer.h"

#include "ctranslate2/ops/activation.h"
#include "ctranslate2/ops/add.h"
#include "ctranslate2/ops/layer_norm.h"

namespace ctranslate2 {
  namespace layers {

    DepthwiseSeparableConv1D::DepthwiseSeparableConv1D(const models::Model& model,
                                                        const std::string& scope,
                                                        const dim_t stride,
                                                        const dim_t padding,
                                                        const dim_t dilation)
      : _depthwise_conv(stride, padding, dilation)
      , _pointwise_conv(model, scope + "/pointwise", nullptr, true) {
    }

    void DepthwiseSeparableConv1D::operator()(const StorageView& input, StorageView& output) const {
      const Device device = input.device();
      const DataType dtype = input.dtype();
      
      StorageView depthwise_output(dtype, device);
      
      // Apply depthwise convolution (assuming depthwise weights are stored separately)
      // This is a simplified implementation - in practice, you'd load the depthwise weights
      _depthwise_conv(input, input, depthwise_output); // Placeholder - needs proper weight loading
      
      // Apply pointwise convolution (1x1 conv)
      _pointwise_conv(depthwise_output, output);
    }

    FastConformerSubsampling::FastConformerSubsampling(const models::Model& model,
                                                       const std::string& scope,
                                                       const dim_t subsampling_factor)
      : _conv1(build_optional_layer<DepthwiseSeparableConv1D>(model, scope + "/conv1", 2, 1, 1))
      , _conv2(build_optional_layer<DepthwiseSeparableConv1D>(model, scope + "/conv2", 2, 1, 1))
      , _conv3(model, scope + "/conv3")
      , _activation_type(ops::ActivationType::ReLU)
      , _subsampling_factor(subsampling_factor) {
    }

    void FastConformerSubsampling::operator()(const StorageView& input, StorageView& output) const {
      const Device device = input.device();
      const DataType dtype = input.dtype();

      StorageView x = input;
      
      // First depthwise separable conv (2x downsampling)
      if (_conv1) {
        StorageView conv1_out(dtype, device);
        (*_conv1)(x, conv1_out);
        ops::ReLU()(conv1_out, conv1_out);
        x = std::move(conv1_out);
      }

      // Second depthwise separable conv (2x downsampling) 
      if (_conv2) {
        StorageView conv2_out(dtype, device);
        (*_conv2)(x, conv2_out);
        ops::ReLU()(conv2_out, conv2_out);
        x = std::move(conv2_out);
      }

      // Final linear projection
      _conv3(x, output);
    }

    FastConformerEncoderLayer::FastConformerEncoderLayer(const models::Model& model,
                                                         const std::string& scope,
                                                         const dim_t num_heads,
                                                         const dim_t window_size,
                                                         const bool use_global_token,
                                                         const bool pre_norm,
                                                         const ops::ActivationType activation_type)
      : _self_attn_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/self_attn_layer_norm"))
      , _conv_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/conv_layer_norm"))
      , _ff_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/ff_layer_norm"))
      , _self_attention(std::make_unique<MultiHeadAttention>(model, scope + "/self_attention", num_heads, true, pre_norm))
      , _conv_norm(build_optional_layer<LayerNorm>(model, scope + "/conv_norm"))
      , _conv_pointwise1(build_optional_layer<Dense>(model, scope + "/conv_pointwise1"))
      , _conv_depthwise(std::make_unique<ops::DepthwiseConv1D>(1, 1, 1))
      , _conv_pointwise2(build_optional_layer<Dense>(model, scope + "/conv_pointwise2"))
      , _ff(model, scope + "/ffn", pre_norm, activation_type)
      , _pre_norm(pre_norm)
      , _activation_type(activation_type) {
    }

    void FastConformerEncoderLayer::operator()(const StorageView& input,
                                               const StorageView* lengths,
                                               StorageView& output,
                                               const Padder* padder,
                                               StorageView* position_bias) const {
      PROFILE("FastConformerEncoderLayer");
      
      const Device device = input.device();
      const DataType dtype = input.dtype();
      
      StorageView residual = input;
      StorageView x = input;

      // 1. Multi-head attention module
      if (_self_attn_layer_norm && _pre_norm) {
        (*_self_attn_layer_norm)(x, x);
      }
      
      StorageView attn_output(dtype, device);
      (*_self_attention)(x, x, lengths, attn_output, nullptr, nullptr, nullptr, padder, padder, true, position_bias);
      
      ops::Add()(residual, attn_output, x);
      
      if (_self_attn_layer_norm && !_pre_norm) {
        (*_self_attn_layer_norm)(x, x);
      }

      // 2. Convolution module
      residual = x;
      if (_conv_layer_norm && _pre_norm) {
        (*_conv_layer_norm)(x, x);
      }

      StorageView conv_output(dtype, device);
      if (_conv_pointwise1 && _conv_depthwise && _conv_pointwise2) {
        // Pointwise expansion
        StorageView pw1_out(dtype, device);
        (*_conv_pointwise1)(x, pw1_out);
        ops::GELU()(pw1_out, pw1_out); // GELU activation

        // Depthwise convolution
        StorageView dw_out(dtype, device);
        // For depthwise conv, we need weights - this is a placeholder
        // In practice, you'd load the depthwise weights from the model
        // Using pw1_out as both weight and input for now
        (*_conv_depthwise)(pw1_out, pw1_out, dw_out);
        
        if (_conv_norm) {
          (*_conv_norm)(dw_out, dw_out);
        }
        ops::Swish()(dw_out, dw_out);

        // Pointwise projection
        (*_conv_pointwise2)(dw_out, conv_output);
      } else {
        conv_output = x;
      }

      ops::Add()(residual, conv_output, x);
      
      if (_conv_layer_norm && !_pre_norm) {
        (*_conv_layer_norm)(x, x);
      }

      // 3. Feed forward module
      _ff(x, output);
    }

    FastConformerEncoder::FastConformerEncoder(const models::Model& model, const std::string& scope)
      : _subsampling(build_optional_layer<FastConformerSubsampling>(model, scope + "/subsampling"))
      , _linear(build_optional_layer<Dense>(model, scope + "/linear"))
      , _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _use_relative_position(model.config.value("use_relative_position", true)) {
      
      const dim_t num_layers = model.config["encoder"]["num_layers"];
      const dim_t num_heads = model.config["encoder"]["num_heads"];
      const dim_t window_size = model.config.value("encoder/window_size", -1);
      const bool use_global_token = model.config.value("encoder/use_global_token", false);
      
      _layers.reserve(num_layers);
      for (dim_t i = 0; i < num_layers; ++i) {
        const std::string layer_scope = scope + "/layers/" + std::to_string(i);
        _layers.emplace_back(std::make_unique<FastConformerEncoderLayer>(
          model, layer_scope, num_heads, window_size, use_global_token));
      }
    }

    void FastConformerEncoder::operator()(const StorageView& features,
                                          const StorageView& lengths,
                                          StorageView& output) {
      PROFILE("FastConformerEncoder");
      
      const Device device = features.device();
      const DataType dtype = features.dtype();
      
      StorageView x = features;

      // Apply subsampling
      if (_subsampling) {
        StorageView subsampled(dtype, device);
        (*_subsampling)(x, subsampled);
        x = std::move(subsampled);
      }

      // Apply linear projection
      if (_linear) {
        StorageView projected(dtype, device);
        (*_linear)(x, projected);
        x = std::move(projected);
      }

      // Apply layer norm
      if (_layer_norm) {
        (*_layer_norm)(x, x);
      }

      // Apply encoder layers
      StorageView layer_output(dtype, device);
      for (const auto& layer : _layers) {
        (*layer)(x, &lengths, layer_output);
        x = std::move(layer_output);
        layer_output = StorageView(dtype, device);
      }

      output = std::move(x);
    }

  }
}
