#include <gtest/gtest.h>

#include "ctranslate2/models/canary.h"
#include "ctranslate2/ops/depthwise_conv1d.h"
#include "ctranslate2/ops/local_attention.h"

namespace ctranslate2 {
  namespace test {

    class CanaryTest : public ::testing::Test {
    protected:
      void SetUp() override {
        // Setup test data
      }
    };

    TEST_F(CanaryTest, DepthwiseConv1DBasic) {
      const Device device = Device::CPU;
      const DataType dtype = DataType::FLOAT32;
      
      // Create test input: batch=1, length=10, channels=4
      StorageView input({1, 10, 4}, dtype, device);
      input.fill(1.0f);
      
      // Create test weight: kernel=3, channels=4
      StorageView weight({3, 4}, dtype, device);
      weight.fill(0.5f);
      
      // Create test bias
      StorageView bias({4}, dtype, device);
      bias.fill(0.1f);
      
      StorageView output(dtype, device);
      
      ops::DepthwiseConv1D conv(/*stride=*/1, /*padding=*/1, /*dilation=*/1);
      conv(input, weight, bias, output);
      
      // Check output shape: (1, 10, 4)
      EXPECT_EQ(output.dim(0), 1);
      EXPECT_EQ(output.dim(1), 10);
      EXPECT_EQ(output.dim(2), 4);
      
      // Check output values (simple sanity check)
      const float* output_data = output.data<float>();
      EXPECT_GT(output_data[0], 0.0f);
    }

    TEST_F(CanaryTest, LocalAttentionBasic) {
      const Device device = Device::CPU;
      const DataType dtype = DataType::FLOAT32;
      
      const dim_t batch_size = 1;
      const dim_t seq_len = 8;
      const dim_t num_heads = 2;
      const dim_t head_dim = 4;
      const dim_t model_dim = num_heads * head_dim;
      
      // Create test queries, keys, values
      StorageView queries({batch_size, seq_len, model_dim}, dtype, device);
      StorageView keys({batch_size, seq_len, model_dim}, dtype, device);
      StorageView values({batch_size, seq_len, model_dim}, dtype, device);
      
      queries.fill(1.0f);
      keys.fill(0.5f);
      values.fill(0.8f);
      
      StorageView output(dtype, device);
      
      ops::LocalAttention local_attn(num_heads, /*window_size=*/4, /*use_global_token=*/false);
      local_attn(queries, keys, values, nullptr, output);
      
      // Check output shape
      EXPECT_EQ(output.dim(0), batch_size);
      EXPECT_EQ(output.dim(1), seq_len);
      EXPECT_EQ(output.dim(2), model_dim);
      
      // Check output values (sanity check)
      const float* output_data = output.data<float>();
      EXPECT_GT(output_data[0], 0.0f);
      EXPECT_LT(output_data[0], 1.0f);
    }

    TEST_F(CanaryTest, CanaryOptionsDefaults) {
      models::CanaryOptions options;
      
      EXPECT_EQ(options.beam_size, 5);
      EXPECT_EQ(options.task, "asr");
      EXPECT_EQ(options.source_lang, "en");
      EXPECT_EQ(options.target_lang, "en");
      EXPECT_TRUE(options.pnc);
      EXPECT_FALSE(options.timestamps);
    }

  }
}
