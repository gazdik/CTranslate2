#pragma once

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/fastconformer.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    struct CanaryOptions {
      // Beam size to use for beam search (set 1 to run greedy search).
      size_t beam_size = 5;

      // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
      // The decoding will continue until beam_size*patience hypotheses are finished.
      float patience = 1;

      // Exponential penalty applied to the length during beam search.
      float length_penalty = 1;

      // Penalty applied to the score of previously generated tokens, as described in
      // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
      float repetition_penalty = 1;

      // Prevent repetitions of ngrams with this size (set 0 to disable).
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 512;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // High temperatures increase randomness.
      float sampling_temperature = 1;

      // Number of hypotheses to include in the result.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Include log probs of each token in the result
      bool return_logits_vocab = false;

      // Task type: "asr" for transcription, "ast" for translation
      std::string task = "asr";

      // Source language code (e.g., "en", "es", "de", "fr")
      std::string source_lang = "en";

      // Target language code (e.g., "en", "es", "de", "fr")
      std::string target_lang = "en";

      // Enable punctuation and capitalization
      bool pnc = true;

      // Enable timestamps
      bool timestamps = false;
    };

    struct CanaryGenerationResult {
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;
      std::vector<std::vector<StorageView>> logits;

      size_t num_sequences() const {
        return sequences.size();
      }

      bool has_scores() const {
        return !scores.empty();
      }
    };

    class CanaryModel : public Model {
    public:
      virtual ~CanaryModel() = default;
      
      const Vocabulary& get_vocabulary() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

      // Get supported languages
      const std::vector<std::string>& get_supported_languages() const {
        return _supported_languages;
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
      std::vector<std::string> _supported_languages;
    };

    class CanaryReplica : public ModelReplica {
    public:
      static std::unique_ptr<CanaryReplica> create_from_model(const Model& model);

      CanaryReplica(const std::shared_ptr<const CanaryModel>& model);

      size_t n_mels() const {
        return _n_mels;
      }

      const std::vector<std::string>& get_supported_languages() const {
        return _model->get_supported_languages();
      }

      StorageView encode(StorageView features, const bool to_cpu);

      std::vector<CanaryGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<std::string>>& prompts,
               const CanaryOptions& options);

      std::vector<CanaryGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<size_t>>& prompts,
               const CanaryOptions& options);

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(StorageView features);

    private:
      const std::shared_ptr<const CanaryModel> _model;
      const std::unique_ptr<layers::FastConformerEncoder> _encoder;
      const std::unique_ptr<layers::Decoder> _decoder;

      // Special token IDs
      size_t _sot_id;
      size_t _eot_id;
      size_t _task_asr_id;
      size_t _task_ast_id;
      size_t _pnc_on_id;
      size_t _pnc_off_id;
      size_t _timestamps_on_id;
      size_t _timestamps_off_id;
      
      // Language token IDs
      std::unordered_map<std::string, size_t> _lang_ids;
      
      size_t _n_mels;

      StorageView maybe_encode(StorageView features);
      
      std::vector<size_t> build_prompt(const CanaryOptions& options) const;
    };

    class Canary : public ReplicaPool<CanaryReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      size_t n_mels() const;
      const std::vector<std::string>& get_supported_languages() const;

      std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

      std::vector<std::future<CanaryGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<std::string>> prompts,
               CanaryOptions options = {});

      std::vector<std::future<CanaryGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<size_t>> prompts,
               CanaryOptions options = {});

      std::vector<std::future<std::vector<std::pair<std::string, float>>>>
      detect_language(const StorageView& features);
    };

  }
}
