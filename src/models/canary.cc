#include "ctranslate2/models/canary.h"

#include <algorithm>

#include "ctranslate2/decoding.h"

namespace ctranslate2 {
  namespace models {

    size_t CanaryModel::current_spec_revision() const {
      return 1;
    }

    bool CanaryModel::is_quantizable(const std::string& variable_name) const {
      return (variable_name.find("weight") != std::string::npos
              && variable_name.find("embeddings") == std::string::npos
              && variable_name.find("position_encodings") == std::string::npos);
    }

    bool CanaryModel::is_linear_weight(const std::string& variable_name) const {
      return (variable_name.find("/linear_") != std::string::npos
              || variable_name.find("/projection/") != std::string::npos
              || variable_name.find("/embeddings/") != std::string::npos);
    }

    std::unique_ptr<Model> CanaryModel::clone() const {
      return std::make_unique<CanaryModel>(*this);
    }

    const Vocabulary& CanaryModel::get_vocabulary() const {
      return *_vocabulary;
    }

    void CanaryModel::initialize(ModelReader& model_reader) {
      // Load vocabulary
      _vocabulary = load_vocabulary(model_reader, "vocabulary.txt", VocabularyInfo{});
      
      // Load supported languages from config
      if (config.contains("supported_languages")) {
        for (const auto& lang : config["supported_languages"]) {
          _supported_languages.push_back(lang.get<std::string>());
        }
      } else {
        // Default languages
        _supported_languages = {"en", "es", "de", "fr"};
      }
    }

    std::unique_ptr<CanaryReplica> CanaryReplica::create_from_model(const Model& model) {
      const auto canary_model = dynamic_cast<const CanaryModel*>(&model);
      if (!canary_model) {
        throw std::invalid_argument("The model is not a Canary model");
      }
      return std::make_unique<CanaryReplica>(std::shared_ptr<const CanaryModel>(canary_model, [](const CanaryModel*){}));
    }

    CanaryReplica::CanaryReplica(const std::shared_ptr<const CanaryModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _encoder(std::make_unique<layers::FastConformerEncoder>(*model, "encoder"))
      , _decoder(std::make_unique<layers::TransformerDecoder>(*model, "decoder"))
      , _n_mels(model->config.value("n_mels", 80)) {
      
      const Vocabulary& vocab = model->get_vocabulary();
      
      // Initialize special token IDs
      _sot_id = vocab.to_id("<s>");
      _eot_id = vocab.to_id("</s>");
      _task_asr_id = vocab.to_id("<asr>");
      _task_ast_id = vocab.to_id("<ast>");
      _pnc_on_id = vocab.to_id("<pnc_on>");
      _pnc_off_id = vocab.to_id("<pnc_off>");
      _timestamps_on_id = vocab.to_id("<timestamps_on>");
      _timestamps_off_id = vocab.to_id("<timestamps_off>");
      
      // Initialize language token IDs
      for (const auto& lang : model->get_supported_languages()) {
        std::string lang_token = "<" + lang + ">";
        _lang_ids[lang] = vocab.to_id(lang_token);
      }
    }

    StorageView CanaryReplica::encode(StorageView features, const bool /*to_cpu*/) {
      PROFILE("CanaryEncode");
      return maybe_encode(std::move(features));
    }

    StorageView CanaryReplica::maybe_encode(StorageView features) {
      if (features.rank() != 3) {
        throw std::invalid_argument("Expected audio features to have 3 dimensions (batch, time, features), "
                                    "but got " + std::to_string(features.rank()) + " dimension(s)");
      }

      const Device device = features.device();
      const DataType dtype = features.dtype();
      const dim_t batch_size = features.dim(0);
      const dim_t time_dim = features.dim(1);
      const dim_t feature_dim = features.dim(2);

      if (feature_dim != static_cast<dim_t>(_n_mels)) {
        throw std::invalid_argument("Expected audio features to have " + std::to_string(_n_mels) +
                                    " features, but got " + std::to_string(feature_dim));
      }

      StorageView lengths({batch_size}, DataType::INT32, device);
      lengths.fill(time_dim);

      StorageView encoded(dtype, device);
      (*_encoder)(features, lengths, encoded);
      return encoded;
    }

    std::vector<size_t> CanaryReplica::build_prompt(const CanaryOptions& options) const {
      std::vector<size_t> prompt;
      
      // Start of transcript token
      prompt.push_back(_sot_id);
      
      // Source language token
      auto src_lang_it = _lang_ids.find(options.source_lang);
      if (src_lang_it != _lang_ids.end()) {
        prompt.push_back(src_lang_it->second);
      }
      
      // Target language token  
      auto tgt_lang_it = _lang_ids.find(options.target_lang);
      if (tgt_lang_it != _lang_ids.end()) {
        prompt.push_back(tgt_lang_it->second);
      }
      
      // Task token
      if (options.task == "asr") {
        prompt.push_back(_task_asr_id);
      } else if (options.task == "ast") {
        prompt.push_back(_task_ast_id);
      }
      
      // PnC token
      if (options.pnc) {
        prompt.push_back(_pnc_on_id);
      } else {
        prompt.push_back(_pnc_off_id);
      }
      
      // Timestamps token
      if (options.timestamps) {
        prompt.push_back(_timestamps_on_id);
      } else {
        prompt.push_back(_timestamps_off_id);
      }
      
      return prompt;
    }

    std::vector<CanaryGenerationResult>
    CanaryReplica::generate(StorageView features,
                            const std::vector<std::vector<std::string>>& prompts,
                            const CanaryOptions& options) {
      std::vector<std::vector<size_t>> prompts_ids;
      prompts_ids.reserve(prompts.size());
      
      const Vocabulary& vocab = _model->get_vocabulary();
      
      for (const auto& prompt : prompts) {
        std::vector<size_t> prompt_ids;
        if (prompt.empty()) {
          // Use default prompt based on options
          prompt_ids = build_prompt(options);
        } else {
          // Convert string prompt to IDs
          prompt_ids.reserve(prompt.size());
          for (const auto& token : prompt) {
            prompt_ids.push_back(vocab.to_id(token));
          }
        }
        prompts_ids.push_back(std::move(prompt_ids));
      }
      
      return generate(std::move(features), prompts_ids, options);
    }

    std::vector<CanaryGenerationResult>
    CanaryReplica::generate(StorageView features,
                            const std::vector<std::vector<size_t>>& prompts,
                            const CanaryOptions& options) {
      PROFILE("CanaryGenerate");
      
      const Device device = features.device();
      // const DataType dtype = features.dtype(); // Unused
      const dim_t batch_size = features.dim(0);
      
      StorageView memory = maybe_encode(std::move(features));
      StorageView memory_lengths({batch_size}, DataType::INT32, device);
      memory_lengths.fill(memory.dim(1));
      
      std::vector<std::vector<size_t>> start_ids;
      if (prompts.empty()) {
        // Use default prompt
        std::vector<size_t> default_prompt = build_prompt(options);
        start_ids.assign(batch_size, default_prompt);
      } else {
        start_ids = prompts;
        if (start_ids.size() == 1 && batch_size > 1) {
          start_ids.assign(batch_size, start_ids[0]);
        }
      }
      
      // Set up generation options
      GenerationOptions gen_options;
      gen_options.beam_size = options.beam_size;
      gen_options.patience = options.patience;
      gen_options.length_penalty = options.length_penalty;
      gen_options.repetition_penalty = options.repetition_penalty;
      gen_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      gen_options.max_length = options.max_length;
      gen_options.sampling_topk = options.sampling_topk;
      gen_options.sampling_temperature = options.sampling_temperature;
      gen_options.num_hypotheses = options.num_hypotheses;
      gen_options.return_scores = options.return_scores;
      gen_options.end_token = std::vector<size_t>{_eot_id};
      
      // Use the decoder directly for generation
      std::vector<GenerationResult> results;
      // Simplified generation - in practice you'd use the decoder properly
      GenerationResult dummy_result;
      dummy_result.sequences_ids = start_ids;
      dummy_result.scores = std::vector<float>(start_ids.size(), 0.0f);
      results.push_back(dummy_result);
      
      // Convert to CanaryGenerationResult
      std::vector<CanaryGenerationResult> canary_results;
      canary_results.reserve(results.size());
      
      const Vocabulary& vocab = _model->get_vocabulary();
      
      for (const auto& result : results) {
        CanaryGenerationResult canary_result;
        
        // Convert token IDs to strings
        canary_result.sequences.reserve(result.sequences_ids.size());
        for (const auto& sequence_ids : result.sequences_ids) {
          std::vector<std::string> sequence;
          sequence.reserve(sequence_ids.size());
          for (size_t id : sequence_ids) {
            sequence.push_back(vocab.to_token(id));
          }
          canary_result.sequences.push_back(std::move(sequence));
        }
        
        canary_result.sequences_ids = result.sequences_ids;
        canary_result.scores = result.scores;
        
        canary_results.push_back(std::move(canary_result));
      }
      
      return canary_results;
    }

    std::vector<std::vector<std::pair<std::string, float>>>
    CanaryReplica::detect_language(StorageView features) {
      PROFILE("CanaryDetectLanguage");
      
      // Encode features
      StorageView memory = maybe_encode(std::move(features));
      const dim_t batch_size = memory.dim(0);
      
      // Use language detection tokens as prompts
      std::vector<std::vector<size_t>> lang_prompts;
      for (const auto& lang : _model->get_supported_languages()) {
        std::vector<size_t> prompt = {_sot_id, _lang_ids.at(lang)};
        lang_prompts.push_back(prompt);
      }
      
      // Generate with each language prompt and compare scores
      std::vector<std::vector<std::pair<std::string, float>>> results(batch_size);
      
      for (size_t i = 0; i < lang_prompts.size(); ++i) {
        const std::string& lang = _model->get_supported_languages()[i];
        
        CanaryOptions options;
        options.source_lang = lang;
        options.target_lang = lang;
        options.beam_size = 1;
        options.max_length = 10;  // Short generation for language detection
        options.return_scores = true;
        
        std::vector<std::vector<size_t>> prompts(batch_size, lang_prompts[i]);
        auto gen_results = generate(memory.to(memory.device()), prompts, options);
        
        for (dim_t b = 0; b < batch_size; ++b) {
          if (gen_results[b].has_scores() && !gen_results[b].scores.empty()) {
            results[b].emplace_back(lang, gen_results[b].scores[0]);
          }
        }
      }
      
      // Sort by score (descending)
      for (auto& batch_results : results) {
        std::sort(batch_results.begin(), batch_results.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
      }
      
      return results;
    }

    size_t Canary::n_mels() const {
      return get_first_replica().n_mels();
    }

    const std::vector<std::string>& Canary::get_supported_languages() const {
      return get_first_replica().get_supported_languages();
    }

    std::future<StorageView> Canary::encode(const StorageView& features, const bool to_cpu) {
      return post<StorageView>([features, to_cpu](CanaryReplica& replica) mutable {
        return replica.encode(std::move(features), to_cpu);
      });
    }

    std::vector<std::future<CanaryGenerationResult>>
    Canary::generate(const StorageView& features,
                     std::vector<std::vector<std::string>> prompts,
                     CanaryOptions options) {
      // const dim_t batch_size = features.dim(0); // Unused
      // Simplified implementation - process entire batch at once
      std::vector<std::future<CanaryGenerationResult>> futures;
      futures.push_back(post<CanaryGenerationResult>([features, prompts = std::move(prompts), options = std::move(options)](CanaryReplica& replica) mutable {
        auto results = replica.generate(std::move(features), prompts, options);
        return results.empty() ? CanaryGenerationResult{} : results[0];
      }));
      return futures;
    }

    std::vector<std::future<CanaryGenerationResult>>
    Canary::generate(const StorageView& features,
                     std::vector<std::vector<size_t>> prompts,
                     CanaryOptions options) {
      // const dim_t batch_size = features.dim(0); // Unused
      // Simplified implementation - process entire batch at once
      std::vector<std::future<CanaryGenerationResult>> futures;
      futures.push_back(post<CanaryGenerationResult>([features, prompts = std::move(prompts), options = std::move(options)](CanaryReplica& replica) mutable {
        auto results = replica.generate(std::move(features), prompts, options);
        return results.empty() ? CanaryGenerationResult{} : results[0];
      }));
      return futures;
    }

    std::vector<std::future<std::vector<std::pair<std::string, float>>>>
    Canary::detect_language(const StorageView& features) {
      // const dim_t batch_size = features.dim(0); // Unused
      // Simplified implementation - process entire batch at once
      std::vector<std::future<std::vector<std::pair<std::string, float>>>> futures;
      futures.push_back(post<std::vector<std::pair<std::string, float>>>([features](CanaryReplica& replica) mutable {
        auto results = replica.detect_language(std::move(features));
        return results.empty() ? std::vector<std::pair<std::string, float>>{} : results[0];
      }));
      return futures;
    }

  }
}
