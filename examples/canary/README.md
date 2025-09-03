# Canary Model Integration

This directory contains examples and documentation for using NVIDIA's Canary models with CTranslate2.

## Overview

Canary is a multilingual encoder-decoder model for automatic speech recognition (ASR) and automatic speech translation (AST). It uses a FastConformer encoder and Transformer decoder architecture.

### Key Features

- **Multilingual**: Supports English, Spanish, German, French (and more in v2)
- **Multi-task**: Both transcription (ASR) and translation (AST)
- **Fast**: Uses FastConformer with 8x downsampling and local attention
- **Efficient**: 2.8x faster than standard Conformer while maintaining accuracy

## Model Conversion

Convert a NeMo Canary model to CTranslate2 format:

```bash
python -m ctranslate2.converters.nemo_canary \
    --model_path /path/to/canary.nemo \
    --output_dir /path/to/ct2_model \
    --quantization float16
```

## Usage Examples

### Basic Transcription

```python
import ctranslate2
import numpy as np

# Load model
model = ctranslate2.models.Canary("/path/to/ct2_model")

# Prepare audio features (log-mel spectrogram)
# Shape: (batch_size, time_steps, n_mels=80)
features = np.random.randn(1, 1000, 80).astype(np.float32)

# Transcribe
options = ctranslate2.models.CanaryOptions()
options.task = "asr"
options.source_lang = "en"
options.target_lang = "en"
options.pnc = True

results = model.generate(features, options=options)
print("Transcription:", results[0].sequences[0])
```

### Translation

```python
# Translate from English to Spanish
options = ctranslate2.models.CanaryOptions()
options.task = "ast"
options.source_lang = "en"
options.target_lang = "es"
options.pnc = True

results = model.generate(features, options=options)
print("Translation:", results[0].sequences[0])
```

### Language Detection

```python
# Detect the language of the audio
lang_probs = model.detect_language(features)
print("Detected languages:", lang_probs[0])  # [(lang, prob), ...]
```

### With Timestamps

```python
# Enable word-level timestamps
options = ctranslate2.models.CanaryOptions()
options.task = "asr"
options.source_lang = "en" 
options.target_lang = "en"
options.timestamps = True

results = model.generate(features, options=options)
print("Transcription with timestamps:", results[0].sequences[0])
```

## Model Architecture

### FastConformer Encoder

- **8x Downsampling**: Reduces sequence length early for efficiency
- **Depthwise Separable Convolutions**: Reduces computational cost
- **Local Attention**: Handles long sequences with windowed attention
- **Global Tokens**: Optional global attention for better context

### Transformer Decoder

- Standard transformer decoder with cross-attention to encoder
- Special control tokens for task, language, and formatting options

## Control Tokens

Canary uses special tokens to control generation:

- `<asr>`, `<ast>`: Task selection (transcription vs translation)
- `<en>`, `<es>`, `<de>`, `<fr>`: Language codes
- `<pnc_on>`, `<pnc_off>`: Punctuation and capitalization
- `<timestamps_on>`, `<timestamps_off>`: Timestamp generation

## Performance

- **Speed**: >1000 RTFx on modern GPUs
- **Accuracy**: WER 5.77% on MCV 16.1 test sets
- **Memory**: 4x memory savings vs standard Conformer
- **Compute**: 2.8x faster inference

## Requirements

- CUDA-capable GPU (recommended)
- Audio preprocessing to log-mel spectrograms (80 features)
- Sample rate: 16kHz
- Window: 25ms with 10ms stride

## Limitations

- Requires BPE tokenization for 8x downsampling with CTC
- Limited to supported languages (expandable)
- Optimal for 16kHz audio (resampling needed for other rates)
