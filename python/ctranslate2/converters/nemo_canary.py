"""Converter for NeMo Canary models."""

import argparse
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, model_spec


class CanaryConverter(Converter):
    """Converts NeMo Canary models to CTranslate2 format."""

    def __init__(self, model_path: str, vocab_path: Optional[str] = None):
        """Initializes the Canary converter.
        
        Args:
            model_path: Path to the NeMo .nemo file or extracted directory
            vocab_path: Optional path to vocabulary file
        """
        self._model_path = model_path
        self._vocab_path = vocab_path
        self._model_config = None
        self._model_weights = None
        
    def _load(self):
        """Load the NeMo Canary model."""
        import torch
        from omegaconf import OmegaConf
        
        if self._model_path.endswith('.nemo'):
            # Extract .nemo file
            with tempfile.TemporaryDirectory() as temp_dir:
                shutil.unpack_archive(self._model_path, temp_dir)
                config_path = os.path.join(temp_dir, 'model_config.yaml')
                weights_path = os.path.join(temp_dir, 'model_weights.ckpt')
                
                self._model_config = OmegaConf.load(config_path)
                self._model_weights = torch.load(weights_path, map_location='cpu')
        else:
            # Load from directory
            config_path = os.path.join(self._model_path, 'model_config.yaml')
            weights_path = os.path.join(self._model_path, 'model_weights.ckpt')
            
            self._model_config = OmegaConf.load(config_path)
            self._model_weights = torch.load(weights_path, map_location='cpu')

    def _get_model_spec(self, activation_type: common_spec.Activation) -> model_spec.LanguageModelSpec:
        """Build model specification."""
        config = self._model_config
        
        # Extract model dimensions
        encoder_config = config.encoder
        decoder_config = config.decoder
        
        spec = model_spec.LanguageModelSpec.from_config(
            (encoder_config.d_model, decoder_config.d_model),
            activation=activation_type,
        )
        
        # Set Canary-specific configuration
        spec.config = {
            "model_type": "canary",
            "encoder_type": "fastconformer",
            "decoder_type": "transformer",
            "n_mels": config.preprocessor.features,
            "sample_rate": config.sample_rate,
            "subsampling_factor": encoder_config.subsampling_factor,
            "num_encoder_layers": encoder_config.n_layers,
            "num_decoder_layers": decoder_config.n_layers,
            "encoder_attention_heads": encoder_config.n_heads,
            "decoder_attention_heads": decoder_config.n_heads,
            "encoder_ffn_dim": encoder_config.d_model * encoder_config.ff_expansion_factor,
            "decoder_ffn_dim": decoder_config.d_model * decoder_config.ff_expansion_factor,
            "conv_kernel_size": encoder_config.conv_kernel_size,
            "window_size": getattr(encoder_config, 'att_context_size', [-1, -1]),
            "use_global_token": getattr(encoder_config, 'use_global_token', False),
            "supported_languages": config.get('supported_languages', ['en', 'es', 'de', 'fr']),
        }
        
        return spec

    def _get_vocabulary(self) -> List[str]:
        """Extract vocabulary from the model."""
        if self._vocab_path:
            with open(self._vocab_path, 'r', encoding='utf-8') as f:
                vocab = [line.strip() for line in f]
        else:
            # Extract from tokenizer in model config
            tokenizer_config = self._model_config.tokenizer
            if hasattr(tokenizer_config, 'vocab_file'):
                with open(tokenizer_config.vocab_file, 'r', encoding='utf-8') as f:
                    vocab = [line.strip() for line in f]
            else:
                # Default vocabulary for demonstration
                vocab = ['<pad>', '<s>', '</s>', '<unk>'] + [f'<{lang}>' for lang in ['en', 'es', 'de', 'fr']]
                vocab += ['<asr>', '<ast>', '<pnc_on>', '<pnc_off>', '<timestamps_on>', '<timestamps_off>']
                
        return vocab

    def _convert_encoder_layer(self, layer_weights: Dict[str, Any], layer_idx: int) -> Dict[str, np.ndarray]:
        """Convert a FastConformer encoder layer."""
        converted = {}
        prefix = f"encoder/layers/{layer_idx}"
        
        # Self-attention
        if "self_attn" in layer_weights:
            attn_weights = layer_weights["self_attn"]
            
            # Query, Key, Value projections
            if "linear_q" in attn_weights:
                converted[f"{prefix}/self_attention/linear_0/weight"] = attn_weights["linear_q"]["weight"].numpy().T
                if "bias" in attn_weights["linear_q"]:
                    converted[f"{prefix}/self_attention/linear_0/bias"] = attn_weights["linear_q"]["bias"].numpy()
            
            if "linear_k" in attn_weights:
                converted[f"{prefix}/self_attention/linear_1/weight"] = attn_weights["linear_k"]["weight"].numpy().T
                if "bias" in attn_weights["linear_k"]:
                    converted[f"{prefix}/self_attention/linear_1/bias"] = attn_weights["linear_k"]["bias"].numpy()
                    
            if "linear_v" in attn_weights:
                converted[f"{prefix}/self_attention/linear_2/weight"] = attn_weights["linear_v"]["weight"].numpy().T
                if "bias" in attn_weights["linear_v"]:
                    converted[f"{prefix}/self_attention/linear_2/bias"] = attn_weights["linear_v"]["bias"].numpy()
            
            # Output projection
            if "linear_out" in attn_weights:
                converted[f"{prefix}/self_attention/linear_3/weight"] = attn_weights["linear_out"]["weight"].numpy().T
                if "bias" in attn_weights["linear_out"]:
                    converted[f"{prefix}/self_attention/linear_3/bias"] = attn_weights["linear_out"]["bias"].numpy()

        # Layer norms
        if "self_attn_layer_norm" in layer_weights:
            ln_weights = layer_weights["self_attn_layer_norm"]
            converted[f"{prefix}/self_attn_layer_norm/weight"] = ln_weights["weight"].numpy()
            converted[f"{prefix}/self_attn_layer_norm/bias"] = ln_weights["bias"].numpy()

        # Convolution module
        if "conv_module" in layer_weights:
            conv_weights = layer_weights["conv_module"]
            
            # Pointwise convolutions
            if "pointwise_conv1" in conv_weights:
                converted[f"{prefix}/conv_pointwise1/weight"] = conv_weights["pointwise_conv1"]["weight"].numpy().T
                if "bias" in conv_weights["pointwise_conv1"]:
                    converted[f"{prefix}/conv_pointwise1/bias"] = conv_weights["pointwise_conv1"]["bias"].numpy()
                    
            if "depthwise_conv" in conv_weights:
                converted[f"{prefix}/conv_depthwise/weight"] = conv_weights["depthwise_conv"]["weight"].numpy()
                if "bias" in conv_weights["depthwise_conv"]:
                    converted[f"{prefix}/conv_depthwise/bias"] = conv_weights["depthwise_conv"]["bias"].numpy()
                    
            if "pointwise_conv2" in conv_weights:
                converted[f"{prefix}/conv_pointwise2/weight"] = conv_weights["pointwise_conv2"]["weight"].numpy().T
                if "bias" in conv_weights["pointwise_conv2"]:
                    converted[f"{prefix}/conv_pointwise2/bias"] = conv_weights["pointwise_conv2"]["bias"].numpy()

        # Feed forward network
        if "ffn" in layer_weights:
            ffn_weights = layer_weights["ffn"]
            
            if "linear_0" in ffn_weights:
                converted[f"{prefix}/ffn/linear_0/weight"] = ffn_weights["linear_0"]["weight"].numpy().T
                if "bias" in ffn_weights["linear_0"]:
                    converted[f"{prefix}/ffn/linear_0/bias"] = ffn_weights["linear_0"]["bias"].numpy()
                    
            if "linear_1" in ffn_weights:
                converted[f"{prefix}/ffn/linear_1/weight"] = ffn_weights["linear_1"]["weight"].numpy().T
                if "bias" in ffn_weights["linear_1"]:
                    converted[f"{prefix}/ffn/linear_1/bias"] = ffn_weights["linear_1"]["bias"].numpy()

        return converted

    def _convert_decoder_layer(self, layer_weights: Dict[str, Any], layer_idx: int) -> Dict[str, np.ndarray]:
        """Convert a transformer decoder layer."""
        converted = {}
        prefix = f"decoder/layers/{layer_idx}"
        
        # Self-attention
        if "self_attn" in layer_weights:
            attn_weights = layer_weights["self_attn"]
            
            # Combined QKV projection (common in transformer implementations)
            if "qkv_proj" in attn_weights:
                qkv_weight = attn_weights["qkv_proj"]["weight"].numpy().T
                d_model = qkv_weight.shape[0]
                head_dim = d_model // 3
                
                converted[f"{prefix}/self_attention/linear_0/weight"] = qkv_weight[:, :head_dim]
                converted[f"{prefix}/self_attention/linear_1/weight"] = qkv_weight[:, head_dim:2*head_dim]
                converted[f"{prefix}/self_attention/linear_2/weight"] = qkv_weight[:, 2*head_dim:]
                
                if "bias" in attn_weights["qkv_proj"]:
                    qkv_bias = attn_weights["qkv_proj"]["bias"].numpy()
                    converted[f"{prefix}/self_attention/linear_0/bias"] = qkv_bias[:head_dim]
                    converted[f"{prefix}/self_attention/linear_1/bias"] = qkv_bias[head_dim:2*head_dim]
                    converted[f"{prefix}/self_attention/linear_2/bias"] = qkv_bias[2*head_dim:]
            
            # Output projection
            if "out_proj" in attn_weights:
                converted[f"{prefix}/self_attention/linear_3/weight"] = attn_weights["out_proj"]["weight"].numpy().T
                if "bias" in attn_weights["out_proj"]:
                    converted[f"{prefix}/self_attention/linear_3/bias"] = attn_weights["out_proj"]["bias"].numpy()

        # Cross-attention (encoder-decoder attention)
        if "cross_attn" in layer_weights:
            cross_attn_weights = layer_weights["cross_attn"]
            
            if "q_proj" in cross_attn_weights:
                converted[f"{prefix}/cross_attention/linear_0/weight"] = cross_attn_weights["q_proj"]["weight"].numpy().T
                if "bias" in cross_attn_weights["q_proj"]:
                    converted[f"{prefix}/cross_attention/linear_0/bias"] = cross_attn_weights["q_proj"]["bias"].numpy()
                    
            if "kv_proj" in cross_attn_weights:
                kv_weight = cross_attn_weights["kv_proj"]["weight"].numpy().T
                d_model = kv_weight.shape[0]
                head_dim = d_model // 2
                
                converted[f"{prefix}/cross_attention/linear_1/weight"] = kv_weight[:, :head_dim]
                converted[f"{prefix}/cross_attention/linear_2/weight"] = kv_weight[:, head_dim:]
                
                if "bias" in cross_attn_weights["kv_proj"]:
                    kv_bias = cross_attn_weights["kv_proj"]["bias"].numpy()
                    converted[f"{prefix}/cross_attention/linear_1/bias"] = kv_bias[:head_dim]
                    converted[f"{prefix}/cross_attention/linear_2/bias"] = kv_bias[head_dim:]

        return converted

    def _convert_weights(self) -> Dict[str, np.ndarray]:
        """Convert model weights to CTranslate2 format."""
        weights = {}
        state_dict = self._model_weights.get('state_dict', self._model_weights)
        
        # Convert encoder
        encoder_layers = []
        for key, value in state_dict.items():
            if key.startswith('encoder.layers.'):
                layer_idx = int(key.split('.')[2])
                if layer_idx >= len(encoder_layers):
                    encoder_layers.extend([{} for _ in range(layer_idx + 1 - len(encoder_layers))])
                
                layer_key = '.'.join(key.split('.')[3:])
                encoder_layers[layer_idx][layer_key] = value
        
        for i, layer_weights in enumerate(encoder_layers):
            weights.update(self._convert_encoder_layer(layer_weights, i))
        
        # Convert decoder
        decoder_layers = []
        for key, value in state_dict.items():
            if key.startswith('decoder.layers.'):
                layer_idx = int(key.split('.')[2])
                if layer_idx >= len(decoder_layers):
                    decoder_layers.extend([{} for _ in range(layer_idx + 1 - len(decoder_layers))])
                
                layer_key = '.'.join(key.split('.')[3:])
                decoder_layers[layer_idx][layer_key] = value
        
        for i, layer_weights in enumerate(decoder_layers):
            weights.update(self._convert_decoder_layer(layer_weights, i))
        
        # Convert embeddings and other components
        for key, value in state_dict.items():
            if key.startswith('encoder.embed_tokens'):
                weights["encoder/embeddings/weight"] = value.numpy()
            elif key.startswith('decoder.embed_tokens'):
                weights["decoder/embeddings/weight"] = value.numpy()
            elif key.startswith('decoder.output_projection'):
                weights["decoder/projection/weight"] = value.numpy().T
        
        return weights

    def convert(self, output_dir: str, quantization: Optional[str] = None, **kwargs) -> None:
        """Convert the model to CTranslate2 format.
        
        Args:
            output_dir: Output directory for the converted model
            quantization: Quantization type (e.g., "int8", "int16", "float16")
            **kwargs: Additional conversion arguments
        """
        self._load()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model specification
        activation_type = common_spec.Activation.SWISH  # FastConformer typically uses Swish
        spec = self._get_model_spec(activation_type)
        
        # Convert weights
        weights = self._convert_weights()
        
        # Get vocabulary
        vocabulary = self._get_vocabulary()
        
        # Save vocabulary
        vocab_path = os.path.join(output_dir, "vocabulary.txt")
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            for token in vocabulary:
                vocab_file.write(token + "\n")
        
        # Save model
        utils.save_model(
            spec=spec,
            weights=weights,
            output_dir=output_dir,
            quantization=quantization,
        )


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo Canary models to CTranslate2")
    parser.add_argument("--model_path", required=True, help="Path to NeMo .nemo file or model directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for converted model")
    parser.add_argument("--vocab_path", help="Optional path to vocabulary file")
    parser.add_argument("--quantization", choices=["int8", "int16", "float16"], help="Quantization type")
    
    args = parser.parse_args()
    
    converter = CanaryConverter(args.model_path, args.vocab_path)
    converter.convert(args.output_dir, args.quantization)
    
    print(f"Model converted successfully to {args.output_dir}")


if __name__ == "__main__":
    main()
