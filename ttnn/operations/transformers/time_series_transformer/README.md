# Time Series Transformer TTNN Implementation

**Issue**: tenstorrent/tt-metal#32140  
**Bounty**: $1500  
**Status**: Stage 1 Complete

## Overview

This is a TTNN (Tenstorrent Neural Network) implementation of the Time Series Transformer model for probabilistic time-series forecasting on Tenstorrent hardware.

Based on the HuggingFace Transformers Time Series Transformer architecture.

## Architecture

The Time Series Transformer is a vanilla encoder-decoder Transformer architecture for probabilistic time-series forecasting:

- **Value Embedding**: Projects historical values with optional lag features
- **Temporal Feature Embeddings**: Encodes past and future time features
- **Static Feature Embeddings**: Handles categorical and real static features
- **Transformer Encoder**: Self-attention over historical context
- **Transformer Decoder**: 
  - Masked self-attention (causal masking)
  - Cross-attention to encoder outputs
- **Distribution Head**: Probabilistic outputs (Student-t, Normal, or Negative Binomial)

## Key Features

- **Probabilistic Forecasting**: Outputs distribution parameters instead of point estimates
- **Multiple Distribution Types**: Student-t (default), Normal, Negative Binomial
- **Rich Feature Support**: Past values, temporal features, static features, lag features
- **Teacher-Forcing Training**: Efficient training paradigm
- **Autoregressive Generation**: Flexible inference with multiple samples

## Usage

```python
import ttnn
from ttnn.operations.transformers.time_series_transformer import (
    TtTimeSeriesTransformer,
    TimeSeriesTransformerConfig,
)

# Initialize device
device = ttnn.open_device(device_id=0)

# Create config
config = TimeSeriesTransformerConfig(
    context_length=512,
    prediction_length=96,
    input_size=1,
    d_model=128,
)

# Create model with parameters
model = TtTimeSeriesTransformer(device, config, parameters)

# Prepare inputs
past_values_tt = ttnn.from_torch(past_values, dtype=ttnn.bfloat16)
past_time_features_tt = ttnn.from_torch(past_time_features, dtype=ttnn.bfloat16)
future_time_features_tt = ttnn.from_torch(future_time_features, dtype=ttnn.bfloat16)

# Forward pass
dist_params = model(past_values_tt, past_time_features_tt, future_time_features_tt)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_length` | 512 | Input context length |
| `prediction_length` | 96 | Output prediction length |
| `input_size` | 1 | Number of target variables |
| `d_model` | 128 | Model dimension |
| `encoder_layers` | 2 | Number of encoder layers |
| `decoder_layers` | 2 | Number of decoder layers |
| `encoder_attention_heads` | 4 | Number of encoder attention heads |
| `decoder_attention_heads` | 4 | Number of decoder attention heads |
| `encoder_ffn_dim` | 512 | Encoder feed-forward dimension |
| `decoder_ffn_dim` | 512 | Decoder feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `distribution_output` | "student_t" | Distribution type |

## Performance Targets

### Stage 1 (Complete)
- Model runs on Tenstorrent hardware without errors
- Inference throughput: ≥100 sequences/second
- Latency: <50ms for single sequence

### Stage 2 (Planned)
- Optimal memory configurations
- Efficient sharding strategies
- Operation fusion

### Stage 3 (Planned)
- Flash Attention or equivalent
- Optimized KV-cache management
- Pipeline encoder/decoder
- 500+ sequences/second throughput

## References

- [HuggingFace Time Series Transformer Documentation](https://huggingface.co/docs/transformers/en/model_doc/time_series_transformer)
- [TTNN Model Bring-up Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
- [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
