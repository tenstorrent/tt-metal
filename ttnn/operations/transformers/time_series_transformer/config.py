# Configuration for Time Series Transformer TTNN Implementation
# Issue: tenstorrent/tt-metal#32140
# Bounty: $1500

"""
Configuration module for Time Series Transformer TTNN implementation.
Implements a vanilla encoder-decoder Transformer for probabilistic time-series forecasting.
Based on HuggingFace Transformers Time Series Transformer architecture.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TimeSeriesTransformerConfig:
    """Configuration for Time Series Transformer model."""
    
    # Model architecture
    context_length: int = 512       # Input context length (past values)
    prediction_length: int = 96     # Output prediction length (future values)
    input_size: int = 1             # Number of target variables
    num_time_features: int = 5      # Number of time features
    
    # Transformer architecture
    d_model: int = 128              # Model dimension
    encoder_layers: int = 2         # Number of encoder layers
    decoder_layers: int = 2         # Number of decoder layers
    encoder_attention_heads: int = 4 # Number of attention heads
    decoder_attention_heads: int = 4 # Number of attention heads
    encoder_ffn_dim: int = 512      # Encoder feed-forward dimension
    decoder_ffn_dim: int = 512      # Decoder feed-forward dimension
    dropout: float = 0.1            # Dropout rate
    
    # Distribution parameters
    distribution_output: str = "student_t"  # Distribution type
    
    # Feature dimensions
    num_static_categorical_features: int = 0
    num_static_real_features: int = 0
    cardinality: List[int] = []     # Cardinalities for categorical features
    embedding_dimension: int = 16    # Embedding dimension
    
    # Lag features
    lags_sequence: List[int] = None  # Lag indices
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.lags_sequence is None:
            self.lags_sequence = [1, 2, 3, 4, 5, 6, 7, 14, 28]
