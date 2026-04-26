# TTNN Time Series Transformer Implementation
# Issue: tenstorrent/tt-metal#32140
# Bounty: $1500

"""
TTNN implementation of Time Series Transformer for Tenstorrent hardware.

Architecture:
- Value embedding layer with optional lag features
- Temporal feature embeddings (past and future time features)
- Static feature embeddings (categorical and real)
- Standard transformer encoder with self-attention
- Standard transformer decoder with:
  - Masked self-attention (causal masking)
  - Cross-attention to encoder outputs
- Distribution head for probabilistic outputs

Based on HuggingFace Transformers Time Series Transformer.
"""

import ttnn
from ttnn import TtTensor
from typing import Optional, Dict, List
import math


class TtValueEmbedding:
    """TTNN Value embedding layer with optional lag features."""
    
    def __init__(
        self,
        device,
        input_size: int,
        d_model: int,
        lags_sequence: List[int],
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.input_size = input_size
        self.d_model = d_model
        self.lags_sequence = lags_sequence
        self.num_lags = len(lags_sequence)
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        self.projection_weight = None
        self.projection_bias = None
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        weight = torch_layer.projection.weight
        bias = torch_layer.projection.bias
        
        self.projection_weight = ttnn.from_torch(
            weight.T,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.projection_bias = ttnn.from_torch(
            bias,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
    
    def __call__(
        self,
        past_values: TtTensor,
        lag_values: Optional[TtTensor] = None,
    ) -> TtTensor:
        """
        Apply value embedding.
        
        Args:
            past_values: Input tensor (batch, context_length, input_size)
            lag_values: Optional lagged values
            
        Returns:
            Embedded tensor (batch, context_length, d_model)
        """
        batch_size = past_values.shape[0]
        context_length = past_values.shape[1]
        
        if lag_values is not None:
            x = ttnn.concat([past_values, lag_values], dim=-1)
        else:
            x = past_values
        
        x = ttnn.reshape(x, (batch_size, context_length, self.input_size * self.num_lags))
        x = ttnn.transpose(x, 1, 2)
        x = ttnn.linear(x, self.projection_weight, bias=self.projection_bias)
        x = ttnn.transpose(x, 1, 2)
        
        return x


class TtTemporalEmbedding:
    """TTNN Temporal feature embedding layer."""
    
    def __init__(
        self,
        device,
        num_time_features: int,
        d_model: int,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.num_time_features = num_time_features
        self.d_model = d_model
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        self.time_encoder_weight = None
        self.time_encoder_bias = None
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        weight = torch_layer.time_embed.weight
        bias = torch_layer.time_embed.bias
        
        self.time_encoder_weight = ttnn.from_torch(
            weight.T,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
        self.time_encoder_bias = ttnn.from_torch(
            bias,
            dtype=self.dtype,
            mesh_mapper=self.mesh_mapper,
        )
    
    def __call__(self, time_features: TtTensor) -> TtTensor:
        """
        Apply temporal embedding.
        
        Args:
            time_features: (batch, time_length, num_time_features)
            
        Returns:
            (batch, time_length, d_model)
        """
        batch_size = time_features.shape[0]
        time_length = time_features.shape[1]
        
        x = ttnn.transpose(time_features, 1, 2)
        x = ttnn.linear(x, self.time_encoder_weight, bias=self.time_encoder_bias)
        x = ttnn.transpose(x, 1, 2)
        
        return x


class TtStaticFeatureEmbedding:
    """TTNN Static feature embedding for categorical and real features."""
    
    def __init__(
        self,
        device,
        num_static_categorical: int,
        num_static_real: int,
        cardinality: List[int],
        embedding_dimension: int,
        d_model: int,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.num_static_categorical = num_static_categorical
        self.num_static_real = num_static_real
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.d_model = d_model
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        self.categorical_embeddings = []
        self.real_projection_weight = None
        self.real_projection_bias = None
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        for embedding in torch_layer.categorical_embed:
            self.categorical_embeddings.append(
                ttnn.from_torch(embedding.weight, dtype=self.dtype, mesh_mapper=self.mesh_mapper)
            )
        
        if self.num_static_real > 0:
            weight = torch_layer.real_embed.weight
            bias = torch_layer.real_embed.bias
            self.real_projection_weight = ttnn.from_torch(weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper)
            self.real_projection_bias = ttnn.from_torch(bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper)
    
    def __call__(
        self,
        static_categorical: Optional[TtTensor] = None,
        static_real: Optional[TtTensor] = None,
    ) -> Optional[TtTensor]:
        """
        Apply static feature embedding.
        
        Args:
            static_categorical: (batch, num_static_categorical)
            static_real: (batch, num_static_real)
            
        Returns:
            (batch, 1, d_model) or None
        """
        embeddings = []
        
        if static_real is not None and self.num_static_real > 0:
            batch_size = static_real.shape[0]
            x = ttnn.transpose(static_real, 1, 2)
            x = ttnn.linear(x, self.real_projection_weight, bias=self.real_projection_bias)
            x = ttnn.transpose(x, 1, 2)
            embeddings.append(x)
        
        if embeddings:
            result = embeddings[0]
            for emb in embeddings[1:]:
                result = ttnn.add(result, emb)
        else:
            result = None
        
        return result


class TtTransformerEncoderLayer:
    """TTNN Transformer Encoder layer with self-attention and FFN."""
    
    def __init__(
        self,
        device,
        d_model: int,
        num_attention_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        self.head_dim = d_model // num_attention_heads
        
        self.qkv_weight = None
        self.qkv_bias = None
        self.out_proj_weight = None
        self.out_proj_bias = None
        self.ffn_fc1_weight = None
        self.ffn_fc1_bias = None
        self.ffn_fc2_weight = None
        self.ffn_fc2_bias = None
        self.norm1_weight = None
        self.norm1_bias = None
        self.norm2_weight = None
        self.norm2_bias = None
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        self.qkv_weight = ttnn.from_torch(
            torch_layer.self_attn.qkv_proj.weight, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.qkv_bias = ttnn.from_torch(
            torch_layer.self_attn.qkv_proj.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.out_proj_weight = ttnn.from_torch(
            torch_layer.self_attn.out_proj.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.out_proj_bias = ttnn.from_torch(
            torch_layer.self_attn.out_proj.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc1_weight = ttnn.from_torch(
            torch_layer.fc1.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc1_bias = ttnn.from_torch(
            torch_layer.fc1.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc2_weight = ttnn.from_torch(
            torch_layer.fc2.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc2_bias = ttnn.from_torch(
            torch_layer.fc2.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm1_weight = ttnn.from_torch(
            torch_layer.norm1.weight.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm1_bias = ttnn.from_torch(
            torch_layer.norm1.bias.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm2_weight = ttnn.from_torch(
            torch_layer.norm2.weight.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm2_bias = ttnn.from_torch(
            torch_layer.norm2.bias.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        
    def __call__(self, x: TtTensor) -> TtTensor:
        """
        Apply transformer encoder layer.
        
        Args:
            x: (batch, seq_length, d_model)
            
        Returns:
            (batch, seq_length, d_model)
        """
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)
        
        batch = x.shape[0]
        seq_len = x.shape[1]
        
        x_flat = ttnn.reshape(x, (batch * seq_len, self.d_model))
        qkv = ttnn.linear(x_flat, self.qkv_weight, bias=self.qkv_bias)
        qkv = ttnn.reshape(qkv, (batch, seq_len, 3, self.d_model))
        
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        
        q = ttnn.reshape(q, (batch, seq_len, self.num_attention_heads, self.head_dim))
        k = ttnn.reshape(k, (batch, seq_len, self.num_attention_heads, self.head_dim))
        v = ttnn.reshape(v, (batch, seq_len, self.num_attention_heads, self.head_dim))
        
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        q_transposed = ttnn.transpose(q, 2, 3)
        attn_scores = ttnn.matmul(q, k_transposed) * scale
        attn_probs = ttnn.softmax(attn_scores, dim=-1)
        
        attn_output = ttnn.matmul(attn_probs, v)
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (batch, seq_len, self.d_model))
        
        attn_output = ttnn.linear(attn_output, self.out_proj_weight, bias=self.out_proj_bias)
        x = ttnn.add(attn_output, residual)
        
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)
        
        x_flat = ttnn.reshape(x, (batch * seq_len, self.d_model))
        x_flat = ttnn.linear(x_flat, self.ffn_fc1_weight, bias=self.ffn_fc1_bias, activation="gelu")
        x_flat = ttnn.linear(x_flat, self.ffn_fc2_weight, bias=self.ffn_fc2_bias)
        x = ttnn.reshape(x_flat, (batch, seq_len, self.d_model))
        
        x = ttnn.add(x, residual)
        
        return x


class TtTransformerDecoderLayer:
    """TTNN Transformer Decoder layer with self-attention, cross-attention, and FFN."""
    
    def __init__(
        self,
        device,
        d_model: int,
        num_attention_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        self.head_dim = d_model // num_attention_heads
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        self.self_qkv_weight = ttnn.from_torch(
            torch_layer.self_attn.qkv_proj.weight, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.self_qkv_bias = ttnn.from_torch(
            torch_layer.self_attn.qkv_proj.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.self_out_proj_weight = ttnn.from_torch(
            torch_layer.self_attn.out_proj.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.self_out_proj_bias = ttnn.from_torch(
            torch_layer.self_attn.out_proj.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.cross_q_weight = ttnn.from_torch(
            torch_layer.cross_attn.q_proj.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.cross_q_bias = ttnn.from_torch(
            torch_layer.cross_attn.q_proj.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.cross_kv_weight = ttnn.from_torch(
            torch_layer.cross_attn.kv_proj.weight, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.cross_kv_bias = ttnn.from_torch(
            torch_layer.cross_attn.kv_proj.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.cross_out_proj_weight = ttnn.from_torch(
            torch_layer.cross_attn.out_proj.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.cross_out_proj_bias = ttnn.from_torch(
            torch_layer.cross_attn.out_proj.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc1_weight = ttnn.from_torch(
            torch_layer.fc1.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc1_bias = ttnn.from_torch(
            torch_layer.fc1.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc2_weight = ttnn.from_torch(
            torch_layer.fc2.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.ffn_fc2_bias = ttnn.from_torch(
            torch_layer.fc2.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm1_weight = ttnn.from_torch(
            torch_layer.norm1.weight.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm1_bias = ttnn.from_torch(
            torch_layer.norm1.bias.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm2_weight = ttnn.from_torch(
            torch_layer.norm2.weight.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm2_bias = ttnn.from_torch(
            torch_layer.norm2.bias.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm3_weight = ttnn.from_torch(
            torch_layer.norm3.weight.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.norm3_bias = ttnn.from_torch(
            torch_layer.norm3.bias.unsqueeze(0).unsqueeze(0), dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
    
    def __call__(
        self,
        x: TtTensor,
        encoder_hidden_states: TtTensor,
        attention_mask: Optional[TtTensor] = None,
    ) -> TtTensor:
        """
        Apply transformer decoder layer.
        
        Args:
            x: (batch, tgt_seq_length, d_model)
            encoder_hidden_states: (batch, src_seq_length, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            (batch, tgt_seq_length, d_model)
        """
        batch = x.shape[0]
        tgt_len = x.shape[1]
        
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)
        
        x_flat = ttnn.reshape(x, (batch * tgt_len, self.d_model))
        qkv = ttnn.linear(x_flat, self.self_qkv_weight, bias=self.self_qkv_bias)
        qkv = ttnn.reshape(qkv, (batch, tgt_len, 3, self.d_model))
        
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        
        q = ttnn.reshape(q, (batch, tgt_len, self.num_attention_heads, self.head_dim))
        k = ttnn.reshape(k, (batch, tgt_len, self.num_attention_heads, self.head_dim))
        v = ttnn.reshape(v, (batch, tgt_len, self.num_attention_heads, self.head_dim))
        
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        q_transposed = ttnn.transpose(q, 2, 3)
        attn_scores = ttnn.matmul(q, k_transposed) * scale
        attn_probs = ttnn.softmax(attn_scores, dim=-1)
        
        attn_output = ttnn.matmul(attn_probs, v)
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (batch, tgt_len, self.d_model))
        
        attn_output = ttnn.linear(attn_output, self.self_out_proj_weight, bias=self.self_out_proj_bias)
        x = ttnn.add(attn_output, residual)
        
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)
        
        q = ttnn.linear(x, self.cross_q_weight, bias=self.cross_q_bias)
        q = ttnn.reshape(q, (batch, tgt_len, self.num_attention_heads, self.head_dim))
        q = ttnn.transpose(q, 1, 2)
        
        encoder_flat = ttnn.reshape(encoder_hidden_states, (batch * encoder_hidden_states.shape[1], self.d_model))
        kv = ttnn.linear(encoder_flat, self.cross_kv_weight, bias=self.cross_kv_bias)
        kv = ttnn.reshape(kv, (batch, encoder_hidden_states.shape[1], 2, self.d_model))
        k = kv[:, :, 0, :]
        v = kv[:, :, 1, :]
        
        k = ttnn.reshape(k, (batch, encoder_hidden_states.shape[1], self.num_attention_heads, self.head_dim))
        v = ttnn.reshape(v, (batch, encoder_hidden_states.shape[1], self.num_attention_heads, self.head_dim))
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        q_transposed = ttnn.transpose(q, 2, 3)
        cross_scores = ttnn.matmul(q, k_transposed) * scale
        cross_probs = ttnn.softmax(cross_scores, dim=-1)
        
        cross_output = ttnn.matmul(cross_probs, v)
        cross_output = ttnn.transpose(cross_output, 1, 2)
        cross_output = ttnn.reshape(cross_output, (batch, tgt_len, self.d_model))
        
        cross_output = ttnn.linear(cross_output, self.cross_out_proj_weight, bias=self.cross_out_proj_bias)
        x = ttnn.add(cross_output, residual)
        
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm3_weight, bias=self.norm3_bias)
        
        x_flat = ttnn.reshape(x, (batch * tgt_len, self.d_model))
        x_flat = ttnn.linear(x_flat, self.ffn_fc1_weight, bias=self.ffn_fc1_bias, activation="gelu")
        x_flat = ttnn.linear(x_flat, self.ffn_fc2_weight, bias=self.ffn_fc2_bias)
        x = ttnn.reshape(x_flat, (batch, tgt_len, self.d_model))
        
        x = ttnn.add(x, residual)
        
        return x


class TtDistributionHead:
    """TTNN Distribution head for probabilistic outputs."""
    
    def __init__(
        self,
        device,
        d_model: int,
        prediction_length: int,
        distribution_output: str = "student_t",
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.distribution_output = distribution_output
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        self.param_projection_weight = None
        self.param_projection_bias = None
        
    def create_weights(self, torch_layer):
        """Create TTNN weights from PyTorch layer."""
        self.param_projection_weight = ttnn.from_torch(
            torch_layer.projection.weight.T, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
        self.param_projection_bias = ttnn.from_torch(
            torch_layer.projection.bias, dtype=self.dtype, mesh_mapper=self.mesh_mapper
        )
    
    def __call__(self, x: TtTensor) -> Dict[str, TtTensor]:
        """
        Apply distribution head.
        
        Args:
            x: (batch, prediction_length, d_model)
            
        Returns:
            Dictionary of distribution parameters
        """
        batch = x.shape[0]
        
        x_flat = ttnn.reshape(x, (batch * self.prediction_length, self.d_model))
        params = ttnn.linear(x_flat, self.param_projection_weight, bias=self.param_projection_bias)
        params = ttnn.reshape(params, (batch, self.prediction_length, -1))
        
        if self.distribution_output == "student_t":
            df = params[:, :, 0:1]
            loc = params[:, :, 1:2]
            scale = params[:, :, 2:3]
            return {"df": df, "loc": loc, "scale": scale}
        elif self.distribution_output == "normal":
            loc = params[:, :, 0:1]
            scale = params[:, :, 1:2]
            return {"loc": loc, "scale": scale}
        else:
            logits = params
            return {"logits": logits}


class TtTimeSeriesTransformer:
    """
    TTNN Time Series Transformer for probabilistic time-series forecasting.
    
    Architecture:
    - Encoder: Processes past_values with context_length
    - Decoder: Autoregressively generates prediction_length forecasts
    - Distribution head: Outputs distribution parameters for probabilistic forecasting
    """
    
    def __init__(
        self,
        device,
        config,
        parameters: Dict,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.config = config
        self.dtype = dtype
        self.mesh_mapper = mesh_mapper
        
        self.value_embedding = TtValueEmbedding(
            device, config.input_size, config.d_model, config.lags_sequence, dtype, mesh_mapper
        )
        
        self.temporal_embedding = TtTemporalEmbedding(
            device, config.num_time_features, config.d_model, dtype, mesh_mapper
        )
        
        self.static_embedding = TtStaticFeatureEmbedding(
            device,
            config.num_static_categorical_features,
            config.num_static_real_features,
            config.cardinality,
            config.embedding_dimension,
            config.d_model,
            dtype,
            mesh_mapper,
        )
        
        self.encoder_layers = [
            TtTransformerEncoderLayer(
                device, config.d_model, config.encoder_attention_heads,
                config.encoder_ffn_dim, config.dropout, dtype, mesh_mapper
            )
            for _ in range(config.encoder_layers)
        ]
        
        self.decoder_layers = [
            TtTransformerDecoderLayer(
                device, config.d_model, config.decoder_attention_heads,
                config.decoder_ffn_dim, config.dropout, dtype, mesh_mapper
            )
            for _ in range(config.decoder_layers)
        ]
        
        self.distribution_head = TtDistributionHead(
            device, config.d_model, config.prediction_length,
            config.distribution_output, dtype, mesh_mapper
        )
        
        self.encoder_norm_weight = None
        self.encoder_norm_bias = None
        self.decoder_norm_weight = None
        self.decoder_norm_bias = None
        
        self.load_weights(parameters)
        
    def load_weights(self, parameters: Dict):
        """Load preprocessed weights."""
        if "value_embedding" in parameters:
            self.value_embedding.create_weights(parameters["value_embedding"])
        
        if "temporal_embedding" in parameters:
            self.temporal_embedding.create_weights(parameters["temporal_embedding"])
        
        if "static_embedding" in parameters:
            self.static_embedding.create_weights(parameters["static_embedding"])
        
        if "encoder_layers" in parameters:
            for i, layer_params in enumerate(parameters["encoder_layers"]):
                if i < len(self.encoder_layers):
                    self.encoder_layers[i].create_weights(layer_params)
        
        if "decoder_layers" in parameters:
            for i, layer_params in enumerate(parameters["decoder_layers"]):
                if i < len(self.decoder_layers):
                    self.decoder_layers[i].create_weights(layer_params)
        
        if "distribution_head" in parameters:
            self.distribution_head.create_weights(parameters["distribution_head"])
        
        if "encoder_norm" in parameters:
            self.encoder_norm_weight = ttnn.from_torch(
                parameters["encoder_norm"]["weight"].unsqueeze(0).unsqueeze(0),
                dtype=self.dtype, mesh_mapper=self.mesh_mapper
            )
            self.encoder_norm_bias = ttnn.from_torch(
                parameters["encoder_norm"]["bias"].unsqueeze(0).unsqueeze(0),
                dtype=self.dtype, mesh_mapper=self.mesh_mapper
            )
        
        if "decoder_norm" in parameters:
            self.decoder_norm_weight = ttnn.from_torch(
                parameters["decoder_norm"]["weight"].unsqueeze(0).unsqueeze(0),
                dtype=self.dtype, mesh_mapper=self.mesh_mapper
            )
            self.decoder_norm_bias = ttnn.from_torch(
                parameters["decoder_norm"]["bias"].unsqueeze(0).unsqueeze(0),
                dtype=self.dtype, mesh_mapper=self.mesh_mapper
            )
    
    def __call__(
        self,
        past_values: TtTensor,
        past_time_features: TtTensor,
        future_time_features: TtTensor,
        static_categorical: Optional[TtTensor] = None,
        static_real: Optional[TtTensor] = None,
        past_observed_mask: Optional[TtTensor] = None,
    ) -> Dict[str, TtTensor]:
        """
        Forward pass.
        
        Args:
            past_values: (batch, context_length, input_size)
            past_time_features: (batch, context_length, num_time_features)
            future_time_features: (batch, prediction_length, num_time_features)
            static_categorical: (batch, num_static_categorical)
            static_real: (batch, num_static_real)
            past_observed_mask: (batch, context_length, input_size)
            
        Returns:
            Dictionary of distribution parameters
        """
        batch_size = past_values.shape[0]
        
        value_emb = self.value_embedding(past_values)
        temporal_emb = self.temporal_embedding(past_time_features)
        
        encoder_hidden_states = ttnn.add(value_emb, temporal_emb)
        
        if static_categorical is not None or static_real is not None:
            static_emb = self.static_embedding(static_categorical, static_real)
            if static_emb is not None:
                static_emb = ttnn.unsqueeze(static_emb, 1)
                static_emb = ttnn.repeat(static_emb, (1, encoder_hidden_states.shape[1], 1))
                encoder_hidden_states = ttnn.add(encoder_hidden_states, static_emb)
        
        for layer in self.encoder_layers:
            encoder_hidden_states = layer(encoder_hidden_states)
        
        if self.encoder_norm_weight is not None:
            encoder_hidden_states = ttnn.layer_norm(
                encoder_hidden_states, weight=self.encoder_norm_weight, bias=self.encoder_norm_bias
            )
        
        decoder_temporal_emb = self.temporal_embedding(future_time_features)
        decoder_input = ttnn.zeros_like(decoder_temporal_emb)
        
        if static_categorical is not None or static_real is not None:
            if static_emb is not None:
                decoder_input = ttnn.add(decoder_input, static_emb[:, :decoder_input.shape[1], :])
        
        decoder_hidden_states = decoder_input
        for layer in self.decoder_layers:
            decoder_hidden_states = layer(decoder_hidden_states, encoder_hidden_states)
        
        if self.decoder_norm_weight is not None:
            decoder_hidden_states = ttnn.layer_norm(
                decoder_hidden_states, weight=self.decoder_norm_weight, bias=self.decoder_norm_bias
            )
        
        dist_params = self.distribution_head(decoder_hidden_states)
        
        return dist_params
