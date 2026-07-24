import torch
import torch.nn as nn
from typing import Optional, Tuple
from tt_metal.common import TT_DTYPE_TO_TORCH_DTYPE
from tt_metal.ttnn.operations.primary import concatenate, reshape, softmax, matmul, layernorm, gelu
from tt_metal.ttnn.operations.vision_language import image_embedding, text_embedding, cross_attention

class LFM25VL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_embedder = image_embedding(config.image_size, config.patch_size, config.embed_dim)
        self.text_embedder = text_embedding(config.vocab_size, config.embed_dim)
        self.cross_attention = cross_attention(config.num_heads, config.embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image processing
        image_embeddings = self.image_embedder(image)

        # Text processing
        text_embeddings = self.text_embedder(text)

        # Cross-modal attention
        combined_embeddings = self.cross_attention(image_embeddings, text_embeddings)

        # Transformer layers
        for layer in self.layers:
            combined_embeddings = layer(combined_embeddings)

        return combined_embeddings

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.layernorm1 = layernorm(config.embed_dim)
        self.layernorm2 = layernorm(config.embed_dim)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv_proj(x)
        qkv = reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = softmax(attn_scores, dim=-1)
        output = matmul(attn_probs, v)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        return self.out_proj(output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.activation = gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x