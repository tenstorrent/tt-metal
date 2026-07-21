import torch
import torch.nn as nn
from tt_metal.common import TT_DTYPE_TO_TORCH_DTYPE
from tt_metal.ttnn.operations.primary import concatenate, reshape, softmax, matmul, layernorm, gelu

def image_embedding(image_size: int, patch_size: int, embed_dim: int) -> nn.Module:
    """Create image embedding module"""
    num_patches = (image_size // patch_size) ** 2
    return nn.Sequential(
        nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
        nn.Flatten(2),
        nn.Linear(num_patches, embed_dim),
        layernorm(embed_dim)
    )

def text_embedding(vocab_size: int, embed_dim: int) -> nn.Module:
    """Create text embedding module"""
    return nn.Sequential(
        nn.Embedding(vocab_size, embed_dim),
        layernorm(embed_dim)
    )

def cross_attention(num_heads: int, embed_dim: int) -> nn.Module:
    """Create cross-modal attention module"""
    class CrossAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

        def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
            batch_size = image_embeddings.shape[0]

            # Project queries from image embeddings
            q = self.q_proj(image_embeddings)
            q = reshape(q, (batch_size, -1, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

            # Project keys and values from text embeddings
            k = self.k_proj(text_embeddings)
            v = self.v_proj(text_embeddings)
            k = reshape(k, (batch_size, -1, self.num_heads, self.head_dim)).permute(0, 2, 3, 1)
            v = reshape(v, (batch_size, -1, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

            # Calculate attention scores
            attn_scores = matmul(q, k) / (self.head_dim ** 0.5)
            attn_probs = softmax(attn_scores, dim=-1)

            # Apply attention to values
            output = matmul(attn_probs, v)
            output = output.permute(0, 2, 1, 3).reshape(batch_size, -1, embed_dim)

            return self.out_proj(output)

    return CrossAttention()