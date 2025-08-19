# Windowed Scaled Dot Product Attention

This directory contains the implementation of windowed scaled dot product attention (SDPA) for TTNN. The windowed SDPA operation is a specialized variant of standard SDPA that efficiently computes attention within predefined windows, creating a block-diagonal attention pattern.

## Key Differences from Standard SDPA

1. **Interface**: Instead of accepting an explicit `attn_mask` tensor, windowed SDPA accepts a list of cumulative window sequence lengths (`cu_window_seqlens`).

2. **Memory Efficiency**: The attention mask is generated on-the-fly within the kernel, avoiding the need to store and transfer large attention mask tensors.

3. **Use Case**: Particularly useful for vision transformers with windowed attention mechanisms (e.g., Qwen2.5-VL) where attention is restricted to specific windows in the sequence.

## API

```python
ttnn.transformer.windowed_scaled_dot_product_attention(
    input_tensor_q,      # Query tensor [B x NH x S x D]
    input_tensor_k,      # Key tensor [B x NH x S x D]
    input_tensor_v,      # Value tensor [B x NH x S x D]
    cu_window_seqlens,   # List of cumulative window lengths
    is_causal=True,      # Apply causal masking within windows
    scale=None,          # Scale factor (defaults to 1/sqrt(D))
    memory_config=None,  # Memory configuration
    program_config=None, # SDPA program configuration
    compute_kernel_config=None,
    queue_id=0
)
```

## Example

```python
# Define windows: 3 windows of sizes 10, 15, and 20 tokens
cu_window_seqlens = [0, 10, 25, 45]

# This creates attention windows:
# - Window 1: tokens 0-9 attend to each other
# - Window 2: tokens 10-24 attend to each other
# - Window 3: tokens 25-44 attend to each other

output = ttnn.transformer.windowed_scaled_dot_product_attention(
    q, k, v, cu_window_seqlens
)
```
