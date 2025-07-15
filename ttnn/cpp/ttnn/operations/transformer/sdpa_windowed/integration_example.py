#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration example showing how to use windowed SDPA in vision transformers.
This demonstrates how the Qwen2.5-VL model could be modified to use the windowed SDPA.
"""

import torch
import ttnn


def vision_attention_with_windowed_sdpa(
    q_heads,  # Query heads tensor
    k_heads,  # Key heads tensor
    v_heads,  # Value heads tensor
    cu_window_seqlens,  # Cumulative window sequence lengths
    scale=None,
    is_causal=False,
    compute_kernel_config=None,
    program_config=None,
):
    """
    Example of using windowed SDPA instead of regular SDPA with explicit mask.

    This function would replace the current SDPA call in VisionAttention.forward_prefill()
    """

    # Convert cu_window_seqlens to list if it's a tensor
    if isinstance(cu_window_seqlens, torch.Tensor):
        cu_window_seqlens_list = cu_window_seqlens.tolist()
    else:
        cu_window_seqlens_list = cu_window_seqlens

    # Use windowed SDPA - no need to pass attention mask
    attn_output = ttnn.transformer.windowed_scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        cu_window_seqlens_list,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )

    return attn_output


def modified_vision_block_forward(
    vision_block,
    x,
    rot_mats,
    cu_window_seqlens=None,  # New parameter
    tt_mask=None,  # Keep for compatibility
):
    """
    Example of modified VisionBlock forward that can use either windowed SDPA or regular SDPA.
    """
    # If cu_window_seqlens is provided, use windowed SDPA
    if cu_window_seqlens is not None:
        # Pass cu_window_seqlens to attention module
        # This would require modifying VisionAttention.forward() to accept cu_window_seqlens
        attn_out = vision_block.attention.forward(
            x,
            rot_mats=rot_mats,
            cu_window_seqlens=cu_window_seqlens,
            tt_mask=None,  # Don't need mask when using windowed SDPA
        )
    else:
        # Fall back to regular SDPA with mask
        attn_out = vision_block.attention.forward(
            x,
            rot_mats=rot_mats,
            tt_mask=tt_mask,
        )

    # Rest of the block forward logic remains the same
    return attn_out


def performance_comparison_example():
    """
    Example showing the performance benefits of windowed SDPA.
    """
    import time

    # Example parameters
    batch_size = 1
    num_heads = 8
    seq_len = 2048
    head_dim = 64

    # Define windows
    cu_window_seqlens = [0, 512, 1024, 1536, 2048]

    # Create tensors
    device = ttnn.CreateDevice(0)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT)
    k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT)
    v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT)

    # Method 1: Regular SDPA with explicit mask (current approach)
    start_time = time.time()

    # Create windowed attention mask
    attention_mask = torch.full([1, 1, seq_len, seq_len], -1e9, dtype=torch.bfloat16)
    for i in range(1, len(cu_window_seqlens)):
        attention_mask[
            ...,
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
            cu_window_seqlens[i - 1] : cu_window_seqlens[i],
        ] = 0

    # Transfer mask to device (this is expensive for large sequences!)
    mask_tt = ttnn.from_torch(attention_mask, device=device, layout=ttnn.TILE_LAYOUT)

    # Run regular SDPA
    output1 = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        attn_mask=mask_tt,
        is_causal=False,
    )

    method1_time = time.time() - start_time

    # Method 2: Windowed SDPA (new approach)
    start_time = time.time()

    # No mask creation or transfer needed!
    output2 = ttnn.transformer.windowed_scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        cu_window_seqlens,
        is_causal=False,
    )

    method2_time = time.time() - start_time

    print(f"Regular SDPA with mask: {method1_time:.4f}s")
    print(f"Windowed SDPA: {method2_time:.4f}s")
    print(f"Speedup: {method1_time / method2_time:.2f}x")

    # Benefits:
    # 1. No mask tensor creation (saves host memory)
    # 2. No mask transfer to device (saves time and device memory)
    # 3. Mask generated on-the-fly in kernel (better performance)

    ttnn.CloseDevice(device)


if __name__ == "__main__":
    performance_comparison_example()
