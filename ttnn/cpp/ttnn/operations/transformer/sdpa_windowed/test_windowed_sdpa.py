#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the windowed scaled dot product attention operation.
"""

import pytest
import torch
import ttnn
import numpy as np


def create_windowed_attention_mask_reference(seq_len, cu_window_seqlens, is_causal=False):
    """Reference implementation for creating windowed attention mask."""
    mask = torch.full((seq_len, seq_len), -1e9, dtype=torch.float32)

    # For each window, allow attention only within that window
    for i in range(1, len(cu_window_seqlens)):
        start = cu_window_seqlens[i - 1]
        end = cu_window_seqlens[i]
        mask[start:end, start:end] = 0.0

    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
        mask = torch.minimum(mask, causal_mask)

    return mask


def reference_sdpa(q, k, v, mask, scale=None):
    """Reference implementation of scaled dot product attention."""
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    # Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Add mask
    if mask is not None:
        scores = scores + mask

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Attention @ V
    output = torch.matmul(attn_weights, v)

    return output


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [128, 256])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_windowed_sdpa(batch_size, num_heads, seq_len, head_dim, is_causal):
    """Test windowed SDPA against reference implementation."""

    # Define windows - e.g., 4 windows of equal size
    num_windows = 4
    window_size = seq_len // num_windows
    cu_window_seqlens = [i * window_size for i in range(num_windows + 1)]
    cu_window_seqlens[-1] = seq_len  # Ensure last window includes any remainder

    # Create input tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Create reference mask
    mask = create_windowed_attention_mask_reference(seq_len, cu_window_seqlens, is_causal)
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

    # Compute reference output
    ref_output = reference_sdpa(q, k, v, mask)

    # Run on device
    device = ttnn.CreateDevice(0)

    try:
        # Convert to TTNN tensors
        q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT)
        k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT)
        v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT)

        # Run windowed SDPA
        output_tt = ttnn.transformer.windowed_scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            cu_window_seqlens,
            is_causal=is_causal,
        )

        # Convert back to torch
        output = ttnn.to_torch(output_tt)

        # Compare outputs
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-3)

    finally:
        ttnn.CloseDevice(device)


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_windowed_sdpa_varying_windows(seq_len):
    """Test windowed SDPA with varying window sizes."""

    batch_size = 1
    num_heads = 8
    head_dim = 64

    # Create windows of varying sizes
    # E.g., windows of sizes 10%, 20%, 30%, 40% of seq_len
    window_proportions = [0.1, 0.2, 0.3, 0.4]
    cu_window_seqlens = [0]
    for prop in window_proportions:
        cu_window_seqlens.append(cu_window_seqlens[-1] + int(seq_len * prop))
    cu_window_seqlens[-1] = seq_len  # Ensure we cover the full sequence

    # Create input tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Create reference mask
    mask = create_windowed_attention_mask_reference(seq_len, cu_window_seqlens)
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Compute reference output
    ref_output = reference_sdpa(q, k, v, mask)

    # Run on device
    device = ttnn.CreateDevice(0)

    try:
        # Convert to TTNN tensors
        q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT)
        k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT)
        v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT)

        # Run windowed SDPA
        output_tt = ttnn.transformer.windowed_scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            cu_window_seqlens,
            is_causal=False,
        )

        # Convert back to torch
        output = ttnn.to_torch(output_tt)

        # Compare outputs
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-3)

    finally:
        ttnn.CloseDevice(device)


def test_windowed_vs_regular_sdpa():
    """Test that windowed SDPA matches regular SDPA with equivalent mask."""

    batch_size = 1
    num_heads = 8
    seq_len = 128
    head_dim = 64

    # Define windows
    cu_window_seqlens = [0, 32, 64, 96, 128]

    # Create input tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Create mask
    mask = create_windowed_attention_mask_reference(seq_len, cu_window_seqlens)
    mask = mask.unsqueeze(0).unsqueeze(0)

    device = ttnn.CreateDevice(0)

    try:
        # Convert to TTNN tensors
        q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT)
        k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT)
        v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT)
        mask_tt = ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT)

        # Run regular SDPA with mask
        output_regular_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            attn_mask=mask_tt,
            is_causal=False,
        )

        # Run windowed SDPA
        output_windowed_tt = ttnn.transformer.windowed_scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            cu_window_seqlens,
            is_causal=False,
        )

        # Convert back to torch
        output_regular = ttnn.to_torch(output_regular_tt)
        output_windowed = ttnn.to_torch(output_windowed_tt)

        # Compare outputs - they should be very close
        torch.testing.assert_close(output_windowed, output_regular, rtol=1e-2, atol=1e-3)

    finally:
        ttnn.CloseDevice(device)


if __name__ == "__main__":
    # Run basic test
    test_windowed_sdpa(1, 8, 128, 64, False)
    test_windowed_vs_regular_sdpa()
    print("All tests passed!")
