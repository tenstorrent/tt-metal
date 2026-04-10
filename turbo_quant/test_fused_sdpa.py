#!/usr/bin/env python3
"""Test fused TQ SDPA with full dequant (centroid gather × norm multiply)."""
import sys
import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from turbo_quant.quantizer import TurboQuantMSE


def reference_sdpa(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def test_fused_dequant_sdpa(device, seq_len=128, head_dim=128, nqh=8, nkh=8, bits=3):
    """Test full dequant pipeline: BFP4 indices + BF16 norms → centroid gather × norm → SDPA."""
    print(f"\n=== Fused dequant SDPA (seq={seq_len}, heads={nqh}/{nkh}, bits={bits}) ===")
    B = 1
    scale = head_dim**-0.5
    seq_padded = ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)

    quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    centroids = quantizer.codebook.centroids.tolist()
    print(f"  Centroids: {[f'{c:.4f}' for c in centroids]}")

    # Generate random K/V, quantize, dequantize for reference
    k_raw = torch.randn(B, nkh, seq_padded, head_dim)
    v_raw = torch.randn(B, nkh, seq_padded, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)

    k_indices, k_norms = quantizer.quantize(k_raw)  # uint8 [B,H,S,D], float [B,H,S,1]
    v_indices, v_norms = quantizer.quantize(v_raw)

    # Reference: centroid_gather(indices) × norms (no rotation — absorbed into weights)
    k_centroids = quantizer.codebook.dequantize(k_indices.long())
    v_centroids = quantizer.codebook.dequantize(v_indices.long())
    k_dequant = k_centroids * k_norms
    v_dequant = v_centroids * v_norms

    # CPU reference SDPA on dequantized data
    heads_per_kv = nqh // nkh
    k_exp = k_dequant.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else k_dequant
    v_exp = v_dequant.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else v_dequant
    ref_out = reference_sdpa(q_raw, k_exp, v_exp, scale)
    print(f"  Ref output range: [{ref_out.min():.4f}, {ref_out.max():.4f}]")

    # Send to device: indices as BFP4, norms as BF16
    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_idx_dev = ttnn.from_torch(k_indices.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_idx_dev = ttnn.from_torch(v_indices.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Norms: [B, H, S, 1] → pad last dim to 32 for tile layout
    k_norms_padded = torch.zeros(B, nkh, seq_padded, 32)
    v_norms_padded = torch.zeros(B, nkh, seq_padded, 32)
    k_norms_padded[..., 0] = k_norms.squeeze(-1)
    v_norms_padded[..., 0] = v_norms.squeeze(-1)
    k_norms_dev = ttnn.from_torch(k_norms_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_norms_dev = ttnn.from_torch(v_norms_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    page_table_dev = ttnn.from_torch(
        torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    cur_pos_dev = ttnn.from_torch(
        torch.tensor([seq_len - 1], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    print("  Calling turbo_quant_sdpa_decode (full dequant)...")
    out_dev = ttnn.experimental.turbo_quant_sdpa_decode(
        q_dev, k_idx_dev, k_norms_dev, v_idx_dev, v_norms_dev, page_table_dev, cur_pos_dev, centroids, scale
    )
    out_cpu = ttnn.to_torch(out_dev).float()
    for t in [q_dev, k_idx_dev, v_idx_dev, k_norms_dev, v_norms_dev, page_table_dev, cur_pos_dev, out_dev]:
        ttnn.deallocate(t)

    print(f"  Output range: [{out_cpu.min():.4f}, {out_cpu.max():.4f}]")

    if torch.isnan(out_cpu).any() or torch.isinf(out_cpu).any():
        print(f"  FAIL: NaN/Inf")
        print(f"  Output[0,0,0,:8]: {out_cpu[0,0,0,:8].tolist()}")
        return False

    cos = torch.nn.functional.cosine_similarity(out_cpu.flatten().unsqueeze(0), ref_out.flatten().unsqueeze(0)).item()
    print(f"  Cosine vs dequantized ref: {cos:.6f}")
    if cos > 0.95:
        print("  PASS")
    else:
        print(f"  Output[0,0,0,:8]: {out_cpu[0,0,0,:8].tolist()}")
        print(f"  Ref[0,0,0,:8]:    {ref_out[0,0,0,:8].tolist()}")
        print("  FAIL")
    return cos > 0.95


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        ok = True
        ok &= test_fused_dequant_sdpa(device, nqh=1, nkh=1)
        ok &= test_fused_dequant_sdpa(device, nqh=8, nkh=8)
        ok &= test_fused_dequant_sdpa(device, nqh=32, nkh=8)
        print(f"\n{'All tests passed!' if ok else 'Some tests FAILED'}")
    finally:
        ttnn.close_device(device)
