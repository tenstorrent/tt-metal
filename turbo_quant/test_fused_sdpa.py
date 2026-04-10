#!/usr/bin/env python3
"""Test TQ SDPA with BFP4 K/V (typecast in compute kernel)."""
import torch
import ttnn


def reference_sdpa(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def test_bfp4_sdpa(device, seq_len=128, head_dim=128, nqh=1, nkh=1):
    """Test TQ SDPA with BFP4 K/V (compute kernel typecasts BFP4→BF16)."""
    print(f"\n=== BFP4 SDPA test (seq={seq_len}, heads={nqh}/{nkh}) ===")
    B = 1
    scale = head_dim**-0.5
    seq_padded = ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)

    q_raw = torch.randn(B, nqh, 1, head_dim)
    k_raw = torch.randn(B, nkh, seq_padded, head_dim)
    v_raw = torch.randn(B, nkh, seq_padded, head_dim)

    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_bfp4 = ttnn.from_torch(k_raw, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_bfp4 = ttnn.from_torch(v_raw, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    # BFP4 roundtrip values for CPU reference
    k_rt = ttnn.to_torch(k_bfp4).float()
    v_rt = ttnn.to_torch(v_bfp4).float()

    heads_per_kv = nqh // nkh
    k_exp = k_rt.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else k_rt
    v_exp = v_rt.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else v_rt
    ref_out = reference_sdpa(q_raw, k_exp, v_exp, scale)

    # Standard SDPA with BFP4-rounded BF16 data (ground truth on device)
    k_bf16 = ttnn.from_torch(k_rt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_bf16 = ttnn.from_torch(v_rt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    std_out = ttnn.transformer.scaled_dot_product_attention(q_dev, k_bf16, v_bf16, is_causal=False, scale=scale)
    std_cpu = ttnn.to_torch(std_out).float()
    ttnn.deallocate(std_out)
    ttnn.deallocate(k_bf16)
    ttnn.deallocate(v_bf16)

    # Dummy norms (not used in typecast-only mode)
    norms_cpu = torch.zeros(B, nkh, seq_padded, 32)
    k_norms_dev = ttnn.from_torch(norms_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_norms_dev = ttnn.from_torch(norms_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    page_table_dev = ttnn.from_torch(
        torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    cur_pos_dev = ttnn.from_torch(
        torch.tensor([seq_len - 1], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    centroids = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    print("  Calling turbo_quant_sdpa_decode (BFP4 typecast mode)...")
    out_dev = ttnn.experimental.turbo_quant_sdpa_decode(
        q_dev,
        k_bfp4,
        k_norms_dev,
        v_bfp4,
        v_norms_dev,
        page_table_dev,
        cur_pos_dev,
        centroids,
        scale,
    )
    out_cpu = ttnn.to_torch(out_dev).float()

    for t in [q_dev, k_bfp4, v_bfp4, k_norms_dev, v_norms_dev, page_table_dev, cur_pos_dev, out_dev]:
        ttnn.deallocate(t)

    print(f"  TQ range: [{out_cpu.min():.4f}, {out_cpu.max():.4f}]")

    if torch.isnan(out_cpu).any() or torch.isinf(out_cpu).any():
        print(f"  FAIL: NaN/Inf detected")
        print(f"  TQ[0,0,0,:8]: {out_cpu[0,0,0,:8].tolist()}")
        return False

    cos_ref = torch.nn.functional.cosine_similarity(
        out_cpu.flatten().unsqueeze(0), ref_out.flatten().unsqueeze(0)
    ).item()
    cos_std = torch.nn.functional.cosine_similarity(
        out_cpu.flatten().unsqueeze(0), std_cpu.flatten().unsqueeze(0)
    ).item()
    print(f"  TQ vs CPU ref: {cos_ref:.6f}")
    print(f"  TQ vs std SDPA: {cos_std:.6f}")

    if cos_ref > 0.95:
        print("  PASS")
    else:
        print(f"  TQ[0,0,0,:8]:  {out_cpu[0,0,0,:8].tolist()}")
        print(f"  Std[0,0,0,:8]: {std_cpu[0,0,0,:8].tolist()}")
        print("  FAIL")
    return cos_ref > 0.95


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_bfp4_sdpa(device, nqh=1, nkh=1)
        test_bfp4_sdpa(device, nqh=8, nkh=8)
        test_bfp4_sdpa(device, nqh=32, nkh=8)
    finally:
        ttnn.close_device(device)
