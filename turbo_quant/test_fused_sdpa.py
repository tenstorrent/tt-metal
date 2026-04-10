#!/usr/bin/env python3
"""Test TQ SDPA with BF16 K/V (K transposed in reader, matching standard SDPA)."""
import torch
import ttnn


def reference_sdpa(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def test_bf16_sdpa(device, seq_len=128, head_dim=128, nqh=1, nkh=1):
    print(f"\n=== BF16 SDPA test (seq={seq_len}, heads={nqh}/{nkh}) ===")
    B = 1
    scale = head_dim**-0.5
    seq_padded = ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)

    q_raw = torch.randn(B, nqh, 1, head_dim)
    k_raw = torch.randn(B, nkh, seq_padded, head_dim)
    v_raw = torch.randn(B, nkh, seq_padded, head_dim)

    heads_per_kv = nqh // nkh
    k_exp = k_raw.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else k_raw
    v_exp = v_raw.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else v_raw
    ref_out = reference_sdpa(q_raw, k_exp, v_exp, scale)

    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_dev = ttnn.from_torch(k_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_dev = ttnn.from_torch(v_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    std_out = ttnn.transformer.scaled_dot_product_attention(q_dev, k_dev, v_dev, is_causal=False, scale=scale)
    std_cpu = ttnn.to_torch(std_out).float()
    print(
        f"  Standard SDPA cosine: {torch.nn.functional.cosine_similarity(std_cpu.flatten().unsqueeze(0), ref_out.flatten().unsqueeze(0)).item():.6f}"
    )
    ttnn.deallocate(std_out)

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

    print("  Calling turbo_quant_sdpa_decode...")
    out_dev = ttnn.experimental.turbo_quant_sdpa_decode(
        q_dev,
        k_dev,
        k_norms_dev,
        v_dev,
        v_norms_dev,
        page_table_dev,
        cur_pos_dev,
        centroids,
        scale,
    )
    out_cpu = ttnn.to_torch(out_dev).float()

    for t in [q_dev, k_dev, v_dev, k_norms_dev, v_norms_dev, page_table_dev, cur_pos_dev, out_dev]:
        ttnn.deallocate(t)

    print(f"  TQ range: [{out_cpu.min():.4f}, {out_cpu.max():.4f}]")
    cos_ref = torch.nn.functional.cosine_similarity(
        out_cpu.flatten().unsqueeze(0), ref_out.flatten().unsqueeze(0)
    ).item()
    cos_std = torch.nn.functional.cosine_similarity(
        out_cpu.flatten().unsqueeze(0), std_cpu.flatten().unsqueeze(0)
    ).item()
    print(f"  TQ vs ref: {cos_ref:.6f}")
    print(f"  TQ vs std: {cos_std:.6f}")

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
        test_bf16_sdpa(device, nqh=1, nkh=1)
        test_bf16_sdpa(device, nqh=8, nkh=8)
        test_bf16_sdpa(device, nqh=32, nkh=8)
    finally:
        ttnn.close_device(device)
