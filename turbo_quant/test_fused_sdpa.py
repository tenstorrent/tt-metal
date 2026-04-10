#!/usr/bin/env python3
"""Test fused TQ SDPA: full dequant mode + pre-rescaled mode."""
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from turbo_quant.quantizer import TurboQuantMSE


def reference_sdpa(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def test_full_dequant(device, seq_len=128, head_dim=128, nqh=8, nkh=8, bits=3):
    """Full dequant: BFP4 indices + norms → centroid gather × norm → SDPA."""
    print(f"\n=== Full dequant (seq={seq_len}, heads={nqh}/{nkh}) ===")
    B, scale, seq_p = 1, head_dim**-0.5, ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)
    q = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    centroids = q.codebook.centroids.tolist()
    k_raw, v_raw = torch.randn(B, nkh, seq_p, head_dim), torch.randn(B, nkh, seq_p, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)
    k_idx, k_n = q.quantize(k_raw)
    v_idx, v_n = q.quantize(v_raw)
    k_c, v_c = q.codebook.dequantize(k_idx.long()), q.codebook.dequantize(v_idx.long())
    k_d, v_d = k_c * k_n, v_c * v_n
    hpk = nqh // nkh
    ref = reference_sdpa(
        q_raw,
        k_d.repeat_interleave(hpk, 1) if hpk > 1 else k_d,
        v_d.repeat_interleave(hpk, 1) if hpk > 1 else v_d,
        scale,
    )

    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ki = ttnn.from_torch(k_idx.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    vi = ttnn.from_torch(v_idx.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    kn_p, vn_p = torch.zeros(B, nkh, seq_p, 32), torch.zeros(B, nkh, seq_p, 32)
    kn_p[..., 0], vn_p[..., 0] = k_n.squeeze(-1), v_n.squeeze(-1)
    kn = ttnn.from_torch(kn_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vn = ttnn.from_torch(vn_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pt = ttnn.from_torch(
        torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    cp = ttnn.from_torch(
        torch.tensor([seq_len - 1], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    out = ttnn.experimental.turbo_quant_sdpa_decode(q_dev, ki, kn, vi, vn, pt, cp, centroids, scale, pre_rescaled=False)
    o = ttnn.to_torch(out).float()
    for t in [q_dev, ki, vi, kn, vn, pt, cp, out]:
        ttnn.deallocate(t)
    cos = torch.nn.functional.cosine_similarity(o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"  Cosine: {cos:.6f} {'PASS' if cos > 0.95 else 'FAIL'}")
    return cos > 0.95


def test_pre_rescaled(device, seq_len=128, head_dim=128, nqh=8, nkh=8):
    """Pre-rescaled: BFP4 values are centroid×norm, typecast only → SDPA."""
    print(f"\n=== Pre-rescaled (seq={seq_len}, heads={nqh}/{nkh}) ===")
    B, scale, seq_p = 1, head_dim**-0.5, ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)
    k_raw, v_raw = torch.randn(B, nkh, seq_p, head_dim), torch.randn(B, nkh, seq_p, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)

    # Store as BFP4 (pre-rescaled: values not indices)
    k_bfp4 = ttnn.from_torch(k_raw, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_bfp4 = ttnn.from_torch(v_raw, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    k_rt, v_rt = ttnn.to_torch(k_bfp4).float(), ttnn.to_torch(v_bfp4).float()
    hpk = nqh // nkh
    ref = reference_sdpa(
        q_raw,
        k_rt.repeat_interleave(hpk, 1) if hpk > 1 else k_rt,
        v_rt.repeat_interleave(hpk, 1) if hpk > 1 else v_rt,
        scale,
    )

    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dummy_n = ttnn.from_torch(
        torch.zeros(B, nkh, seq_p, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    pt = ttnn.from_torch(
        torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    cp = ttnn.from_torch(
        torch.tensor([seq_len - 1], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Warmup + benchmark
    for _ in range(2):
        out = ttnn.experimental.turbo_quant_sdpa_decode(
            q_dev, k_bfp4, dummy_n, v_bfp4, dummy_n, pt, cp, [0.0] * 8, scale, pre_rescaled=True
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(5):
        out = ttnn.experimental.turbo_quant_sdpa_decode(
            q_dev, k_bfp4, dummy_n, v_bfp4, dummy_n, pt, cp, [0.0] * 8, scale, pre_rescaled=True
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) / 5 * 1000

    out = ttnn.experimental.turbo_quant_sdpa_decode(
        q_dev, k_bfp4, dummy_n, v_bfp4, dummy_n, pt, cp, [0.0] * 8, scale, pre_rescaled=True
    )
    o = ttnn.to_torch(out).float()
    for t in [q_dev, k_bfp4, v_bfp4, dummy_n, pt, cp, out]:
        ttnn.deallocate(t)
    cos = torch.nn.functional.cosine_similarity(o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"  Cosine: {cos:.6f}  Latency: {ms:.2f} ms {'PASS' if cos > 0.99 else 'FAIL'}")
    return cos > 0.99


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        ok = True
        ok &= test_full_dequant(device, nqh=8, nkh=8)
        ok &= test_full_dequant(device, nqh=32, nkh=8)
        ok &= test_pre_rescaled(device, nqh=8, nkh=8)
        ok &= test_pre_rescaled(device, nqh=32, nkh=8)
        ok &= test_pre_rescaled(device, seq_len=512, nqh=8, nkh=8)
        ok &= test_pre_rescaled(device, seq_len=2048, nqh=8, nkh=8)
        print(f"\n{'All tests passed!' if ok else 'Some tests FAILED'}")
    finally:
        ttnn.close_device(device)
