#!/usr/bin/env python3
"""Synthetic validation: BFP4 K/V fed to standard SDPA decode at various seqlens.

Tests whether the standard scaled_dot_product_attention_decode kernel can accept
BFP4 K/V inputs and produce correct results at sequence lengths from 128 to 128K.

Usage:
    PYTHONPATH=/localdev/mtairum/tt-metal python turbo_quant/test_bfp4_paged_sdpa.py
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from turbo_quant.quantizer import TurboQuantMSE


def reference_sdpa(q, k, v, scale):
    """CPU reference SDPA."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def test_bfp4_sdpa_decode(device, seq_len, head_dim=128, nqh=8, nkh=8, bits=3):
    """Feed BFP4 K/V to standard SDPA decode and check correctness."""
    B = 1
    scale = head_dim**-0.5
    seq_padded = ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)

    # Quantize with TurboQuant, then compute centroid x norm (pre-rescaled values)
    quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    k_raw = torch.randn(B, nkh, seq_padded, head_dim)
    v_raw = torch.randn(B, nkh, seq_padded, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)

    k_idx, k_norms = quantizer.quantize(k_raw)
    v_idx, v_norms = quantizer.quantize(v_raw)
    k_rescaled = quantizer.codebook.dequantize(k_idx.long()) * k_norms  # centroid x norm
    v_rescaled = quantizer.codebook.dequantize(v_idx.long()) * v_norms

    # Push as BFP4 to device
    k_bfp4 = ttnn.from_torch(k_rescaled, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_bfp4 = ttnn.from_torch(v_rescaled, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Round-trip BFP4 to get the actual values the SDPA will see (for fair CPU reference)
    k_rt = ttnn.to_torch(k_bfp4).float()
    v_rt = ttnn.to_torch(v_bfp4).float()

    # CPU reference using round-tripped BFP4 values
    hpk = nqh // nkh
    k_exp = k_rt.repeat_interleave(hpk, dim=1) if hpk > 1 else k_rt
    v_exp = v_rt.repeat_interleave(hpk, dim=1) if hpk > 1 else v_rt
    ref_out = reference_sdpa(q_raw, k_exp, v_exp, scale)

    # Device SDPA decode
    q_dev = ttnn.from_torch(
        q_raw.permute(2, 0, 1, 3),  # [1, B, NQH, DH] for decode layout
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    cur_pos = ttnn.from_torch(
        torch.tensor([seq_len - 1], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    t0 = time.perf_counter()
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        q_dev, k_bfp4, v_bfp4, cur_pos_tensor=cur_pos, scale=scale
    )
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_cpu = ttnn.to_torch(out).float()
    # Output is [1, B, NQH, DH], permute back to [B, NQH, 1, DH] for comparison
    out_cpu = out_cpu.permute(1, 0, 2, 3).unsqueeze(2)

    # Cleanup
    for t in [q_dev, k_bfp4, v_bfp4, cur_pos, out]:
        ttnn.deallocate(t)

    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(out_cpu.flatten().unsqueeze(0), ref_out.flatten().unsqueeze(0)).item()

    return cos, elapsed_ms


def main():
    device = ttnn.open_device(device_id=0)

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    nqh, nkh, hd, bits = 8, 8, 128, 3

    print(f"\n{'='*80}")
    print(f"  BFP4 K/V + Standard SDPA Decode — {nqh}Q/{nkh}KV heads, hd={hd}, bits={bits}")
    print(f"{'='*80}")
    print(f"\n{'Seq Len':>10} | {'Cosine':>10} | {'Latency':>10} | {'Status':>10} | {'BFP4 KV MB':>10}")
    print("-" * 65)

    all_pass = True
    for seq_len in seq_lens:
        kv_mb = 2 * nkh * ((seq_len + 31) // 32 * 32) * hd * 0.5 / (1024 * 1024)
        try:
            cos, ms = test_bfp4_sdpa_decode(device, seq_len, hd, nqh, nkh, bits)
            passed = cos > 0.99
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"{seq_len:>10} | {cos:>10.4f} | {ms:>8.2f}ms | {status:>10} | {kv_mb:>10.1f}")
        except Exception as e:
            all_pass = False
            err = str(e)[:50]
            print(f"{seq_len:>10} | {'—':>10} | {'—':>10} | {'ERROR':>10} | {kv_mb:>10.1f}  {err}")

    print(f"\n{'='*80}")
    print(f"  Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*80}")

    ttnn.close_device(device)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
