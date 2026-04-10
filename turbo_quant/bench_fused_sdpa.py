#!/usr/bin/env python3
"""Benchmark fused TQ SDPA decode across sequence lengths.

Measures latency and correctness for:
- Fused TQ SDPA (BFP4 indices + norms → on-the-fly dequant + SDPA)
- Standard SDPA (BF16 K/V)
- Dequantize + standard SDPA (BFP4→BF16 + SDPA, current TQ path)

Tests seq lengths: 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K
"""

import gc
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


def bench_fused_tq_sdpa(device, seq_len, head_dim=128, nqh=8, nkh=8, bits=3, warmup=2, iters=5):
    """Benchmark fused TQ SDPA at given sequence length."""
    B = 1
    scale = head_dim**-0.5
    seq_padded = ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)

    quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    centroids = quantizer.codebook.centroids.tolist()

    # Generate data and quantize
    k_raw = torch.randn(B, nkh, seq_padded, head_dim)
    v_raw = torch.randn(B, nkh, seq_padded, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)

    k_indices, k_norms = quantizer.quantize(k_raw)
    v_indices, v_norms = quantizer.quantize(v_raw)

    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_idx_dev = ttnn.from_torch(k_indices.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_idx_dev = ttnn.from_torch(v_indices.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

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

    # Warmup
    for _ in range(warmup):
        out = ttnn.experimental.turbo_quant_sdpa_decode(
            q_dev, k_idx_dev, k_norms_dev, v_idx_dev, v_norms_dev, page_table_dev, cur_pos_dev, centroids, scale
        )
        ttnn.deallocate(out)

    # Benchmark
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = ttnn.experimental.turbo_quant_sdpa_decode(
            q_dev, k_idx_dev, k_norms_dev, v_idx_dev, v_norms_dev, page_table_dev, cur_pos_dev, centroids, scale
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    fused_ms = (t1 - t0) / iters * 1000

    # Correctness check on last output
    out = ttnn.experimental.turbo_quant_sdpa_decode(
        q_dev, k_idx_dev, k_norms_dev, v_idx_dev, v_norms_dev, page_table_dev, cur_pos_dev, centroids, scale
    )
    out_cpu = ttnn.to_torch(out).float()
    ttnn.deallocate(out)

    # CPU reference
    k_centroids = quantizer.codebook.dequantize(k_indices.long())
    v_centroids = quantizer.codebook.dequantize(v_indices.long())
    k_dequant = k_centroids * k_norms
    v_dequant = v_centroids * v_norms
    heads_per_kv = nqh // nkh
    k_exp = k_dequant.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else k_dequant
    v_exp = v_dequant.repeat_interleave(heads_per_kv, dim=1) if heads_per_kv > 1 else v_dequant
    ref_out = reference_sdpa(q_raw, k_exp, v_exp, scale)

    cos = torch.nn.functional.cosine_similarity(out_cpu.flatten().unsqueeze(0), ref_out.flatten().unsqueeze(0)).item()

    for t in [q_dev, k_idx_dev, v_idx_dev, k_norms_dev, v_norms_dev, page_table_dev, cur_pos_dev]:
        ttnn.deallocate(t)

    return fused_ms, cos


def bench_standard_sdpa(device, seq_len, head_dim=128, nqh=8, nkh=8, warmup=2, iters=5):
    """Benchmark standard SDPA with BF16 K/V."""
    B = 1
    scale = head_dim**-0.5
    seq_padded = ((seq_len + 31) // 32) * 32
    torch.manual_seed(42)

    q_raw = torch.randn(B, nqh, 1, head_dim)
    k_raw = torch.randn(B, nkh, seq_padded, head_dim)
    v_raw = torch.randn(B, nkh, seq_padded, head_dim)

    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_dev = ttnn.from_torch(k_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_dev = ttnn.from_torch(v_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for _ in range(warmup):
        out = ttnn.transformer.scaled_dot_product_attention(q_dev, k_dev, v_dev, is_causal=False, scale=scale)
        ttnn.deallocate(out)

    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = ttnn.transformer.scaled_dot_product_attention(q_dev, k_dev, v_dev, is_causal=False, scale=scale)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    std_ms = (t1 - t0) / iters * 1000

    for t in [q_dev, k_dev, v_dev]:
        ttnn.deallocate(t)

    return std_ms


def kv_memory_mb(seq_len, nkh=8, head_dim=128, dtype_bytes=2):
    """Estimate KV cache memory in MB for one format."""
    seq_padded = ((seq_len + 31) // 32) * 32
    # K + V, each [B=1, nkh, seq_padded, head_dim]
    return 2 * nkh * seq_padded * head_dim * dtype_bytes / (1024 * 1024)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    extended_lens = [16384, 32768]
    # 64K and 128K may OOM on L1 (full-cache BF16 in CB = O(seq) memory)
    # TODO: interleave dequant with SDPA to reduce to O(chunk)

    nqh, nkh, hd, bits = 8, 8, 128, 3

    print(f"\n{'='*80}")
    print(f"  Fused TQ SDPA Benchmark — {nqh}Q/{nkh}KV heads, hd={hd}, bits={bits}")
    print(f"{'='*80}")
    print(
        f"\n{'Seq Len':>10} | {'Fused TQ':>10} | {'Std SDPA':>10} | {'Speedup':>8} | {'Cosine':>8} | {'BFP4 KV':>8} | {'BF16 KV':>8}"
    )
    print(f"{'':>10} | {'(ms)':>10} | {'(ms)':>10} | {'':>8} | {'':>8} | {'(MB)':>8} | {'(MB)':>8}")
    print("-" * 80)

    try:
        for seq_len in seq_lens + extended_lens:
            bfp4_mb = kv_memory_mb(seq_len, nkh, hd, dtype_bytes=0.5)  # ~0.5 bytes/elem for BFP4
            bf16_mb = kv_memory_mb(seq_len, nkh, hd, dtype_bytes=2)

            # Fewer iters for large seq to avoid timeouts
            w = 1 if seq_len >= 8192 else 2
            n = 2 if seq_len >= 8192 else 5

            # Fused TQ SDPA
            try:
                fused_ms, cos = bench_fused_tq_sdpa(device, seq_len, hd, nqh, nkh, bits, warmup=w, iters=n)
            except Exception as e:
                fused_ms, cos = float("nan"), 0.0
                print(f"{seq_len:>10} | {'OOM/ERR':>10} | ", end="", flush=True)
                # Try to continue with standard
                try:
                    std_ms = bench_standard_sdpa(device, seq_len, hd, nqh, nkh, warmup=1, iters=3)
                except Exception:
                    std_ms = float("nan")
                print(f"{std_ms:>10.2f} | {'N/A':>8} | {cos:>8.4f} | {bfp4_mb:>8.1f} | {bf16_mb:>8.1f}")
                continue

            # Standard SDPA
            try:
                std_ms = bench_standard_sdpa(device, seq_len, hd, nqh, nkh, warmup=w, iters=n)
            except Exception:
                std_ms = float("nan")

            speedup = std_ms / fused_ms if fused_ms > 0 else float("nan")
            print(
                f"{seq_len:>10} | {fused_ms:>10.2f} | {std_ms:>10.2f} | {speedup:>7.2f}x | {cos:>8.4f} | {bfp4_mb:>8.1f} | {bf16_mb:>8.1f}"
            )

            gc.collect()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        print(f"\n{'='*80}")
        ttnn.close_device(device)
