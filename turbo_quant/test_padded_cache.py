#!/usr/bin/env python3
"""Reproduce e2e padded-cache scenario in a unit test.

The e2e flow uses max_num_blocks=1024 even for seq=128, leaving 1020 unfilled
blocks of zero data. The standard test_paged_fused.py uses blocks_per_batch+4
padding (only ~1 extra block), which doesn't exercise the partial-fill case.

This test allocates a 1024-block cache, writes only the first N blocks with
real data (rest stay zero), runs SDPA with cur_pos = real_seq_len-1 so the
loop should be capped at ceil((real_seq+1)/128) chunks.

If the cur_pos cap is correct, output should match the torch reference computed
on just the real data, since the kernel should ignore unfilled positions
beyond cur_pos within the iterated chunks.
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from turbo_quant.quantizer import TurboQuantMSE


def reference_sdpa(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def run(device, real_seq_len, max_num_blocks, head_dim=128, nqh=8, nkh=8, bits=3):
    """Run with `real_seq_len` real positions, padded to `max_num_blocks` total blocks."""
    B = 1
    scale = head_dim**-0.5
    block_size = 32
    real_blocks = (real_seq_len + block_size - 1) // block_size
    real_seq_padded = real_blocks * block_size

    assert real_blocks <= max_num_blocks, "real must fit in padded"

    torch.manual_seed(42)
    quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    centroids = quantizer.codebook.centroids.tolist()

    # Generate real K/V padded out to whole blocks (fill end with zeros).
    k_real = torch.zeros(B, nkh, real_seq_padded, head_dim)
    v_real = torch.zeros(B, nkh, real_seq_padded, head_dim)
    k_real[:, :, :real_seq_len] = torch.randn(B, nkh, real_seq_len, head_dim)
    v_real[:, :, :real_seq_len] = torch.randn(B, nkh, real_seq_len, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)

    # Quantize real K/V (no rotation — same as test_paged_fused.py).
    k_idx, k_n = quantizer.quantize(k_real)
    v_idx, v_n = quantizer.quantize(v_real)

    # Pack into paged layout: [max_num_blocks, nkh, block_size, D]. First
    # real_blocks pages get real data; rest stay zero.
    def pack_paged_idx(t, D):
        # t: [B, nkh, real_seq_padded, D] -> [max_num_blocks, nkh, block_size, D]
        t = t.reshape(B, nkh, real_blocks, block_size, D)
        t = t.permute(0, 2, 1, 3, 4).reshape(B * real_blocks, nkh, block_size, D)
        out = torch.zeros(max_num_blocks, nkh, block_size, D)
        out[:real_blocks] = t
        return out

    def pack_paged_norms(t):
        # t: [B, nkh, real_seq_padded, 1] -> [max_num_blocks, nkh, block_size, 32]
        t = t.reshape(B, nkh, real_blocks, block_size, 1)
        t = t.permute(0, 2, 1, 3, 4).reshape(B * real_blocks, nkh, block_size, 1)
        out_raw = torch.zeros(max_num_blocks, nkh, block_size, 1)
        out_raw[:real_blocks] = t
        out = torch.zeros(max_num_blocks, nkh, block_size, 32)
        out[..., 0] = out_raw.squeeze(-1)
        return out

    k_idx_paged = pack_paged_idx(k_idx.float(), head_dim)
    v_idx_paged = pack_paged_idx(v_idx.float(), head_dim)
    k_n_paged = pack_paged_norms(k_n)
    v_n_paged = pack_paged_norms(v_n)

    # Upload to device.
    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ki = ttnn.from_torch(k_idx_paged, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    vi = ttnn.from_torch(v_idx_paged, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    kn = ttnn.from_torch(k_n_paged, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vn = ttnn.from_torch(v_n_paged, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Identity page_table over the full max_num_blocks (so virtual block i -> physical i).
    page_table_cpu = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B, max_num_blocks)
    page_table = ttnn.from_torch(page_table_cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # cur_pos = real_seq_len - 1 (attend to positions 0..real_seq_len-1).
    cur_pos = ttnn.from_torch(
        torch.tensor([real_seq_len - 1], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Reference: attention over real positions only (k_real, v_real have
    # zeros beyond real_seq_len, so dequantizing them and slicing to real
    # gives the proper reference).
    k_c = quantizer.codebook.dequantize(k_idx[:, :, :real_seq_len].long())
    v_c = quantizer.codebook.dequantize(v_idx[:, :, :real_seq_len].long())
    k_d = k_c * k_n[:, :, :real_seq_len]
    v_d = v_c * v_n[:, :, :real_seq_len]
    hpk = nqh // nkh
    ref = reference_sdpa(
        q_raw,
        k_d.repeat_interleave(hpk, 1) if hpk > 1 else k_d,
        v_d.repeat_interleave(hpk, 1) if hpk > 1 else v_d,
        scale,
    )

    out = ttnn.experimental.turbo_quant_sdpa_decode(
        q_dev, ki, kn, vi, vn, page_table, cur_pos, centroids, scale, pre_rescaled=False
    )
    o = ttnn.to_torch(out).float()
    cos = torch.nn.functional.cosine_similarity(o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    expected_chunks = (real_seq_len + 127) // 128
    print(
        f"real_seq={real_seq_len:>5}, max_blocks={max_num_blocks:>5}, "
        f"k_num_chunks={max_num_blocks // 4:>4}, valid_k_chunks={expected_chunks:>3}: "
        f"cos={cos:.6f}  {'PASS' if cos > 0.95 else 'FAIL'}"
    )

    for t in [q_dev, ki, vi, kn, vn, page_table, cur_pos, out]:
        ttnn.deallocate(t)

    return cos > 0.95


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        ok = True
        # Same total blocks but varying real fill. Tests cap behavior with padding.
        print("--- max_num_blocks = 1024 (matches eval_e2e.py default) ---")
        for real in [42, 128, 256]:
            try:
                ok &= run(device, real, max_num_blocks=1024)
            except Exception as e:
                print(f"real={real}: ERROR {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                ok = False
        print("\n--- small padding (matches existing unit test) ---")
        for real in [128, 256, 512]:
            real_blocks = (real + 31) // 32
            try:
                ok &= run(device, real, max_num_blocks=real_blocks + 4)
            except Exception as e:
                print(f"real={real}: ERROR {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                ok = False
        print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}")
    finally:
        ttnn.close_device(device)
