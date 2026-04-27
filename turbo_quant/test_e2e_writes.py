#!/usr/bin/env python3
"""Reproduce e2e write path: use paged_update_cache instead of from_torch.

The standard test_paged_fused.py uses from_torch to populate the paged cache.
The e2e flow uses paged_update_cache to write K/V one position at a time.
If those two paths produce different on-device byte representations of BFP4
indices, the fused kernel may see corrupted data in e2e but not in unit tests.

This test:
  1. Allocates a paged BFP4 cache (zero-initialized) with max_num_blocks=1024
  2. Uses paged_update_cache to write quantized K/V at each position
  3. Calls SDPA with cur_pos = real_seq_len-1
  4. Compares to torch reference computed on the real positions
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


def run(device, real_seq_len, max_num_blocks, head_dim=128, nqh=32, nkh=8, bits=3, pre_rotate_q=True):
    B = 1
    scale = head_dim**-0.5
    block_size = 32
    real_blocks = (real_seq_len + block_size - 1) // block_size
    real_seq_padded = real_blocks * block_size
    assert real_blocks <= max_num_blocks

    torch.manual_seed(42)
    quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    centroids = quantizer.codebook.centroids.tolist()

    # Reference K/V for the real positions.
    k_real = torch.randn(B, nkh, real_seq_len, head_dim)
    v_real = torch.randn(B, nkh, real_seq_len, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)
    # Pre-rotate Q if requested (mimics e2e pre_rotate_query). We rotate RAW Q
    # so that the kernel receives Q × Π. Reference must use raw Q × K_rot^T,
    # because K stored in cache is K × Π (rotated by quantize), and the
    # kernel computes (Q × Π) × (K × Π)^T = Q × K^T (in original space).
    # So reference is Q × K^T, but the kernel computes that via rotation.
    if pre_rotate_q:
        q_kernel = q_raw @ quantizer.rotation.to(q_raw.dtype)
    else:
        q_kernel = q_raw

    # Quantize each position's K/V (will write one-at-a-time via paged_update_cache).
    # quantizer.quantize returns (idx_uint8, norms) for [..., D]
    k_idx_all, k_n_all = quantizer.quantize(k_real)  # [B, nkh, seq, D], [B, nkh, seq, 1]
    v_idx_all, v_n_all = quantizer.quantize(v_real)

    # Allocate zero-init paged cache: [max_num_blocks, nkh, block_size, D] BFP4.
    zero_idx_paged = torch.zeros(max_num_blocks, nkh, block_size, head_dim, dtype=torch.bfloat16)
    zero_norms_paged = torch.zeros(max_num_blocks, nkh, block_size, 32, dtype=torch.bfloat16)

    k_indices_dev = ttnn.from_torch(zero_idx_paged, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_indices_dev = ttnn.from_torch(zero_idx_paged, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    k_norms_dev = ttnn.from_torch(zero_norms_paged, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_norms_dev = ttnn.from_torch(zero_norms_paged, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Identity page_table: virt block i -> phys block i.
    page_table_cpu = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B, max_num_blocks)
    page_table = ttnn.from_torch(page_table_cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Write each position one at a time via paged_update_cache (mimics decode loop).
    _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    _shard_spec_idx = ttnn.ShardSpec(_shard_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
    _shard_mem_idx = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_idx)
    _shard_spec_norms = ttnn.ShardSpec(_shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
    _shard_mem_norms = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_norms)

    for pos in range(real_seq_len):
        # K/V indices for this position: [B, nkh, 1, D] BF16 (integer values 0..7).
        ki_pos = k_idx_all[:, :, pos : pos + 1, :].float().to(torch.bfloat16)
        vi_pos = v_idx_all[:, :, pos : pos + 1, :].float().to(torch.bfloat16)
        # Norms: [B, nkh, 1, 1] -> broadcast to [B, nkh, 1, 32].
        kn_pos = k_n_all[:, :, pos : pos + 1, :].to(torch.bfloat16)
        vn_pos = v_n_all[:, :, pos : pos + 1, :].to(torch.bfloat16)
        kn_pos_bcast = torch.zeros(B, nkh, 1, 32, dtype=torch.bfloat16)
        vn_pos_bcast = torch.zeros(B, nkh, 1, 32, dtype=torch.bfloat16)
        kn_pos_bcast[..., 0] = kn_pos.squeeze(-1)
        vn_pos_bcast[..., 0] = vn_pos.squeeze(-1)

        # update_cache expects [seq, B, nkh, D] (the permute (2,0,1,3) in ttnn_integration).
        ki_scatter = ki_pos.permute(2, 0, 1, 3)
        vi_scatter = vi_pos.permute(2, 0, 1, 3)
        kn_scatter = kn_pos_bcast.permute(2, 0, 1, 3)
        vn_scatter = vn_pos_bcast.permute(2, 0, 1, 3)

        ki_dev = ttnn.from_torch(
            ki_scatter, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_mem_idx
        )
        vi_dev = ttnn.from_torch(
            vi_scatter, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_mem_idx
        )
        kn_dev = ttnn.from_torch(
            kn_scatter, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_mem_norms
        )
        vn_dev = ttnn.from_torch(
            vn_scatter, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_mem_norms
        )
        pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=device)

        ttnn.experimental.paged_update_cache(k_indices_dev, ki_dev, update_idxs_tensor=pos_tt, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_indices_dev, vi_dev, update_idxs_tensor=pos_tt, page_table=page_table)
        ttnn.experimental.paged_update_cache(k_norms_dev, kn_dev, update_idxs_tensor=pos_tt, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_norms_dev, vn_dev, update_idxs_tensor=pos_tt, page_table=page_table)

        for t in [ki_dev, vi_dev, kn_dev, vn_dev, pos_tt]:
            ttnn.deallocate(t)

    # Run SDPA at cur_pos = real_seq_len - 1.
    q_dev = ttnn.from_torch(q_kernel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cur_pos = ttnn.from_torch(
        torch.tensor([real_seq_len - 1], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Reference: attention over real positions only. K/V stored in cache are
    # already Π-rotated (because quantize() applies rotation). So:
    #   - Use q_kernel (= Q × Π if pre_rotate_q else Q) — matches what kernel sees.
    #   - K/V from dequantize(idx) * norm — matches dequant in kernel (Π-rotated K, V).
    # Then ref computes (q_kernel × K_rot^T) × V_rot, same as kernel.
    k_c = quantizer.codebook.dequantize(k_idx_all.long())
    v_c = quantizer.codebook.dequantize(v_idx_all.long())
    k_d = k_c * k_n_all
    v_d = v_c * v_n_all
    hpk = nqh // nkh
    ref = reference_sdpa(
        q_kernel,
        k_d.repeat_interleave(hpk, 1) if hpk > 1 else k_d,
        v_d.repeat_interleave(hpk, 1) if hpk > 1 else v_d,
        scale,
    )

    out = ttnn.experimental.turbo_quant_sdpa_decode(
        q_dev,
        k_indices_dev,
        k_norms_dev,
        v_indices_dev,
        v_norms_dev,
        page_table,
        cur_pos,
        centroids,
        scale,
        pre_rescaled=False,
    )
    o = ttnn.to_torch(out).float()
    cos = torch.nn.functional.cosine_similarity(o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    expected_chunks = (real_seq_len + 127) // 128
    print(
        f"real_seq={real_seq_len:>5}, max_blocks={max_num_blocks:>5}, "
        f"valid_k_chunks={expected_chunks:>3}: cos={cos:.6f}  {'PASS' if cos > 0.95 else 'FAIL'}"
    )

    for t in [q_dev, k_indices_dev, v_indices_dev, k_norms_dev, v_norms_dev, page_table, cur_pos, out]:
        ttnn.deallocate(t)

    return cos > 0.95


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        ok = True
        for real in [42, 128]:
            try:
                ok &= run(device, real, max_num_blocks=1024)
            except Exception as e:
                print(f"real={real}: ERROR {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                ok = False
        print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}")
    finally:
        ttnn.close_device(device)
