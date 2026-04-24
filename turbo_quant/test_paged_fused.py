#!/usr/bin/env python3
"""Test paged TQ cache + fused SDPA decode at multiple seqlens.

Verifies Step 2 (paged indices/norms) produces same cos (~0.999) as the
contiguous path from Step 1.

Uses identity page_table (virtual block i → physical block i) so the paged
layout is semantically equivalent to contiguous — lets us validate the
paged reader plumbing without running the full decode-loop quantize flow.
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


def run(device, seq_len, head_dim=128, nqh=8, nkh=8, bits=3, shuffle_pages=False):
    B = 1
    scale = head_dim**-0.5
    seq_p = ((seq_len + 31) // 32) * 32
    block_size = 32
    blocks_per_batch = seq_p // block_size
    max_num_blocks = max(blocks_per_batch + 4, 16)
    torch.manual_seed(42)

    quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    centroids = quantizer.codebook.centroids.tolist()

    # Generate K/V data, quantize.
    k_raw = torch.randn(B, nkh, seq_p, head_dim)
    v_raw = torch.randn(B, nkh, seq_p, head_dim)
    q_raw = torch.randn(B, nqh, 1, head_dim)

    k_idx, k_n = quantizer.quantize(k_raw)
    v_idx, v_n = quantizer.quantize(v_raw)

    # Reshape contiguous [B, nkh, seq_p, head_dim] into paged
    # [max_num_blocks, nkh, block_size, head_dim] via identity page_table.
    # Identity mapping: virtual_block=b*blocks_per_batch + i → physical_block
    # We lay out so physical_block i holds sequence tokens [i*block_size : (i+1)*block_size].
    # For batch=1, physical layout matches logical layout 1:1.

    # Optionally shuffle pages: virtual block i → physical block perm[i].
    # We write data at physical positions but read via page_table, so the test
    # verifies the virtual→physical translation actually works.
    if shuffle_pages:
        g = torch.Generator()
        g.manual_seed(7)
        perm = torch.randperm(max_num_blocks, generator=g)
        # Identify which virtual blocks [0..blocks_per_batch-1] are used.
        # perm[i] = physical block that holds virtual block i's data.
        virt_to_phys = perm[:blocks_per_batch]
    else:
        virt_to_phys = torch.arange(blocks_per_batch, dtype=torch.int64)

    def pack_paged_idx(t, head_dim):
        """t: [B, nkh, seq_p, head_dim] → [max_num_blocks, nkh, block_size, head_dim]"""
        # Reshape seq_p into (blocks_per_batch, block_size)
        t = t.reshape(B, nkh, blocks_per_batch, block_size, head_dim)
        # [B, nkh, blocks, block_size, D] → [blocks, nkh, block_size, D] (assume B=1)
        t = t.permute(0, 2, 1, 3, 4).reshape(B * blocks_per_batch, nkh, block_size, head_dim)
        out = torch.zeros(max_num_blocks, nkh, block_size, head_dim)
        # Scatter into physical positions per the virt→phys mapping.
        for v, p in enumerate(virt_to_phys.tolist()):
            out[p] = t[v]
        return out

    def pack_paged_norms(t):
        """t: [B, nkh, seq_p, 1] → [max_num_blocks, nkh, block_size, 1]"""
        t = t.reshape(B, nkh, blocks_per_batch, block_size, 1)
        t = t.permute(0, 2, 1, 3, 4).reshape(B * blocks_per_batch, nkh, block_size, 1)
        out = torch.zeros(max_num_blocks, nkh, block_size, 1)
        for v, p in enumerate(virt_to_phys.tolist()):
            out[p] = t[v]
        return out

    k_idx_paged = pack_paged_idx(k_idx.float(), head_dim)
    v_idx_paged = pack_paged_idx(v_idx.float(), head_dim)
    # For norms, broadcast last dim from 1 → 32 (so BF16 tile has the norm in column 0).
    k_n_paged_raw = pack_paged_norms(k_n)
    v_n_paged_raw = pack_paged_norms(v_n)
    k_n_paged = torch.zeros(max_num_blocks, nkh, block_size, 32)
    v_n_paged = torch.zeros(max_num_blocks, nkh, block_size, 32)
    k_n_paged[..., 0] = k_n_paged_raw.squeeze(-1)
    v_n_paged[..., 0] = v_n_paged_raw.squeeze(-1)

    # Upload paged tensors.
    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ki = ttnn.from_torch(k_idx_paged, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    vi = ttnn.from_torch(v_idx_paged, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    kn = ttnn.from_torch(k_n_paged, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vn = ttnn.from_torch(v_n_paged, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Page table maps virtual→physical block for this batch.
    page_table_cpu = virt_to_phys.to(torch.int32).reshape(B, blocks_per_batch)
    page_table = ttnn.from_torch(page_table_cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cur_pos = ttnn.from_torch(
        torch.tensor([seq_len - 1], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Reference on host.
    k_c = quantizer.codebook.dequantize(k_idx.long())
    v_c = quantizer.codebook.dequantize(v_idx.long())
    k_d, v_d = k_c * k_n, v_c * v_n
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
    tag = "shuffled" if shuffle_pages else "identity"
    print(f"seq={seq_len:>5} (blocks={blocks_per_batch:3d}, {tag}): cos={cos:.6f}  {'PASS' if cos > 0.95 else 'FAIL'}")

    for t in [q_dev, ki, vi, kn, vn, page_table, cur_pos, out]:
        ttnn.deallocate(t)

    return cos > 0.95


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        ok = True
        print("--- identity page_table ---")
        for seq in [128, 256, 512, 1024, 2048]:
            try:
                ok &= run(device, seq)
            except Exception as e:
                print(f"seq={seq}: ERROR {type(e).__name__}: {e}")
                ok = False
        print("\n--- shuffled page_table ---")
        for seq in [128, 256, 512, 1024, 2048]:
            try:
                ok &= run(device, seq, shuffle_pages=True)
            except Exception as e:
                print(f"seq={seq}: ERROR {type(e).__name__}: {e}")
                ok = False
        print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}")
    finally:
        ttnn.close_device(device)
