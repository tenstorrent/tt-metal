#!/usr/bin/env python3
"""Repro for K-split bug with low NQH (matching T3K per-chip work).

T3K with Llama-3.1-8B: NQH=4 per chip, total_work=4. K=8 fails (0% top-1
in eval). N150 at NQH=32 has total_work=32, K is clamped to ≤2, never
exposes the bug. This test runs single-chip with NQH=4 and varies K to
isolate whether the bug is multi-device-specific or low-total_work-specific.
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from models.tt_transformers.tt.common import PagedAttentionConfig
from turbo_quant.codebook import get_codebook
from turbo_quant.ttnn_integration import TTNNTurboQuantCache


def main():
    import os

    device = ttnn.open_device(device_id=0)
    nqh = int(os.environ.get("NQH", 4))
    nkh = int(os.environ.get("NKH", 1))
    head_dim, bits = 128, 3
    scale = head_dim**-0.5
    seq_len = 1024
    block_size = 32
    cur_pos_target = 511

    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=seq_len // block_size)
    tq = TTNNTurboQuantCache(
        device,
        num_layers=1,
        num_kv_heads=nkh,
        head_dim=head_dim,
        max_seq_len=seq_len,
        bits=bits,
        memory_efficient=True,
        paged_config=paged_cfg,
        max_batch_size=1,
    )

    torch.manual_seed(42)
    q_raw = torch.randn(1, nqh, 1, head_dim)

    blocks_per_seq = (seq_len + block_size - 1) // block_size
    page_table_torch = torch.arange(blocks_per_seq, dtype=torch.int32).reshape(1, -1)
    page_table_dev = ttnn.from_torch(page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    for step in range(cur_pos_target + 1):
        k_new = torch.randn(1, nkh, 1, head_dim)
        v_new = torch.randn(1, nkh, 1, head_dim)
        k_dev = ttnn.from_torch(k_new, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_dev = ttnn.from_torch(v_new, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cp_dev = ttnn.from_torch(
            torch.tensor([step], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        tq.update_cache(k_dev, v_dev, layer_idx=0, current_pos=cp_dev, page_table=page_table_dev)
        ttnn.deallocate(k_dev)
        ttnn.deallocate(v_dev)
        ttnn.deallocate(cp_dev)

    q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cur_pos_dev = ttnn.from_torch(
        torch.tensor([cur_pos_target], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    centroids = get_codebook(head_dim, bits, device="cpu", dtype=torch.float32).centroids.tolist()

    outs = {}
    for K in [1, 2, 4, 8]:
        outs_list = ttnn.experimental.turbo_quant_sdpa_decode(
            q_dev,
            tq.k_indices_dev[0],
            tq.k_norms_dev[0],
            tq.v_indices_dev[0],
            tq.v_norms_dev[0],
            page_table_dev,
            cur_pos_dev,
            centroids,
            scale,
            False,
            K,
            False,  # return_lse
        )
        out = outs_list[0]
        outs[K] = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        flat = outs[K].flatten()
        print(f"K={K:2d}: out_max={flat.abs().max().item():.4f}  first5={[round(v.item(), 4) for v in flat[:5]]}")

    print()
    for K in [2, 4, 8]:
        diff = (outs[1] - outs[K]).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            outs[1].flatten().unsqueeze(0), outs[K].flatten().unsqueeze(0)
        ).item()
        status = "PASS" if diff < 1e-3 else "FAIL"
        print(f"K=1 vs K={K}: max|diff|={diff:.6e}  cos={cos:.6f}  [{status}]")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
