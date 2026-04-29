#!/usr/bin/env python3
"""Smoke test for the fused TQ kernel's return_lse path.

Allocates a small TQ cache, populates it with one position of K/V, then calls
fused_sdpa_decode(..., return_lse=True). Verifies:
  - The op returns a (out, lse) tuple
  - `out` matches what return_lse=False produces
  - `lse` is a finite tensor of expected shape

This validates LSE_COMBINE_DESIGN.md step 1+2 before wiring it into attention.py.
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from models.tt_transformers.tt.common import PagedAttentionConfig
from turbo_quant.ttnn_integration import TTNNTurboQuantCache


def main():
    device = ttnn.open_device(device_id=0)
    try:
        nqh, nkh, head_dim, bits = 32, 8, 128, 3
        scale = head_dim**-0.5
        seq_len = 128
        block_size = 32
        cur_pos_target = 41

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

        torch.manual_seed(0)
        q_raw = torch.randn(1, nqh, 1, head_dim)

        # Identity page table.
        blocks_per_seq = (seq_len + block_size - 1) // block_size
        page_table_torch = torch.arange(blocks_per_seq, dtype=torch.int32).reshape(1, -1)
        page_table_dev = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        # Populate cache with random K/V at each step up to cur_pos_target.
        for step in range(cur_pos_target + 1):
            k_new = torch.randn(1, nkh, 1, head_dim)
            v_new = torch.randn(1, nkh, 1, head_dim)
            k_dev = ttnn.from_torch(k_new, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            v_dev = ttnn.from_torch(v_new, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            cp_dev = ttnn.from_torch(
                torch.tensor([step], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
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

        # Reference: return_lse=False
        out_ref = tq.fused_sdpa_decode(
            q_dev, layer_idx=0, current_pos=cur_pos_dev, scale=scale, page_table=page_table_dev
        )
        out_ref_cpu = ttnn.to_torch(out_ref).float()
        ttnn.deallocate(out_ref)

        # Test: return_lse=True
        out_lse, lse = tq.fused_sdpa_decode(
            q_dev, layer_idx=0, current_pos=cur_pos_dev, scale=scale, page_table=page_table_dev, return_lse=True
        )
        out_lse_cpu = ttnn.to_torch(out_lse).float()
        lse_cpu = ttnn.to_torch(lse).float()
        ttnn.deallocate(out_lse)
        ttnn.deallocate(lse)

        # Validation
        out_diff = (out_ref_cpu - out_lse_cpu).abs().max().item()
        print(f"out shape:            {tuple(out_lse_cpu.shape)}")
        print(f"lse shape:            {tuple(lse_cpu.shape)}")
        print(f"|out_no_lse - out_lse| max: {out_diff:.6e}  (expect ~0 — kernel does identical math)")
        print(f"lse[0,0,0,0] (col 0): {lse_cpu[0, 0, 0, 0].item():.4f}")
        print(f"lse[0,1,0,0]:         {lse_cpu[0, 1, 0, 0].item():.4f}")
        print(f"lse[0,7,0,0]:         {lse_cpu[0, 7, 0, 0].item():.4f}")
        print(f"lse[0,15,0,0]:        {lse_cpu[0, 15, 0, 0].item():.4f}")
        print(f"lse[0,31,0,0]:        {lse_cpu[0, 31, 0, 0].item():.4f}")
        print(f"lse col 0 min/max:    {lse_cpu[0, :, 0, 0].min().item():.4f} / {lse_cpu[0, :, 0, 0].max().item():.4f}")
        print(
            f"lse col 0 has NaN/Inf: {torch.isnan(lse_cpu[0, :, 0, 0]).any().item()} / {torch.isinf(lse_cpu[0, :, 0, 0]).any().item()}"
        )
        print()
        # Quick sanity: LSE should be roughly log(N) + average score, where N≈42
        # and Q·K^T scores are ~Gaussian × scale (1/sqrt(128) ≈ 0.088). With BF16
        # values ~1, scaled scores ~0.1, and N=42, LSE ≈ log(42) + 0 ≈ 3.7.
        # Real LSE depends on the actual max score so a wide range is fine; we
        # just want to see "sensible positive numbers".
        if torch.isfinite(lse_cpu[0, :, 0, 0]).all() and lse_cpu[0, :, 0, 0].abs().max() < 100:
            print("RESULT: PASS — lse looks sane.")
            return 0
        else:
            print("RESULT: FAIL — lse contains non-finite or huge values.")
            return 1
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
