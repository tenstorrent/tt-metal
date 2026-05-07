#!/usr/bin/env python3
"""Smoke test for the hybrid_sdpa_decode fast path with ring_sdpa_pos/old_pos.

Bypasses the model harness — instantiates a small TQ cache directly, populates
it with a few positions, and calls hybrid_sdpa_decode with ring_sdpa_pos and
old_pos as device tensors. Verifies it doesn't hang and produces a finite output.
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
        seq_len = 256
        block_size = 32
        recent_window = 64
        cur_pos_target = 100  # > W → tests both halves

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
            recent_window=recent_window,
        )
        tq.rotation_absorbed = True  # match eval harness

        torch.manual_seed(0)

        # Identity page table (TQ cache).
        blocks_per_seq = (seq_len + block_size - 1) // block_size
        page_table_torch = torch.arange(blocks_per_seq, dtype=torch.int32).reshape(1, -1)
        page_table_dev = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        # Populate cache with random K/V at each step up to cur_pos_target.
        print(f"Populating cache for {cur_pos_target + 1} positions...")
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
            ring_write_dev = ttnn.from_torch(
                torch.tensor([step % tq.ring_W_padded], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            tq.update_cache(
                k_dev,
                v_dev,
                layer_idx=0,
                current_pos=cp_dev,
                page_table=page_table_dev,
                ring_write_pos=ring_write_dev,
            )
            ttnn.deallocate(k_dev)
            ttnn.deallocate(v_dev)
            ttnn.deallocate(cp_dev)
            ttnn.deallocate(ring_write_dev)
        print("Cache populated.")

        q_raw = torch.randn(1, nqh, 1, head_dim)
        q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cur_pos_dev = ttnn.from_torch(
            torch.tensor([cur_pos_target], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Pre-rotate Q (hybrid_sdpa_decode expects pre-rotated Q).
        q_rot = tq.pre_rotate_query(q_dev)
        ttnn.deallocate(q_dev)

        # Now build ring_sdpa_pos and old_pos as device tensors and call fast path.
        W = recent_window
        W_padded = tq.ring_W_padded
        ring_sdpa_pos_int = min(W - 1, cur_pos_target)  # = 63
        old_pos_int = max(0, cur_pos_target - W)  # = 36

        print(f"\nFast path call: cur_pos={cur_pos_target}, ring_sdpa_pos={ring_sdpa_pos_int}, old_pos={old_pos_int}")

        ring_sdpa_pos_dev = ttnn.from_torch(
            torch.tensor([ring_sdpa_pos_int], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        old_pos_dev = ttnn.from_torch(
            torch.tensor([old_pos_int], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # pre_rotate_query is a plain matmul (Q @ rotation) and preserves Q's
        # original shape — q_raw was constructed as [B, NQH, 1, DH], so q_rot
        # is already in the layout hybrid_sdpa_decode expects. (Earlier
        # versions of pre_rotate_query may have returned [1, B, NQH, DH] and
        # required the permute below; that's no longer the case.)
        q_bqhd = q_rot

        out = tq.hybrid_sdpa_decode(
            q_bqhd,
            layer_idx=0,
            current_pos=cur_pos_dev,
            scale=scale,
            page_table=page_table_dev,
            ring_sdpa_pos=ring_sdpa_pos_dev,
            old_pos=old_pos_dev,
        )
        out_cpu = ttnn.to_torch(out).float()
        print(f"Fast path output shape: {tuple(out_cpu.shape)}")
        print(f"Output finite: {torch.isfinite(out_cpu).all().item()}")
        print(f"Output abs max: {out_cpu.abs().max().item():.4f}")

        if torch.isfinite(out_cpu).all() and out_cpu.abs().max() < 100:
            print("\nRESULT: PASS — fast path returns finite, sane output.")
            return 0
        else:
            print("\nRESULT: FAIL — output non-finite or huge.")
            return 1
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
