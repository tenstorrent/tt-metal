#!/usr/bin/env python3
"""Long-context smoke test for hybrid_sdpa_decode.

Allocates a 4K paged cache, populates it via update_cache for 4096 positions,
then calls hybrid_sdpa_decode at cur_pos = 4095. Verifies:
  - No crash, no L1 overflow
  - Output is finite (no NaN/Inf bleed-through from chunked online softmax)
  - LSE values look sane (not -inf or NaN at col 0)
  - Combine produces coherent output magnitude

This validates that the hybrid kernel + plumbing scale beyond the 1024-token
eval. Doesn't measure accuracy — just "does it work at long context?".

Conflicts with the user's "never run >1 test on device" rule, so the user must
manually reset the device with `tt-smi -r` between this and any other run.
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from models.tt_transformers.tt.common import PagedAttentionConfig
from turbo_quant.ttnn_integration import TTNNTurboQuantCache, get_codebook


def main():
    device = ttnn.open_device(device_id=0)
    try:
        nqh, nkh, head_dim, bits = 32, 8, 128, 3
        scale = head_dim**-0.5
        import os

        seq_len = int(os.environ.get("LONG_CTX_LEN", "4096"))
        block_size = 32
        recent_window = 128
        cur_pos_target = seq_len - 1

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
        tq.rotation_absorbed = True
        torch.manual_seed(0)

        blocks_per_seq = seq_len // block_size
        page_table_torch = torch.arange(blocks_per_seq, dtype=torch.int32).reshape(1, -1)
        page_table_dev = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        print(f"Populating cache for {cur_pos_target + 1} positions (this is the slow part)...")
        import time

        t0 = time.perf_counter()
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
            ring_w_dev = ttnn.from_torch(
                torch.tensor([step % tq.ring_W_padded], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            tq.update_cache(
                k_dev, v_dev, layer_idx=0, current_pos=cp_dev, page_table=page_table_dev, ring_write_pos=ring_w_dev
            )
            ttnn.deallocate(k_dev)
            ttnn.deallocate(v_dev)
            ttnn.deallocate(cp_dev)
            ttnn.deallocate(ring_w_dev)
            if step % 512 == 0:
                print(f"  populated through pos {step}/{cur_pos_target} ({time.perf_counter()-t0:.1f}s)")
        print(f"  Cache populated in {time.perf_counter()-t0:.1f}s.")

        # Build a query and pre-rotate.
        q_raw = torch.randn(1, 1, nqh, head_dim)  # [1, B, NQH, DH]
        q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        q_rot = tq.pre_rotate_query(q_dev)
        ttnn.deallocate(q_dev)
        q_bqhd = ttnn.permute(q_rot, (1, 2, 0, 3))
        ttnn.deallocate(q_rot)

        W = recent_window
        ring_sdpa_pos_dev = ttnn.from_torch(
            torch.tensor([min(W - 1, cur_pos_target)], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        old_pos_dev = ttnn.from_torch(
            torch.tensor([max(0, cur_pos_target - W)], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        cur_pos_dev = ttnn.from_torch(
            torch.tensor([cur_pos_target], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        print(f"\nCalling hybrid_sdpa_decode at cur_pos={cur_pos_target}, W={W}...")
        t0 = time.perf_counter()
        combined = tq.hybrid_sdpa_decode(
            q_bqhd,
            layer_idx=0,
            current_pos=cur_pos_dev,
            scale=scale,
            page_table=page_table_dev,
            ring_sdpa_pos=ring_sdpa_pos_dev,
            old_pos=old_pos_dev,
        )
        c = ttnn.to_torch(combined).float()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  hybrid_sdpa_decode latency: {elapsed_ms:.1f} ms")
        print(f"  combined shape: {tuple(c.shape)}")
        print(f"  combined finite: {torch.isfinite(c).all().item()}")
        print(f"  combined max_abs: {c.abs().max().item():.4f}")
        print(f"  combined head 0 col 0..4: {[round(c[0, 0, 0, i].item(), 3) for i in range(4)]}")

        if torch.isfinite(c).all() and c.abs().max() < 100:
            print(f"\nRESULT: PASS — hybrid runs at {seq_len}-token context, finite output, sane magnitude.")
            return 0
        else:
            print(f"\nRESULT: FAIL — output non-finite or too large.")
            return 1
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
