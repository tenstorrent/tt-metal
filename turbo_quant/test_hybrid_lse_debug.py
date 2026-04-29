#!/usr/bin/env python3
"""Print LSE values from both halves of hybrid_sdpa_decode at a single step.

Goal: confirm ring's LSE (pre_rescaled+return_lse, new code path) and old's
LSE (full dequant+return_lse, existing path) are in comparable ranges.
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
        cur_pos_target = 100  # > W

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

        # Match eval: rotation absorbed into W_v (so V comes pre-rotated, K still rotated by setup.rotation)
        tq.rotation_absorbed = True
        torch.manual_seed(0)

        blocks_per_seq = (seq_len + block_size - 1) // block_size
        page_table_torch = torch.arange(blocks_per_seq, dtype=torch.int32).reshape(1, -1)
        page_table_dev = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        # Populate cache
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
        print("Cache populated.")

        # Build Q in [B=1, NQH=32, 1, DH=128] layout (matches eval harness post-permute).
        # This is what the kernel expects for B=1, NQH=32.
        q_raw = torch.randn(1, 1, nqh, head_dim)  # [1, B, NQH, DH]
        q_dev = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # Pre-rotate (rotation_absorbed=False so Q gets rotated).
        q_rot = tq.pre_rotate_query(q_dev)
        ttnn.deallocate(q_dev)
        # Permute [1, B, NQH, DH] → [B, NQH, 1, DH] like attention.py does.
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

        from turbo_quant.ttnn_integration import get_codebook

        centroids = get_codebook(head_dim, bits, device="cpu", dtype=torch.float32).centroids.tolist()

        # Ring SDPA (pre_rescaled=True, return_lse=True) — NEW PATH
        ring_outs = ttnn.experimental.turbo_quant_sdpa_decode(
            q_bqhd,
            tq.k_ring_dev[0],
            tq.k_norms_dev[0],
            tq.v_ring_dev[0],
            tq.v_norms_dev[0],
            tq.ring_page_table_dev,
            ring_sdpa_pos_dev,
            centroids,
            scale,
            True,
            1,
            True,
        )
        out_new = ttnn.to_torch(ring_outs[0]).float()
        lse_new = ttnn.to_torch(ring_outs[1]).float()
        ttnn.deallocate(ring_outs[0])
        ttnn.deallocate(ring_outs[1])

        # Old SDPA (pre_rescaled=False, return_lse=True) — EXISTING PATH
        old_outs = ttnn.experimental.turbo_quant_sdpa_decode(
            q_bqhd,
            tq.k_indices_dev[0],
            tq.k_norms_dev[0],
            tq.v_indices_dev[0],
            tq.v_norms_dev[0],
            page_table_dev,
            old_pos_dev,
            centroids,
            scale,
            False,
            1,
            True,
        )
        out_old = ttnn.to_torch(old_outs[0]).float()
        lse_old = ttnn.to_torch(old_outs[1]).float()
        ttnn.deallocate(old_outs[0])
        ttnn.deallocate(old_outs[1])

        print(f"\ncur_pos={cur_pos_target}, W={W}")
        print(f"ring_sdpa_pos={min(W-1, cur_pos_target)}, old_pos={max(0, cur_pos_target-W)}")
        print(f"\nLSE_new (ring, pre_rescaled+return_lse):")
        print(f"  shape={tuple(lse_new.shape)}")
        for c in [0, 1, 8, 16, 31]:
            v = lse_new[0, 0, 0, c].item()
            print(f"  head 0 col {c:2d}: {v:.4f}")
        print(f"  col 0:  min={lse_new[0,:,0,0].min().item():.4f} max={lse_new[0,:,0,0].max().item():.4f}")
        print(f"  col 1:  min={lse_new[0,:,0,1].min().item():.4f} max={lse_new[0,:,0,1].max().item():.4f}")
        print(f"  col 16: min={lse_new[0,:,0,16].min().item():.4f} max={lse_new[0,:,0,16].max().item():.4f}")

        print(f"\nLSE_old (TQ, full dequant+return_lse):")
        print(f"  shape={tuple(lse_old.shape)}")
        for c in [0, 1, 8, 16, 31]:
            v = lse_old[0, 0, 0, c].item()
            print(f"  head 0 col {c:2d}: {v:.4f}")
        print(f"  col 0:  min={lse_old[0,:,0,0].min().item():.4f} max={lse_old[0,:,0,0].max().item():.4f}")
        print(f"  col 1:  min={lse_old[0,:,0,1].min().item():.4f} max={lse_old[0,:,0,1].max().item():.4f}")
        print(f"  col 16: min={lse_old[0,:,0,16].min().item():.4f} max={lse_old[0,:,0,16].max().item():.4f}")

        print(
            f"\nout_new (ring): finite={torch.isfinite(out_new).all().item()} max_abs={out_new.abs().max().item():.4f}"
        )
        print(f"out_old (TQ): finite={torch.isfinite(out_old).all().item()} max_abs={out_old.abs().max().item():.4f}")

        # Now run the actual _combine_lse and inspect
        print("\n=== Re-running and combining via _combine_lse ===")
        ring_outs2 = ttnn.experimental.turbo_quant_sdpa_decode(
            q_bqhd,
            tq.k_ring_dev[0],
            tq.k_norms_dev[0],
            tq.v_ring_dev[0],
            tq.v_norms_dev[0],
            tq.ring_page_table_dev,
            ring_sdpa_pos_dev,
            centroids,
            scale,
            True,
            1,
            True,
        )
        old_outs2 = ttnn.experimental.turbo_quant_sdpa_decode(
            q_bqhd,
            tq.k_indices_dev[0],
            tq.k_norms_dev[0],
            tq.v_indices_dev[0],
            tq.v_norms_dev[0],
            page_table_dev,
            old_pos_dev,
            centroids,
            scale,
            False,
            1,
            True,
        )
        combined = tq._combine_lse(old_outs2[0], old_outs2[1], ring_outs2[0], ring_outs2[1])
        c = ttnn.to_torch(combined).float()
        print(f"combined shape: {tuple(c.shape)}")
        print(f"combined finite: {torch.isfinite(c).all().item()}")
        print(f"combined max_abs: {c.abs().max().item():.4f}")
        print(f"combined head 0 col 0..4: {c[0, 0, 0, :4].tolist()}")
        print(f"combined head 0 col 64..68: {c[0, 0, 0, 64:68].tolist()}")
        ttnn.deallocate(combined)

        # Compute expected: w_old = exp(L_o-Lmax)/(exp(L_o-Lmax)+exp(L_n-Lmax))
        L_o = lse_old[0, 0, 0, 0].item()
        L_n = lse_new[0, 0, 0, 0].item()
        L_max = max(L_o, L_n)
        import math

        d_o = math.exp(L_o - L_max)
        d_n = math.exp(L_n - L_max)
        w_o = d_o / (d_o + d_n)
        w_n = d_n / (d_o + d_n)
        print(f"\nExpected combine for head 0: w_old={w_o:.4f}, w_new={w_n:.4f}")
        expected_col0 = w_o * out_old[0, 0, 0, 0].item() + w_n * out_new[0, 0, 0, 0].item()
        print(f"Expected combined head 0 col 0: {expected_col0:.6f}")
        print(f"Actual   combined head 0 col 0: {c[0, 0, 0, 0].item():.6f}")

        # Now invoke the actual hybrid_sdpa_decode (the fast path the eval uses)
        print("\n=== via hybrid_sdpa_decode (fast path) ===")
        # Need fresh q since previous one was used
        q_dev2 = ttnn.from_torch(q_raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        q_rot2 = tq.pre_rotate_query(q_dev2)
        ttnn.deallocate(q_dev2)
        q_bqhd2 = ttnn.permute(q_rot2, (1, 2, 0, 3))
        ttnn.deallocate(q_rot2)
        ring_sdpa_pos_dev2 = ttnn.from_torch(
            torch.tensor([min(W - 1, cur_pos_target)], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        old_pos_dev2 = ttnn.from_torch(
            torch.tensor([max(0, cur_pos_target - W)], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        cur_pos_dev2 = ttnn.from_torch(
            torch.tensor([cur_pos_target], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        combined2 = tq.hybrid_sdpa_decode(
            q_bqhd2,
            layer_idx=0,
            current_pos=cur_pos_dev2,
            scale=scale,
            page_table=page_table_dev,
            ring_sdpa_pos=ring_sdpa_pos_dev2,
            old_pos=old_pos_dev2,
        )
        c2 = ttnn.to_torch(combined2).float()
        print(f"hybrid_sdpa_decode combined head 0 col 0: {c2[0, 0, 0, 0].item():.6f}")
        print(f"hybrid_sdpa_decode finite: {torch.isfinite(c2).all().item()}")
        print(f"hybrid_sdpa_decode max_abs: {c2.abs().max().item():.4f}")

        # Approximate sanity: LSE = max·scale + log(N) where max·scale ~ 0..2, log(N)
        # for ring (W=64, all valid) should be ~log(64)=4.16, for old (cur_pos-W+1=37
        # tokens [0..36]) should be ~log(37)=3.6.
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
