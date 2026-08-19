// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GNA fused neighborhood compute -- P2 SKELETON (not yet compiled/wired; the loop nest + CB protocol made
// concrete so P3 (reader) and P5 (program factory) have a precise target). See GNA_FUSED_KERNEL_SPEC.md.
//
// GOAL: plain-flash utilization (~50% of peak) for 3-D neighborhood attention by keeping a spatial REGION's
// K/V resident in L1 and reusing it across the region's query-blocks (FlashAttention tiling), with a
// tile-quantized mask (block-skip dead box-tiles; per-element stamp only on boundary tiles). Exact.
//
// vs the current fused kernel: that gathers each query-block's box fresh (halo re-read ~23x) and runs the
// windowed-narrowing path at ~4%. Here each region K/V tile is read ONCE (by the reader) and every query-
// block in the region reuses the resident copy.
//
// ---- CB protocol (reader -> this compute) ----
//   cb_kv_region : region K/V-union, resident. [region_box_tiles] K then V (or interleaved), TILE layout.
//                  The reader fills this ONCE per region; compute does NOT pop it until the region ends.
//   cb_q_block   : one query-block's Q, [vol_tiles], streamed per query-block (pop after each).
//   cb_tile_mask : per (query-block) the [q_tiles x box_tiles] tri-state tile-mask: 0=all-live (no stamp),
//                  1=dead (skip), 2=boundary (needs the per-element stamp from cb_elem_mask). Small.
//   cb_elem_mask : per-element -inf stamp for boundary tiles only (streamed for the few '2' tiles).
//   cb_out       : this query-block's output [vol_tiles x vDHt], pushed per block.
//   cb_qblk_range: per query-block, its [box_tile_lo, box_tile_hi) within cb_kv_region (its box is a sub-
//                  range of the region-union) -- so a block only visits its own box tiles, not the whole
//                  region-union. Reader computes it from box geometry (block_qtile_k_band, region-relative).

// #include ... (compute_common.hpp, sdpa flash primitives, dataflow) -- P4

/*
compute loop nest (P4 fills the flash body from sdpa.cpp primitives):

for (region = 0; region < num_regions_this_core; ++region) {
    cb_kv_region.wait_front(region_box_tiles);          // reader loaded region K/V ONCE (read-once reuse)

    for (qb = 0; qb < region.num_qblocks; ++qb) {
        cb_q_block.wait_front(vol_tiles);               // this block's Q
        cb_qblk_range.wait_front(1);                    // [lo, hi) box-tile range for this block
        (lo, hi) = read(cb_qblk_range);

        // ONLINE-SOFTMAX FLASH over this block's box-tile sub-range of the resident region K/V:
        init running max m=-inf, sum l=0, acc O=0 (per q-tile)
        for (kt = lo; kt < hi; ++kt) {                  // only the block's box tiles (not full region)
            tile_state = cb_tile_mask[qb][kt]           // 0 live / 1 dead / 2 boundary
            if (tile_state == 1) continue;              // BLOCK-SKIP: dead box-tile, no matmul (the win)
            S = matmul(cb_q_block, cb_kv_region[kt])     // QK for this (q_block, box-tile), PLAIN path
            if (tile_state == 2) S += cb_elem_mask[..]   // boundary tile: per-element -inf stamp
            online-softmax update (m, l, O, S, cb_kv_region_V[kt])   // exp, rescale, PV accumulate
        }
        normalize O by l
        cb_out.push_back(vol_tiles * vDHt)              // block output (block-order)
        cb_q_block.pop_front(vol_tiles); cb_qblk_range.pop_front(1)
    }

    cb_kv_region.pop_front(region_box_tiles);           // done with this region's K/V
}
*/

// KEY DESIGN NOTES for P4:
//  - The block-skip (tile_state==1) is what avoids the box over-compute AND gives the util: a q-block only
//    matmuls its live box-tiles. Most region-union tiles are dead for any single block.
//  - kt_inplace / subblock sizing: the earlier tile-skip hit a streaming subblock_h=1 corruption; here the
//    per-block flash uses the STANDARD (non-streaming) subblock sizing over its box sub-range, so it does NOT
//    reuse that path -- validate subblock config in P4.
//  - Region size (region_box_tiles) is bounded by L1 (cb_kv_region must fit). The program factory (P5) picks
//    it: bigger region = more reuse (fewer DRAM reads) but must fit L1 alongside Q/scores/softmax state.
//  - W-SP: the region's box-union may cross the W-shard edge; the reader pulls from the all-gathered full-W
//    K/V (as the current fused path does), Q stays W-sharded, region set is per-device.
//  - EXACTNESS: box always covers each query's window (mask_table), so with the tile-mask this == exact
//    neighborhood; op-parity gate in P6 must hit 99.9% vs the strided reference.
