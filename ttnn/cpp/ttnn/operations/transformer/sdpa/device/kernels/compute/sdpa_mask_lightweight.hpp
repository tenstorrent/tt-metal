// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lightweight mask stamping for SDPA: the mask tiles are added onto the QK^T scores already in L1 by
// the packer with L1 accumulate, so the scores never come back through DST for the mask. Shared by
// ttnn's SDPA compute kernels (compute_common.hpp) and tt-train's SDPA forward, which stacks several
// heads' query bands in one score CB and passes row_base per band.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

/**
 * Batch-stamp a single tile onto a range of positions in out_cb using L1 accumulate.
 * Caller must have already called copy_init and llk_pack_reconfig_l1_acc(1).
 *
 * @tparam dst_batch  Max tiles per DST cycle (DST register capacity, typically 8 for fp16b half-sync).
 */
template <uint32_t dst_batch>
void stamp_tile_range_l1_acc(
    uint32_t src_cb, uint32_t src_tile_idx, uint32_t out_cb, uint32_t out_offset, uint32_t count) {
    for (uint32_t base = 0; base < count; base += dst_batch) {
        uint32_t batch = (count - base < dst_batch) ? (count - base) : dst_batch;
        tile_regs_acquire();
        for (uint32_t i = 0; i < batch; i++) {
            copy_tile(src_cb, src_tile_idx, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < batch; i++) {
            pack_tile<true>(i, out_cb, out_offset + base + i);
        }
        tile_regs_release();
    }
}

template <uint32_t dst_batch>
void apply_padded_mask_lightweight_runtime(
    uint32_t neginf_cb,
    uint32_t neginf_tile_idx,
    uint32_t out_cb,
    uint32_t num_padded,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {  // first out_cb tile-row of this query band; nonzero when heads span >1 DEST band
    uint32_t start = num_cols - num_padded;

    reconfig_data_format_srca(neginf_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(neginf_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        stamp_tile_range_l1_acc<dst_batch>(
            neginf_cb, neginf_tile_idx, out_cb, (row_base + row) * num_cols + start, num_padded);
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Lightweight partial mask: L1-accumulate a partial mask tile (0 for valid, -inf for padded columns)
 * onto the boundary tile position in out_cb. The partial tile is permanently fronted in the CB.
 *
 * @param mask_cb          CB holding mask tiles, permanently fronted
 * @param partial_tile_idx Index of the partial tile within the CB
 * @param out_cb           QK intermediate CB (already wait-fronted)
 * @param boundary_col     Column index within the chunk where the boundary tile is
 * @param num_cols         Total K tiles per row (Sk_chunk_t)
 * @param num_rows         Q tiles per chunk (Sq_chunk_t)
 */
void apply_partial_mask_lightweight(
    uint32_t mask_cb,
    uint32_t partial_tile_idx,
    uint32_t out_cb,
    uint32_t boundary_col,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {  // first out_cb tile-row of this query band; nonzero when heads span >1 DEST band
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        tile_regs_acquire();
        copy_tile(mask_cb, partial_tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb, (row_base + row) * num_cols + boundary_col);
        tile_regs_release();
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Lightweight causal mask: stamps neginf and diagonal tiles onto QKT using L1 accumulate.
 *
 * For Q tile-row i (0..num_rows-1) processing K chunk starting at k_start_tile:
 *   diag_col = q_start_tile + i - k_start_tile
 *   - diag_col < 0:           entire row above diagonal -> stamp neginf on all num_cols tiles
 *   - 0 <= diag_col < num_cols: diagonal tile at col diag_col, neginf at cols diag_col+1..num_cols-1
 *   - diag_col >= num_cols:     entire row below diagonal -> no mask needed
 * row_base is the first out_cb tile-row of this query band, nonzero when several heads' bands are
 * stacked in out_cb (as in tt-train's SDPA forward, which stamps one band per stacked head).
 */
template <uint32_t dst_batch>
void apply_causal_mask_lightweight(
    uint32_t mask_cb,
    uint32_t neginf_idx,
    uint32_t diag_idx,
    uint32_t out_cb,
    uint32_t q_start_tile,
    uint32_t k_start_tile,
    uint32_t num_rows,
    uint32_t num_cols,
    uint32_t straddle_col = 0,
    uint32_t straddle_jump = 0,
    uint32_t row_base = 0) {
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t row_offset = (row_base + row) * num_cols;
        const int32_t q_pos = static_cast<int32_t>(q_start_tile + row);

        if (straddle_col == 0) {
            // Fast path: K coords contiguous across cols.
            int32_t diag_col = q_pos - static_cast<int32_t>(k_start_tile);
            if (diag_col < 0) {
                // Entire row above diagonal -> stamp all neginf
                stamp_tile_range_l1_acc<dst_batch>(mask_cb, neginf_idx, out_cb, row_offset, num_cols);
            } else if (static_cast<uint32_t>(diag_col) < num_cols) {
                // Stamp the diagonal tile
                tile_regs_acquire();
                copy_tile(mask_cb, diag_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, out_cb, row_offset + static_cast<uint32_t>(diag_col));
                tile_regs_release();

                // Stamp neginf tiles to the right of diagonal
                uint32_t neginf_start = static_cast<uint32_t>(diag_col) + 1;
                if (neginf_start < num_cols) {
                    stamp_tile_range_l1_acc<dst_batch>(
                        mask_cb, neginf_idx, out_cb, row_offset + neginf_start, num_cols - neginf_start);
                }
            }
            // else: diag_col >= num_cols -> entire row below diagonal, no mask needed
        } else {
            // Chunked-prefill straddle: K coord jumps by straddle_jump at col >= straddle_col
            // (the K-chunk crosses a slab boundary). Evaluate per-col.
            for (uint32_t col = 0; col < num_cols; col++) {
                int32_t k_pos = static_cast<int32_t>(k_start_tile) + static_cast<int32_t>(col);
                if (col >= straddle_col) {
                    k_pos += static_cast<int32_t>(straddle_jump);
                }
                if (k_pos > q_pos) {
                    stamp_tile_range_l1_acc<dst_batch>(mask_cb, neginf_idx, out_cb, row_offset + col, 1);
                } else if (k_pos == q_pos) {
                    tile_regs_acquire();
                    copy_tile(mask_cb, diag_idx, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, out_cb, row_offset + col);
                    tile_regs_release();
                }
            }
        }
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}
