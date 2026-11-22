// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_block_ops.hpp"

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
    uint32_t straddle_jump = 0) {
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t row_offset = row * num_cols;
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

/**
 * Context for lightweight mask application.
 * All mask tiles reside in a single CB. This struct stores the pre-resolved mask metadata used when
 * lightweight masking is enabled; enablement itself is controlled by the `lightweight_mask_enabled`
 * template parameter(s), not by default-constructing this context.
 */
struct LightweightMaskContext {
    bool is_causal = false;                       // Causal masking active for this context instance
    uint32_t neginf_tile_idx = 0;                 // Index of -inf tile in the mask CB
    uint32_t causal_diag_tile_idx = 0;            // Index of causal diagonal tile in the mask CB
    uint32_t primary_diag_tile_idx = 0;           // Causal diagonal, or sliding-window trailing-primary tile
    uint32_t sliding_leading_prev_tile_idx = 0;   // Index of previous sliding-window leading tile
    uint32_t sliding_leading_tile_idx = 0;        // Index of current sliding-window leading tile in the mask CB
    uint32_t sliding_trailing_next_tile_idx = 0;  // Index of next sliding-window trailing tile
    uint32_t global_n_padded_tiles = 0;           // Fully padded K tile columns for global_n chunk
    uint32_t local_n_padded_tiles = 0;            // Fully padded K tile columns for local_n chunk
    uint32_t joint_n_padded_tiles = 0;            // Fully padded K tile columns for joint_l chunk
    uint32_t global_n_partial_col = 0;            // Column within tile where global_n padding starts (0 = no partial)
    uint32_t joint_l_partial_col = 0;             // Column within tile where joint_l padding starts (0 = no partial)
    uint32_t global_n_partial_tile_idx = 0;       // Index of global_n partial tile in the mask CB
    uint32_t joint_l_partial_tile_idx = 0;        // Index of joint_l partial tile in the mask CB
    uint32_t straddle_num_padded_tiles = 0;       // Trailing -inf tiles on straddle chunk (0 = inactive)
    uint32_t straddle_mask_chunk_id = 0;          // K chunk index where straddle mask applies

    /**
     * Resolve which mask type applies for a given K chunk and return pre-resolved params.
     * Called once per K chunk; the resolved values are then used per subblock.
     */
    void resolve_for_chunk(
        uint32_t Sk_chunk_t,
        uint32_t k_chunk,
        uint32_t num_local_k_chunks,
        bool ring_iter_needs_global_n_mask,
        bool ring_iter_needs_joint_n_mask,
        bool local_n_needs_masking,
        uint32_t global_n_mask_chunk_id,
        uint32_t local_n_mask_chunk_id,
        uint32_t joint_n_mask_chunk_id,
        uint32_t& out_num_padded,
        uint32_t& out_boundary_col,
        uint32_t& out_partial_tile_idx,
        bool& out_has_partial) const {
        out_num_padded = 0;
        out_has_partial = false;
        out_boundary_col = 0;
        out_partial_tile_idx = 0;

        if (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) {
            out_num_padded = global_n_padded_tiles;
            out_has_partial = (global_n_partial_col > 0);
            out_boundary_col = Sk_chunk_t - out_num_padded - (out_has_partial ? 1 : 0);
            out_partial_tile_idx = global_n_partial_tile_idx;
        } else if (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) {
            out_num_padded = local_n_padded_tiles;
        } else if (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id) {
            out_num_padded = joint_n_padded_tiles;
            out_has_partial = (joint_l_partial_col > 0);
            out_boundary_col = Sk_chunk_t - out_num_padded - (out_has_partial ? 1 : 0);
            out_partial_tile_idx = joint_l_partial_tile_idx;
        } else if (straddle_num_padded_tiles > 0 && k_chunk == straddle_mask_chunk_id) {
            out_num_padded = straddle_num_padded_tiles;
        }
    }

    /**
     * Apply the lightweight mask for a given K chunk (non-streaming path).
     * When is_causal: applies causal stamp, then also applies padding stamp if this K chunk has padding.
     * When !is_causal: applies padding stamp only.
     * L1-accumulate stamps are additive, so applying both is safe (-inf + -inf = -inf, -inf + 0 = -inf).
     *
     * @tparam dst_size  DST register capacity (8 for fp16b half-sync, 4 for fp32_dest_acc).
     */
    template <uint32_t dst_size>
    void apply(
        uint32_t cb_mask_in,
        uint32_t cb_qk_im,
        uint32_t Sk_chunk_t,
        uint32_t Sq_chunk_t,
        uint32_t k_chunk,
        uint32_t num_local_k_chunks,
        bool ring_iter_needs_global_n_mask,
        bool ring_iter_needs_joint_n_mask,
        bool local_n_needs_masking,
        uint32_t global_n_mask_chunk_id,
        uint32_t local_n_mask_chunk_id,
        uint32_t joint_n_mask_chunk_id,
        uint32_t q_start_tile,
        uint32_t k_start_tile,
        uint32_t straddle_col = 0,
        uint32_t straddle_jump = 0) const {
        if (is_causal) {
            apply_causal_mask_lightweight<dst_size>(
                cb_mask_in,
                neginf_tile_idx,
                causal_diag_tile_idx,
                cb_qk_im,
                q_start_tile,
                k_start_tile,
                Sq_chunk_t,
                Sk_chunk_t,
                straddle_col,
                straddle_jump);
        }

        // Apply padding stamp (also when is_causal — the causal stamp doesn't handle K padding).
        uint32_t num_padded, boundary_col, partial_tile_idx;
        bool has_partial;
        resolve_for_chunk(
            Sk_chunk_t,
            k_chunk,
            num_local_k_chunks,
            ring_iter_needs_global_n_mask,
            ring_iter_needs_joint_n_mask,
            local_n_needs_masking,
            global_n_mask_chunk_id,
            local_n_mask_chunk_id,
            joint_n_mask_chunk_id,
            num_padded,
            boundary_col,
            partial_tile_idx,
            has_partial);

        if (has_partial) {
            apply_partial_mask_lightweight(
                cb_mask_in, partial_tile_idx, cb_qk_im, boundary_col, Sk_chunk_t, Sq_chunk_t);
        }
        if (num_padded > 0) {
            apply_padded_mask_lightweight_runtime<dst_size>(
                cb_mask_in, neginf_tile_idx, cb_qk_im, num_padded, Sk_chunk_t, Sq_chunk_t);
        }
    }
};
