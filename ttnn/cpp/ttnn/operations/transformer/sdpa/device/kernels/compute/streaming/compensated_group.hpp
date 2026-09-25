// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

inline void group2_pack_visibility_fence() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

inline void group2_initialize_root(uint32_t root_cb, uint32_t tiles) {
    CircularBuffer(root_cb).reserve_back(tiles);
    tile_regs_acquire();
    MATH((
        SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_zero_sum, 0, VectorMode::None)));
    tile_regs_commit();
    tile_regs_wait();
    configure_single_tile_pack(root_cb);
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t i = 0; i < tiles; ++i) {
        pack_tile<true>(0, root_cb, i);
    }
    tile_regs_release();
    CircularBuffer(root_cb).push_back(tiles);
}

// Paired-column SFPU programs consume two adjacent column tiles. An odd head-dim width
// ends with a single column: duplicate it into both slots (lanes are independent) and
// pack only the first.
inline void group2_copy_pair(uint32_t cb, uint32_t tile, uint32_t dst, bool single) {
    if (single) {
        copy_block(cb, tile, dst, 1);
        copy_block(cb, tile, dst + 1, 1);
    } else {
        copy_block(cb, tile, dst, 2);
    }
}

// Row layouts: root and scratch rows hold two dh-tile planes (root: high/low; scratch:
// PV/local), so a row spans 2 * dh tiles. dh is the head dim in tiles.
inline void group2_bootstrap_row(
    uint32_t root_cb, uint32_t scratch_cb, uint32_t global_row, uint32_t read_row, uint32_t rows = 2, uint32_t dh = 4) {
    const uint32_t row_stride = 2 * dh;
    // K0 has no correction-CB publication, so explicitly publish preceding PV.
    group2_pack_visibility_fence();
    PACK((llk_pack_reconfig_l1_acc(0)));
    // Copy in pieces that fit the BF16 half-sync dest (8 tiles).
    constexpr uint32_t piece = 8;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t c = 0; c < dh; c += piece) {
            const uint32_t n = dh - c < piece ? dh - c : piece;
            configure_pack_width(root_cb, n);
            tile_regs_acquire();
            copy_init(scratch_cb);
            copy_block(scratch_cb, row_stride * (read_row + i) + c, 0, n);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, root_cb, row_stride * (global_row + i) + c);
            tile_regs_release();
        }
    }
}

inline void group2_numerator_row(
    uint32_t root_cb,
    uint32_t scratch_cb,
    uint32_t correction_cb,
    uint32_t root_read_row,
    uint32_t root_write_row,
    uint32_t scratch_read_row,
    uint32_t scratch_write_row,
    bool identity,
    bool boundary,
    bool odd,
    bool has_local,
    uint32_t rows = 2,
    uint32_t dh = 4) {
    const uint32_t row_stride = 2 * dh;
    // Original correction-CB push already publishes PV. Protected/local
    // updates are in-place; the subsequent scratch publication drains them.
    PACK((llk_pack_reconfig_l1_acc(0)));
    if (identity && !boundary) {
        // Odd PV is already local. No data copy, no arithmetic, no lost term.
        return;
    }
    const uint32_t chunk_plane = odd ? dh : 0;
    if (!has_local) {
        // No unmerged local contribution: canonical paired compensated update,
        // with distinct source-chunk and destination-root CBs. Physical local
        // slots may be stale; ALL local readers below are excluded by this branch.
        PACK((ckernel::sfpu::init_sdpa_compensated_block_macros()));
        configure_pack_width(root_cb, 2);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < dh; j += 2) {
                const bool single = j + 1 == dh;
                if (single) {
                    configure_pack_width(root_cb, 1);
                }
                tile_regs_acquire();
                copy_init(root_cb);
                group2_copy_pair(root_cb, row_stride * (root_read_row + i) + j, 0, single);
                group2_copy_pair(root_cb, row_stride * (root_read_row + i) + dh + j, 2, single);
                group2_copy_pair(scratch_cb, row_stride * (scratch_read_row + i) + chunk_plane + j, 4, single);
                if (!identity) {
                    unary_bcast_init<BroadcastType::COL>(correction_cb);
                    unary_bcast<BroadcastType::COL>(correction_cb, i, 6);
                    unary_bcast_uninit<BroadcastType::COL>(correction_cb);
                }
                tile_regs_commit();
                tile_regs_wait();
                PACK((SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_sdpa_identity_state,
                    (2, false),
                    0,
                    VectorMode::None,
                    identity)));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_tile<true>(0, root_cb, row_stride * (root_write_row + i) + j);
                pack_tile<true>(2, root_cb, row_stride * (root_write_row + i) + dh + j);
                tile_regs_release();
                if (single) {
                    configure_pack_width(root_cb, 2);
                }
            }
        }
        return;
    }
    if (identity) {
        if (!odd) {
            PACK((ckernel::sfpu::init_group2_identity_replay()));
        }
        configure_pack_width(root_cb, 2);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < dh; j += 2) {
                const bool single = j + 1 == dh;
                if (single) {
                    configure_pack_width(root_cb, 1);
                }
                tile_regs_acquire();
                copy_init(root_cb);
                group2_copy_pair(root_cb, row_stride * (root_read_row + i) + j, 0, single);
                group2_copy_pair(root_cb, row_stride * (root_read_row + i) + dh + j, 2, single);
                if (!odd) {
                    group2_copy_pair(scratch_cb, row_stride * (scratch_read_row + i) + dh + j, 4, single);
                }
                group2_copy_pair(scratch_cb, row_stride * (scratch_read_row + i) + chunk_plane + j, 6, single);
                tile_regs_commit();
                tile_regs_wait();
                if (odd) {
                    PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_group2_identity_odd_fold, 0, VectorMode::None)));
                } else {
                    PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_group2_identity_replay, 0, VectorMode::None)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_tile<true>(0, root_cb, row_stride * (root_write_row + i) + j);
                pack_tile<true>(2, root_cb, row_stride * (root_write_row + i) + dh + j);
                if (odd) {
                    pack_tile<true>(4, scratch_cb, row_stride * (scratch_write_row + i) + dh + j);
                }
                tile_regs_release();
                if (single) {
                    configure_pack_width(root_cb, 2);
                }
            }
        }
        return;
    }
    // Max changes force a fold, independently of the fixed group boundary.
    configure_single_tile_pack(root_cb);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < dh; ++j) {
            tile_regs_acquire();
            copy_init(root_cb);
            copy_tile(root_cb, row_stride * (root_read_row + i) + j, 0);
            copy_tile(root_cb, row_stride * (root_read_row + i) + dh + j, 1);
            if (!odd) {
                copy_tile(scratch_cb, row_stride * (scratch_read_row + i) + dh + j, 2);
            }
            copy_tile(scratch_cb, row_stride * (scratch_read_row + i) + chunk_plane + j, 3);
            unary_bcast_init<BroadcastType::COL>(correction_cb);
            unary_bcast<BroadcastType::COL>(correction_cb, i, 4);
            unary_bcast_uninit<BroadcastType::COL>(correction_cb);
            tile_regs_commit();
            tile_regs_wait();
            if (odd) {
                PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, calculate_group2_changed_odd_fold, 0, VectorMode::None)));
            } else {
                PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, calculate_group2_changed_fold, 0, VectorMode::None)));
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile<true>(0, root_cb, row_stride * (root_write_row + i) + j);
            pack_tile<true>(1, root_cb, row_stride * (root_write_row + i) + dh + j);
            if (odd) {
                pack_tile<true>(2, scratch_cb, row_stride * (scratch_write_row + i) + dh + j);
            }
            tile_regs_release();
        }
    }
}
