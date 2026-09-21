// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

inline void group2_pack_visibility_fence() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

inline void group2_initialize_root(uint32_t root_cb) {
    CircularBuffer(root_cb).reserve_back(64);
    tile_regs_acquire();
    MATH((
        SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_zero_sum, 0, VectorMode::None)));
    tile_regs_commit();
    tile_regs_wait();
    configure_single_tile_pack(root_cb);
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t i = 0; i < 64; ++i) {
        pack_tile<true>(0, root_cb, i);
    }
    tile_regs_release();
    CircularBuffer(root_cb).push_back(64);
}

inline void group2_bootstrap_row(uint32_t root_cb, uint32_t scratch_cb, uint32_t global_row, uint32_t read_row) {
    // K0 has no correction-CB publication, so explicitly publish preceding PV.
    group2_pack_visibility_fence();
    PACK((llk_pack_reconfig_l1_acc(0)));
    configure_pack_width(root_cb, 4);
    for (uint32_t i = 0; i < 2; ++i) {
        tile_regs_acquire();
        copy_init(scratch_cb);
        copy_block(scratch_cb, 8 * (read_row + i), 0, 4);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, root_cb, 8 * (global_row + i));
        tile_regs_release();
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
    bool has_local) {
    // Original correction-CB push already publishes PV. Protected/local
    // updates are in-place; the subsequent scratch publication drains them.
    PACK((llk_pack_reconfig_l1_acc(0)));
    if (identity && !boundary) {
        // Odd PV is already local. No data copy, no arithmetic, no lost term.
        return;
    }
    const uint32_t chunk_plane = odd ? 4 : 0;
    if (!has_local) {
        // No unmerged local contribution: canonical paired compensated update,
        // with distinct source-chunk and destination-root CBs. Physical local
        // slots may be stale; ALL local readers below are excluded by this branch.
        PACK((ckernel::sfpu::init_sdpa_compensated_block_macros()));
        configure_pack_width(root_cb, 2);
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 4; j += 2) {
                tile_regs_acquire();
                copy_init(root_cb);
                copy_block(root_cb, 8 * (root_read_row + i) + j, 0, 2);
                copy_block(root_cb, 8 * (root_read_row + i) + 4 + j, 2, 2);
                copy_block(scratch_cb, 8 * (scratch_read_row + i) + chunk_plane + j, 4, 2);
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
                pack_tile<true>(0, root_cb, 8 * (root_write_row + i) + j);
                pack_tile<true>(2, root_cb, 8 * (root_write_row + i) + 4 + j);
                tile_regs_release();
            }
        }
        return;
    }
    if (identity) {
        if (!odd) {
            PACK((ckernel::sfpu::init_group2_identity_replay()));
        }
        configure_pack_width(root_cb, 2);
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 4; j += 2) {
                tile_regs_acquire();
                copy_init(root_cb);
                copy_block(root_cb, 8 * (root_read_row + i) + j, 0, 2);
                copy_block(root_cb, 8 * (root_read_row + i) + 4 + j, 2, 2);
                if (!odd) {
                    copy_block(scratch_cb, 8 * (scratch_read_row + i) + 4 + j, 4, 2);
                }
                copy_block(scratch_cb, 8 * (scratch_read_row + i) + chunk_plane + j, 6, 2);
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
                pack_tile<true>(0, root_cb, 8 * (root_write_row + i) + j);
                pack_tile<true>(2, root_cb, 8 * (root_write_row + i) + 4 + j);
                if (odd) {
                    pack_tile<true>(4, scratch_cb, 8 * (scratch_write_row + i) + 4 + j);
                }
                tile_regs_release();
            }
        }
        return;
    }
    // Max changes force a fold, independently of the fixed group boundary.
    configure_single_tile_pack(root_cb);
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            tile_regs_acquire();
            copy_init(root_cb);
            copy_tile(root_cb, 8 * (root_read_row + i) + j, 0);
            copy_tile(root_cb, 8 * (root_read_row + i) + 4 + j, 1);
            if (!odd) {
                copy_tile(scratch_cb, 8 * (scratch_read_row + i) + 4 + j, 2);
            }
            copy_tile(scratch_cb, 8 * (scratch_read_row + i) + chunk_plane + j, 3);
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
            pack_tile<true>(0, root_cb, 8 * (root_write_row + i) + j);
            pack_tile<true>(1, root_cb, 8 * (root_write_row + i) + 4 + j);
            if (odd) {
                pack_tile<true>(2, scratch_cb, 8 * (scratch_write_row + i) + 4 + j);
            }
            tile_regs_release();
        }
    }
}
