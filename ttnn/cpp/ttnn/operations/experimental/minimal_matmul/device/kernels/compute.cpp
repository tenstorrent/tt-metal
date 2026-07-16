// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

#include "tt_metal/tools/profiler/kernel_profiler.hpp"

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/subchunk_bands.hpp"

#ifndef IN0_SUB_CHUNKS
#define IN0_SUB_CHUNKS 1
#endif

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

// For caller: if FUSE_TERNARY defined then out_cb == in_cb
/**
 * Add bias to input block
 * Performs: output = input + bias (row broadcast)
 *
 * stream_output:
 *   - true: Pushes tiles one row at a time (for intermediate output to next stage)
 *   - false: Pushes all tiles at end (for final output)
 */
void add_bias_block(uint32_t in_cb, uint32_t bias_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    add_bcast_rows_init_short(in_cb, bias_cb);
    reconfig_data_format(in_cb, bias_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, tile_id, n, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

void add_bias_and_addcmul_block(
    uint32_t intermediate_cb,
    uint32_t bias_cb,
    uint32_t ternary_a_cb,
    uint32_t ternary_b_cb,
    uint32_t scalar_value,
    uint32_t out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t broadcast_ternary_b) {
    // Note: unary_bcast_tile does not work with fp32_acc_to_dest=True.
    // As a workaround, we perform addcmul through multiple LLKs calls (mul_tiles, mul_unary_tile, add_tiles_bcast).

    const uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t DST_ID = 0;
#ifdef FUSE_BIAS
    // ============================================
    // STEP 1: Add bias block
    // Read from intermediate_cb and write back to intermediate_cb
    // ============================================

    add_bcast_rows_init_short(intermediate_cb, bias_cb);
    reconfig_data_format(intermediate_cb, bias_cb);
    pack_reconfig_data_format(intermediate_cb);

    // Wait for ALL input data ONCE at the beginning
    cb_wait_front(bias_cb, N_block_tiles);

    // Unpacker waits for intermediate_cb to be ready
    cb_wait_front(intermediate_cb, out_block_num_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(intermediate_cb, bias_cb, tile_id, n, DST_ID);

            tile_regs_commit();

            tile_regs_wait();
            pack_tile(DST_ID, intermediate_cb);
            tile_regs_release();
        }
    }

    // Pop input and push output ONCE at the end
    // cb_wait_front(intermediate_cb, out_block_num_tiles); // Unpacker-Packer sync
    // cb_pop_front(intermediate_cb, out_block_num_tiles);
    cb_pop_front(bias_cb, N_block_tiles);

    cb_pop_front(intermediate_cb, out_block_num_tiles);

    // Restore intermediate_cb to ready (+ sync packer/unpacker)
    cb_reserve_back(intermediate_cb, out_block_num_tiles);
    cb_push_back(intermediate_cb, out_block_num_tiles);
#endif  // FUSE_BIAS

    // ============================================
    // STEP 2: Multiply by ternary_b and scalar
    // Read from intermediate_cb and write back to intermediate_cb
    // broadcast_ternary_b: 1 = single row broadcast, 0 = row-by-row streaming
    // ============================================

    cb_wait_front(intermediate_cb, out_block_num_tiles);

    uint32_t tile_id = 0;

    if (broadcast_ternary_b) {
        // === BROADCAST: single row, wait/pop once ===
        cb_wait_front(ternary_b_cb, N_block_tiles);

#ifndef TERNARY_B_IS_FLOAT32
        mul_bcast_rows_init_short(intermediate_cb, ternary_b_cb);
#else
        unary_bcast_init<BroadcastType::ROW>(ternary_b_cb, intermediate_cb);
#endif  // TERNARY_B_IS_FLOAT32

        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb, ternary_b_cb);
        pack_reconfig_data_format(intermediate_cb);

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles_bcast<BroadcastType::ROW>(intermediate_cb, ternary_b_cb, tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                unary_bcast_init<BroadcastType::ROW>(ternary_b_cb, intermediate_cb);
                unary_bcast<BroadcastType::ROW>(ternary_b_cb, n, TERNARY_B_DST_ID);

                copy_tile_to_dst_init_short(intermediate_cb);
                copy_tile(intermediate_cb, tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb);
                tile_regs_release();
                tile_id++;
            }
        }

        cb_pop_front(ternary_b_cb, N_block_tiles);
    } else {
        // === NO BROADCAST: row-by-row, wait/pop per M row ===
#ifndef TERNARY_B_IS_FLOAT32
        mul_tiles_init(intermediate_cb, ternary_b_cb);
#endif
        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb, ternary_b_cb);
        pack_reconfig_data_format(intermediate_cb);

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            cb_wait_front(ternary_b_cb, N_block_tiles);
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles(intermediate_cb, ternary_b_cb, tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                copy_tile_to_dst_init_short(ternary_b_cb);
                copy_tile(ternary_b_cb, n, TERNARY_B_DST_ID);

                copy_tile_to_dst_init_short(intermediate_cb);
                copy_tile(intermediate_cb, tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb);
                tile_regs_release();
                tile_id++;
            }
            cb_pop_front(ternary_b_cb, N_block_tiles);
        }
    }

    cb_pop_front(intermediate_cb, out_block_num_tiles);

    // 'refill' intermediate_cb (also synchronize packer/unpacker)
    cb_reserve_back(intermediate_cb, out_block_num_tiles);
    cb_push_back(intermediate_cb, out_block_num_tiles);

    cb_wait_front(intermediate_cb, out_block_num_tiles);

    add_tiles_init(intermediate_cb, ternary_a_cb);
    reconfig_data_format(intermediate_cb, ternary_a_cb);
    pack_reconfig_data_format(out_cb);

    tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_a tiles
        cb_wait_front(ternary_a_cb, N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

            // ternary_a_cb is pushed one row at a time, so use column index n
            add_tiles(intermediate_cb, ternary_a_cb, tile_id, n, DST_ID);

            tile_regs_commit();

            tile_regs_wait();
            pack_tile(DST_ID, out_cb);
            tile_regs_release();
            tile_id++;
        }

        cb_pop_front(ternary_a_cb, N_block_tiles);
        cb_push_back(out_cb, N_block_tiles);
    }

    cb_pop_front(intermediate_cb, out_block_num_tiles);
}

// Slightly modified from compute_common.hpp
//
// m_out_tile_offset shifts where this call's rows land in the (full) output block. When in0 is
// delivered as M-row sub-chunk bands, each band's in0 CB slot is 0-based (M_block_tiles == band_h,
// in0 indexing starts at 0) but the band's results must accumulate into rows [band_lo, band_lo+band_h)
// of the intermediate block, so pass m_out_tile_offset = band_lo. Whole-block callers pass 0.
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const uint32_t m_out_tile_offset = 0) {
    DeviceZoneScopedN("MATMUL_BLOCK");

    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        // The in0 CB slot is padded to a uniform (M_block_tiles / nb) rows, so a ragged band whose
        // height is not a multiple of subblock_h still has real tiles for a full subblock_h read -- the
        // trailing rows are slack. Compute the full subblock (a smaller rt_dim would need an
        // mm_block_init_short re-init that stalls the engine), then pack only the real rows so a ragged
        // band never writes into the next band's output rows. subblock_h | (M_block_tiles/nb) (host
        // guard) keeps the deep read inside the slot.
        const uint32_t real_h = std::min(subblock_h, M_block_tiles - M_start);
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < real_h; h++) {
                uint32_t h_tile_id = M_start + h + m_out_tile_offset;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);
    // Number of leading (self/local) k-block positions this device owns. The AG schedule always drains
    // the local slice first, so positions [0, num_local_k_blocks) are local (never sub-chunked) and the
    // rest are remote (sub-chunked into IN0_SUB_CHUNKS M-row bands on the first N-block). Kept identical
    // across the dm_in0/dm_in1 senders so per-band CB push/pop counts match.
    [[maybe_unused]] constexpr uint32_t num_local_k_blocks = get_compile_time_arg_val(8);

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);

#ifdef FUSE_TERNARY
    const uint32_t fused_ternary_scalar_uint = get_arg_val<uint32_t>(argidx++);
    const uint32_t broadcast_ternary_b = get_arg_val<uint32_t>(argidx++);
#else
    // Default value when ternary is not fused (not used, helps compiler optimize)
    constexpr uint32_t fused_ternary_scalar_uint = 0;
    constexpr uint32_t broadcast_ternary_b = 1;
#endif

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    constexpr uint32_t in2_cb = tt::CBIndex::c_4;
    constexpr uint32_t ternary_a_cb = tt::CBIndex::c_5;
    constexpr uint32_t ternary_b_cb = tt::CBIndex::c_6;

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    mm_init(in0_cb, in1_cb, intermediate_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    uint32_t current_M_block_tiles = M_block_tiles;
    uint32_t current_N_block_tiles = N_block_tiles;
    uint32_t current_subblock_h = subblock_h;
    uint32_t current_subblock_w = subblock_w;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        current_M_block_tiles = m_tile_end - m_tile;
        current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            current_N_block_tiles = n_tile_end - n_tile;
            current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            mm_block_init_short(
                in0_cb,
                in1_cb,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermediate_cb);
            // Accumulation buffer
            cb_reserve_back(intermediate_cb, out_block_num_tiles);
            // Sub-chunked (banded) matmul. Remote k-blocks on the first N-block arrive as IN0_SUB_CHUNKS
            // M-row bands; local k-blocks and every k-block on later N-blocks arrive whole (nb == 1). in1
            // is re-presented once per band (read once, pushed nb times by the in1 sender), so in0 and in1
            // advance in lockstep and the matmul fires per band. Each k-block position is delivered fresh
            // (no N-stride in0 reuse). With IN0_SUB_CHUNKS == 1 this degenerates to one whole band per
            // position.
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                const uint32_t nb =
                    (n_block_iter == 0 && k_block >= num_local_k_blocks) ? (uint32_t)IN0_SUB_CHUNKS : 1u;
#ifdef AG_INTERLEAVE_BANDS
                // Interleave a forward remote k-block with the following backward one at band granularity:
                // A.b0, B.b0, A.b1, B.b1, ... Both members share band_lo and accumulate into the same
                // output rows; both are past k-block 0 so L1 accumulation is already on, so the two partials
                // sum correctly regardless of order. The position-based pairing matches dm_in0/dm_in1 so the
                // per-band in0/in1 CB counts stay in lockstep.
                if (n_block_iter == 0 && nb > 1 && k_block >= num_local_k_blocks && (k_block + 1) < K_num_blocks &&
                    ((k_block - num_local_k_blocks) & 1u) == 0) {
                    for (uint32_t band = 0; band < nb; band++) {
                        uint32_t band_lo, band_h;
                        balanced_band(current_M_block_tiles, nb, band, band_lo, band_h);
                        if (band_h == 0) {
                            break;
                        }
                        // Reserve a uniform slot of M_block_tiles/nb rows for every band (not the ragged
                        // band_h) so each band tiles the in0 CB exactly and no reservation wraps
                        // fifo_limit mid-block. matmul reads band_h rows and packs the real ones.
                        const uint32_t band_slot_tiles = (M_block_tiles / nb) * K_block_tiles;
                        for (uint32_t member = 0; member < 2; member++) {
                            cb_wait_front(in0_cb, band_slot_tiles);
                            cb_wait_front(in1_cb, in1_block_num_tiles);
                            matmul_blocks(
                                in0_cb,
                                in1_cb,
                                intermediate_cb,
                                band_h,
                                current_N_block_tiles,
                                N_block_tiles,
                                K_block_tiles,
                                current_subblock_h,
                                current_subblock_w,
                                band_lo);
                            cb_pop_front(in0_cb, band_slot_tiles);
                            cb_pop_front(in1_cb, in1_block_num_tiles);
                        }
                    }
                    k_block++;  // the backward member of the pair is consumed here too
                    continue;
                }
#endif
                for (uint32_t band = 0; band < nb; band++) {
                    uint32_t band_lo, band_h;
                    balanced_band(current_M_block_tiles, nb, band, band_lo, band_h);
                    if (band_h == 0) {
                        break;  // only when nb > current_M_block_tiles
                    }
                    // Reserve a uniform slot of M_block_tiles/nb rows for every band (not the ragged
                    // band_h) so each band tiles the in0 CB exactly and no reservation wraps fifo_limit
                    // mid-block. matmul reads band_h rows and packs the real ones.
                    const uint32_t band_slot_tiles = (M_block_tiles / nb) * K_block_tiles;
                    cb_wait_front(in0_cb, band_slot_tiles);
                    cb_wait_front(in1_cb, in1_block_num_tiles);

                    matmul_blocks(
                        in0_cb,
                        in1_cb,
                        intermediate_cb,
                        band_h,
                        current_N_block_tiles,
                        N_block_tiles,
                        K_block_tiles,
                        current_subblock_h,
                        current_subblock_w,
                        band_lo);

                    cb_pop_front(in0_cb, band_slot_tiles);
                    cb_pop_front(in1_cb, in1_block_num_tiles);
                }
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

            cb_reserve_back(out_cb, out_block_num_tiles);
#ifndef FUSE_TERNARY
            cb_wait_front(intermediate_cb, out_block_num_tiles);
#ifndef FUSE_BIAS
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#else
            cb_wait_front(in2_cb, N_block_tiles);
            add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_pop_front(in2_cb, N_block_tiles);
#endif  // FUSE_BIAS
            cb_pop_front(intermediate_cb, out_block_num_tiles);

#else   // FUSE_TERNARY is set
            add_bias_and_addcmul_block(
                intermediate_cb,
                in2_cb,
                ternary_a_cb,
                ternary_b_cb,
                fused_ternary_scalar_uint,
                out_cb,
                M_block_tiles,
                N_block_tiles,
                broadcast_ternary_b);
#endif  // FUSE_TERNARY
        }
    }
}
