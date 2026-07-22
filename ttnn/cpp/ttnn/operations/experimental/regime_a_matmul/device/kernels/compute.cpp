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

// Test-only causal timing zones (compile-gated; mask 0 => no-op => byte-identical). See DIAG_ZONES.
// SumN{1,2} accumulate a repeated region's total time across the k-loop into one zone (2 accumulator slots).
// The compute/TRISC kernel does NOT auto-inject the profiler header (dataflow kernels do), so include it.
#ifdef DIAG_ZONES
#include "tools/profiler/kernel_profiler.hpp"
// Regular start/end zones (SumN accumulators do not flush to the CSV for the compute kernel). K_num_blocks
// is small so per-k-block markers are cheap; the analysis SUMS the per-instance durations to get total wait.
#define RA_ZONE_SUM1(n) DeviceZoneScopedN(n)
#define RA_ZONE_SUM2(n) DeviceZoneScopedN(n)
#else
#define RA_ZONE_SUM1(n)
#define RA_ZONE_SUM2(n)
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

// Like copy_block but NEVER applies the fused activation. Used by split-K NON-root bands (bottom band of a
// Pk>1 chain) so they forward the RAW matmul partial up the reduction chain; the activation is applied
// exactly once at the reduction ROOT (is_top). copy_block itself always applies activation when defined and
// is used only where the block IS the final output (no-fusion path, or activation-only root).
void copy_block_raw(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            copy_tile(in_cb, tile_id, 0 /*dst*/);
            pack_tile(0, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_block_tiles);
    }
}

// Split-K fusion root helper: intermediate_cb += reduce_cb, IN PLACE (result stays in intermediate_cb so the
// existing bias/activation/addcmul primitives can consume it exactly as on the non-split-K path). Uses the
// single-slot in-place refill idiom (wait/pop front, reserve/push back the same slot) already used by
// add_bias_and_addcmul_block. Applies NO fusion — bias/activation/addcmul are applied afterward, once.
void reduce_add_in_place(uint32_t intermediate_cb, uint32_t reduce_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    const uint32_t n_tiles = M_block_tiles * N_block_tiles;
    add_tiles_init(intermediate_cb, reduce_cb);
    reconfig_data_format(intermediate_cb, reduce_cb);
    pack_reconfig_data_format(intermediate_cb);
    cb_wait_front(intermediate_cb, n_tiles);
    for (uint32_t t = 0; t < n_tiles; t++) {
        acquire_dst();
        add_tiles(intermediate_cb, reduce_cb, t, t, 0 /*dst*/);
        pack_tile(0, intermediate_cb);
        release_dst();
    }
    cb_pop_front(intermediate_cb, n_tiles);
    cb_reserve_back(intermediate_cb, n_tiles);
    cb_push_back(intermediate_cb, n_tiles);
}

// Split-K plan B: out = a + b, full elementwise (both M_block_tiles x N_block_tiles). Used by the
// column reduction to add this band's matmul partial (a) to the running sum forwarded up from the band
// below (b). Pushes out_cb one M-row at a time, matching copy_block/add_bias_block.
void reduce_add_block(uint32_t a_cb, uint32_t b_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    add_tiles_init(a_cb, b_cb);
    reconfig_data_format(a_cb, b_cb);
    pack_reconfig_data_format(out_cb);
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            acquire_dst();
            add_tiles(a_cb, b_cb, tile_id, tile_id, 0 /*dst*/);
            pack_tile(0, out_cb);
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

#ifdef RSCATTER
// Ring reduce-scatter helpers (test-only). Copy n_tiles from in_cb starting at tile in_tile_off -> out_cb
// (used to seed the ring with this core's own row-chunk). No cb push/pop (caller manages).
void rs_copy_chunk(uint32_t in_cb, uint32_t in_tile_off, uint32_t out_cb, uint32_t n_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, in_tile_off + i, 0);
        pack_tile(0, out_cb);
        release_dst();
    }
}
// out_cb[0..n] = acc_cb[acc_tile_off..] + add_cb[0..n]. acc_cb is the resident fp32 matmul partial (read at a
// row offset); add_cb is the received bf16 chunk. Adds this core's own row into the running ring sum.
void rs_add_chunk(uint32_t acc_cb, uint32_t acc_tile_off, uint32_t add_cb, uint32_t out_cb, uint32_t n_tiles) {
    add_tiles_init(acc_cb, add_cb);
    reconfig_data_format(acc_cb, add_cb);
    pack_reconfig_data_format(out_cb);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        acquire_dst();
        add_tiles(acc_cb, add_cb, acc_tile_off + i, i, 0);
        pack_tile(0, out_cb);
        release_dst();
    }
}
#endif

// Slightly modified from compute_common.hpp
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
    const uint32_t in0_base = 0,
    const uint32_t in1_base = 0) {
    uint32_t in0_index_offset = in0_base;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = in1_base;
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
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                    dst_index++;
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

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    // split-K plan B: 1 if this is the bottom K-band (no incoming running sum), else 0. Always present.
    [[maybe_unused]] const uint32_t is_reduce_bottom = get_arg_val<uint32_t>(argidx++);
#ifdef REDTREE
    // Fan-in-2 reduction tree (test-only): number of incoming partials to sum before this core's block is
    // final (0 = leaf/copy, 1 = one add, 2 = root two adds). Follows is_reduce_bottom; never combined with
    // fusion, so it cannot collide with the fused is_reduce_top read below.
    const uint32_t tree_num_recv = get_arg_val<uint32_t>(argidx++);
#endif
#ifdef RSCATTER
    // Ring reduce-scatter (test-only): this core's ring position, the ring size P=Pk, and chunk_tiles (tiles
    // per chunk = M_block*N_block / Pk). Follow is_reduce_bottom; unfused only (no collision with is_reduce_top).
    const uint32_t rs_ring_pos = get_arg_val<uint32_t>(argidx++);
    const uint32_t rs_P = get_arg_val<uint32_t>(argidx++);
    const uint32_t rs_chunk_tiles = get_arg_val<uint32_t>(argidx++);
#endif

// Any fusion active => the reduction ROOT (is_top) applies bias/activation/addcmul exactly once after the
// split-K partials are summed. Non-root bands forward the RAW partial (no fusion). When no fusion is active
// this macro is undefined and the output stage is byte-identical to the historical no-fusion path.
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY) || defined(SFPU_OP_INIT_ACTIVATION)
#define REGIME_A_FUSED 1
    // is_top: 1 on the reduction root (Pk==1 => every core; Pk>1 => the top K-band). Present only when fused.
    [[maybe_unused]] const uint32_t is_reduce_top = get_arg_val<uint32_t>(argidx++);
#endif

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
    constexpr uint32_t cb_reduce = tt::CBIndex::c_7;  // split-K B: running sum forwarded up from the band below

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    mm_init(in0_cb, in1_cb, intermediate_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    bool reuse_in0_block = false;

    uint32_t current_M_block_tiles = M_block_tiles;
    uint32_t current_N_block_tiles = N_block_tiles;
    uint32_t current_subblock_h = subblock_h;
    uint32_t current_subblock_w = subblock_w;

#ifdef IN0_KSLICE_RESIDENT
    // Large-Mt ring: cb0 holds the full k-slice (all K_num_blocks in0 blocks, block-major). It is filled
    // ONCE (the writer pushes it incrementally, W compute blocks per ring step) and kept resident so it can
    // be reused across the N_blocks_per_core N-sub-blocks; the k-loop addresses block k_block via the in0_base
    // tile offset instead of popping. Popped once at the end.
    //
    // DEFAULT (progressive): NO startup barrier. The first N-sub-block's k-loop instead waits cumulatively
    // per K block (below), so the first matmul begins as soon as the first ring shard arrives while ring
    // forwarding + in1 reading continue concurrently. Later N-sub-blocks reuse the now-complete resident slice.
#ifdef DIAG_FULL_IN0_WAIT
    // A/B baseline (test-only): the OLD single full-slice startup barrier before any matmul. Exactly the
    // previous production behavior; the per-K cumulative waits below are compiled out.
    cb_wait_front(in0_cb, K_num_blocks * in0_block_num_tiles);
#endif
#endif

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
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
#ifndef IN0_KSLICE_RESIDENT
                cb_wait_front(in0_cb, in0_block_num_tiles);
#else
#ifndef DIAG_FULL_IN0_WAIT
                // Progressive cumulative wait (default resident path): begin matmul k_block as soon as the
                // writer's incremental per-step ring pushes make CUMULATIVE (k_block+1) in0 blocks available.
                // Only the FIRST resident traversal waits (M_blocks_per_core==1, first N-sub-block); once the
                // slice is complete it stays resident and later N-sub-blocks reuse it with no further waits.
                // Repeated cb_wait_front WITHOUT an intervening pop requires cumulative counts (CB API
                // contract) — satisfied since (k_block+1)*in0_block_num_tiles is strictly increasing and CB0
                // is popped only once, after all reuse (below). The writer pushes W blocks at a time, so a
                // wait for a mid-batch boundary is simply satisfied when that W-batch lands.
                if (m_block_iter == 0 && n_block_iter == 0) {
                    RA_ZONE_SUM1("Z_C_IN0WAIT");  // compute stalled on progressive in0 (ring exposure to compute)
                    cb_wait_front(in0_cb, (k_block + 1) * in0_block_num_tiles);
                }
#endif
#endif
                {
                    RA_ZONE_SUM2("Z_C_IN1WAIT");  // compute stalled on in1 delivery (read exposure to compute)
                    cb_wait_front(in1_cb, in1_block_num_tiles);
                }

                matmul_blocks(
                    in0_cb,
                    in1_cb,
                    intermediate_cb,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w
#ifdef IN0_KSLICE_RESIDENT
                    ,
                    k_block * in0_block_num_tiles  // block-major offset into the resident k-slice
#endif
                );

                if (k_block == K_num_blocks - 1) {
                    /**
                     * On next iteration we might get reuse on in0
                     *
                     */
                    if (n_block_iter < N_blocks_per_core - 1) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
                    }
                }
#ifndef IN0_KSLICE_RESIDENT
                if (!reuse_in0_block) {
                    cb_pop_front(in0_cb, in0_block_num_tiles);
                }
#endif
                cb_pop_front(in1_cb, in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

#ifndef RSCATTER
            cb_reserve_back(out_cb, out_block_num_tiles);
#endif
#ifdef REDUCE_K
#ifdef RSCATTER
            // Ring reduce-scatter (test-only, no-fusion). intermediate_cb (fp32, resident) = my matmul partial,
            // row-major. Partitioned into P=Pk contiguous tile-chunks of ct = out_block_num_tiles/Pk tiles
            // (chunk c = tiles [c*ct, (c+1)*ct)). Seed cb_send with my own chunk `rs_ring_pos`; over P-1 rounds
            // receive chunk d=(rs_ring_pos-t-1)%P, add my resident chunk d, and forward the running sum (rounds
            // < P-2) or write my fully-reduced OWNED chunk (last round) -> out_cb.
            {
                const uint32_t P = rs_P;                              // Pk
                const uint32_t ct = rs_chunk_tiles;                   // tiles per chunk = out_block_num_tiles / Pk
                constexpr uint32_t cb_send_cb = tt::CBIndex::c_4;     // compute -> writer send-chunk CB (bf16)
                constexpr uint32_t cb_recv_cb = tt::CBIndex::c_5;     // incoming-chunk CB (bf16), 2 slots
                cb_wait_front(intermediate_cb, out_block_num_tiles);  // resident; popped after the ring
                cb_reserve_back(cb_send_cb, ct);                      // seed: my own chunk `rs_ring_pos`
                rs_copy_chunk(intermediate_cb, rs_ring_pos * ct, cb_send_cb, ct);
                cb_push_back(cb_send_cb, ct);
                for (uint32_t t = 0; t + 1u < P; ++t) {
                    const uint32_t d = (rs_ring_pos + P - t - 1u) % P;  // chunk received + reduced this round
                    cb_wait_front(cb_recv_cb, ct);
                    if (t + 1u < P - 1u) {  // forward the running sum to next round's send
                        cb_reserve_back(cb_send_cb, ct);
                        rs_add_chunk(intermediate_cb, d * ct, cb_recv_cb, cb_send_cb, ct);
                        cb_push_back(cb_send_cb, ct);
                    } else {  // last round: this is my fully-reduced OWNED chunk -> writer writes it to DRAM
                        cb_reserve_back(out_cb, ct);
                        rs_add_chunk(intermediate_cb, d * ct, cb_recv_cb, out_cb, ct);
                        cb_push_back(out_cb, ct);
                    }
                    cb_pop_front(cb_recv_cb, ct);
                }
                cb_pop_front(intermediate_cb, out_block_num_tiles);
            }
#else  // non-RSCATTER REDUCE_K branches (chain / NO_REDUCE / REDTREE / fused) share the intermediate wait
       // Split-K plan B column reduction: bottom band emits its own matmul partial; every other band
       // adds the running sum forwarded up from the band below. The DM then either forwards out_cb up
       // (non-top bands) or writes it to DRAM (top band).
            cb_wait_front(intermediate_cb, out_block_num_tiles);
#ifdef DIAG_NO_REDUCE
            // NO_REDUCE ablation: force the bottom-band copy path on EVERY core so it never waits for or adds
            // cb_reduce. The writer bypasses the matching reduction traffic; only the top band writes.
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_pop_front(intermediate_cb, out_block_num_tiles);
#elif defined(REDTREE)
            // Fan-in-2 reduction tree (test-only, no-fusion). num_recv rounds of reduce-add. Leaf (num_recv==0)
            // forwards its own partial (copy_block); each round r<num_recv-1 accumulates in place; the final
            // round writes intermediate + cb_reduce -> out_cb. num_recv==1 matches the chain non-bottom path;
            // num_recv==0 the chain bottom path. (Tree reassociates the sum -> PCC-preserving, not bit-exact.)
            if (tree_num_recv == 0u) {
                copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            } else {
                for (uint32_t r = 0; r + 1u < tree_num_recv; ++r) {
                    cb_wait_front(cb_reduce, out_block_num_tiles);
                    reduce_add_in_place(intermediate_cb, cb_reduce, M_block_tiles, N_block_tiles);
                    cb_pop_front(cb_reduce, out_block_num_tiles);
                }
                cb_wait_front(cb_reduce, out_block_num_tiles);
                reduce_add_block(intermediate_cb, cb_reduce, out_cb, M_block_tiles, N_block_tiles);
                cb_pop_front(cb_reduce, out_block_num_tiles);
            }
            cb_pop_front(intermediate_cb, out_block_num_tiles);
#elif !defined(REGIME_A_FUSED)
            // NO-FUSION path (byte-identical to the historical Regime-A output stage): top and non-top reduce
            // bands are identical; the writer decides forward-up vs DRAM-write.
            if (is_reduce_bottom) {
                copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            } else {
                cb_wait_front(cb_reduce, out_block_num_tiles);
                reduce_add_block(intermediate_cb, cb_reduce, out_cb, M_block_tiles, N_block_tiles);
                cb_pop_front(cb_reduce, out_block_num_tiles);
            }
            cb_pop_front(intermediate_cb, out_block_num_tiles);
#else
            // FUSION-AWARE split-K: bias/activation/addcmul are applied EXACTLY ONCE at the reduction ROOT
            // (is_reduce_top). Non-root bands forward the RAW partial (no fusion), so the reduction chain sums
            // un-fused partials and the epilogue sees the true A@B (+ reduced K) before bias/act/addcmul.
            if (!is_reduce_top) {
                // Partial-forwarding band. RAW (copy_block_raw / reduce_add_block never apply activation).
                if (is_reduce_bottom) {
                    copy_block_raw(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
                } else {
                    cb_wait_front(cb_reduce, out_block_num_tiles);
                    reduce_add_block(intermediate_cb, cb_reduce, out_cb, M_block_tiles, N_block_tiles);
                    cb_pop_front(cb_reduce, out_block_num_tiles);
                }
                cb_pop_front(intermediate_cb, out_block_num_tiles);
            } else {
                // Reduction ROOT (Pk==1: every core; Pk>1: top band). Sum the forwarded partial into
                // intermediate (in place) when this is a Pk>1 chain, then apply the epilogue ONCE -> out_cb.
                if (!is_reduce_bottom) {
                    cb_wait_front(cb_reduce, out_block_num_tiles);
                    reduce_add_in_place(intermediate_cb, cb_reduce, M_block_tiles, N_block_tiles);
                    cb_pop_front(cb_reduce, out_block_num_tiles);
                }
#if defined(FUSE_TERNARY)
                add_bias_and_addcmul_block(
                    intermediate_cb,
                    in2_cb,
                    ternary_a_cb,
                    ternary_b_cb,
                    fused_ternary_scalar_uint,
                    out_cb,
                    M_block_tiles,
                    N_block_tiles,
                    broadcast_ternary_b);  // pops intermediate_cb internally
#elif defined(FUSE_BIAS)
                cb_wait_front(in2_cb, N_block_tiles);
                add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);  // + activation
                cb_pop_front(in2_cb, N_block_tiles);
                cb_pop_front(intermediate_cb, out_block_num_tiles);
#else   // activation-only root
                copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);  // applies SFPU activation
                cb_pop_front(intermediate_cb, out_block_num_tiles);
#endif  // fusion kind
            }
#endif  // DIAG_NO_REDUCE / REDTREE / no-fusion chain / fused
#endif  // RSCATTER vs non-RSCATTER
#elif !defined(FUSE_TERNARY)
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
#ifdef IN0_KSLICE_RESIDENT
    cb_pop_front(in0_cb, K_num_blocks * in0_block_num_tiles);
#endif
}
