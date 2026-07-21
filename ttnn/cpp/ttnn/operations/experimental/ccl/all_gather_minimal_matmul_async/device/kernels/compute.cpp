// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers_advanced.hpp"

void copy_block(CircularBuffer& in_cb, CircularBuffer& out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    copy_tile_to_dst_init_short(in_cb.get_cb_id());
    reconfig_data_format_srca(in_cb.get_cb_id());
    pack_reconfig_data_format(out_cb.get_cb_id());
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            copy_tile(in_cb.get_cb_id(), tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(fused_act_dst_id, out_cb.get_cb_id());
            tile_regs_release();
            tile_id++;
        }
        out_cb.push_back(N_block_tiles);
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
void add_bias_block(
    CircularBuffer& in_cb,
    CircularBuffer& bias_cb,
    CircularBuffer& out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles) {
    add_bcast_rows_init_short(in_cb.get_cb_id(), bias_cb.get_cb_id());
    reconfig_data_format(in_cb.get_cb_id(), bias_cb.get_cb_id());
    pack_reconfig_data_format(out_cb.get_cb_id());
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(
                in_cb.get_cb_id(), bias_cb.get_cb_id(), tile_id, n, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(fused_act_dst_id, out_cb.get_cb_id());
            tile_regs_release();
            tile_id++;
        }
        out_cb.push_back(N_block_tiles);
    }
}

void add_bias_and_addcmul_block(
    CircularBuffer& intermediate_cb,
    CircularBuffer& bias_cb,
    CircularBuffer& ternary_a_cb,
    CircularBuffer& ternary_b_cb,
    uint32_t scalar_value,
    CircularBuffer& out_cb,
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

    add_bcast_rows_init_short(intermediate_cb.get_cb_id(), bias_cb.get_cb_id());
    reconfig_data_format(intermediate_cb.get_cb_id(), bias_cb.get_cb_id());
    pack_reconfig_data_format(intermediate_cb.get_cb_id());

    // Wait for ALL input data ONCE at the beginning
    bias_cb.wait_front(N_block_tiles);

    // Unpacker waits for intermediate_cb to be ready
    intermediate_cb.wait_front(out_block_num_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();
            add_tiles_bcast<BroadcastType::ROW>(intermediate_cb.get_cb_id(), bias_cb.get_cb_id(), tile_id, n, DST_ID);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_ID, intermediate_cb.get_cb_id());
            tile_regs_release();
        }
    }

    // Pop input and push output ONCE at the end
    // intermediate_cb.wait_front(out_block_num_tiles); // Unpacker-Packer sync
    // intermediate_cb.pop_front(out_block_num_tiles);
    bias_cb.pop_front(N_block_tiles);

    intermediate_cb.pop_front(out_block_num_tiles);

    // Restore intermediate_cb to ready (+ sync packer/unpacker)
    intermediate_cb.reserve_back(out_block_num_tiles);
    intermediate_cb.push_back(out_block_num_tiles);
#endif  // FUSE_BIAS

    // ============================================
    // STEP 2: Multiply by ternary_b and scalar
    // Read from intermediate_cb and write back to intermediate_cb
    // broadcast_ternary_b: 1 = single row broadcast, 0 = row-by-row streaming
    // ============================================

    intermediate_cb.wait_front(out_block_num_tiles);

    uint32_t tile_id = 0;

    if (broadcast_ternary_b) {
        // === BROADCAST: single row, wait/pop once ===
        ternary_b_cb.wait_front(N_block_tiles);

#ifndef TERNARY_B_IS_FLOAT32
        mul_bcast_rows_init_short(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
#else
        unary_bcast_init<BroadcastType::ROW>(ternary_b_cb.get_cb_id(), intermediate_cb.get_cb_id());
#endif  // TERNARY_B_IS_FLOAT32

        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
        pack_reconfig_data_format(intermediate_cb.get_cb_id());

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                // LLK BUG: unary_bcast gives bad values if mixing fp32_acc_to_dest=True and bfloat16 circular buffer
                // (https://github.com/tenstorrent/tt-llk/issues/1338)
                // To avoid the bug, we use:
                // - unary_bcast/mul_binary_tile for fp32 (more accurate)
                // - mul_tiles_bcast for bfloat16 (LLK bug workaround).

                // ternary_b_cb is [1, N], broadcast across M rows
                mul_tiles_bcast<BroadcastType::ROW>(
                    intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id(), tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                unary_bcast_init<BroadcastType::ROW>(ternary_b_cb.get_cb_id(), intermediate_cb.get_cb_id());
                unary_bcast<BroadcastType::ROW>(ternary_b_cb.get_cb_id(), n, TERNARY_B_DST_ID);

                copy_tile_to_dst_init_short(intermediate_cb.get_cb_id());
                copy_tile(intermediate_cb.get_cb_id(), tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb.get_cb_id());
                tile_regs_release();
                tile_id++;
            }
        }

        ternary_b_cb.pop_front(N_block_tiles);
    } else {
        // === NO BROADCAST: row-by-row, wait/pop per M row ===
#ifndef TERNARY_B_IS_FLOAT32
        mul_tiles_init(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
#endif
        binop_with_scalar_tile_init();
        reconfig_data_format(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id());
        pack_reconfig_data_format(intermediate_cb.get_cb_id());

        tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            ternary_b_cb.wait_front(N_block_tiles);
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();

#ifndef TERNARY_B_IS_FLOAT32
                mul_tiles(intermediate_cb.get_cb_id(), ternary_b_cb.get_cb_id(), tile_id, n, DST_ID);
#else
                constexpr uint32_t TERNARY_B_DST_ID = 1;
                copy_tile_to_dst_init_short(ternary_b_cb.get_cb_id());
                copy_tile(ternary_b_cb.get_cb_id(), n, TERNARY_B_DST_ID);

                copy_tile_to_dst_init_short(intermediate_cb.get_cb_id());
                copy_tile(intermediate_cb.get_cb_id(), tile_id, DST_ID);

                mul_binary_tile_init();
                mul_binary_tile(DST_ID, TERNARY_B_DST_ID, DST_ID);
#endif  // TERNARY_B_IS_FLOAT32

                mul_unary_tile(DST_ID, scalar_value);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(DST_ID, intermediate_cb.get_cb_id());
                tile_regs_release();
                tile_id++;
            }
            ternary_b_cb.pop_front(N_block_tiles);
        }
    }

    intermediate_cb.pop_front(out_block_num_tiles);

    // 'refill' intermediate_cb (also synchronize packer/unpacker)
    intermediate_cb.reserve_back(out_block_num_tiles);
    intermediate_cb.push_back(out_block_num_tiles);

    intermediate_cb.wait_front(out_block_num_tiles);

    add_tiles_init(intermediate_cb.get_cb_id(), ternary_a_cb.get_cb_id());
    reconfig_data_format(intermediate_cb.get_cb_id(), ternary_a_cb.get_cb_id());
    pack_reconfig_data_format(out_cb.get_cb_id());

    tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_a tiles
        ternary_a_cb.wait_front(N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

            // ternary_a_cb is pushed one row at a time, so use column index n
            add_tiles(intermediate_cb.get_cb_id(), ternary_a_cb.get_cb_id(), tile_id, n, DST_ID);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(DST_ID, out_cb.get_cb_id());
            tile_regs_release();
            tile_id++;
        }

        ternary_a_cb.pop_front(N_block_tiles);
        out_cb.push_back(N_block_tiles);
    }

    intermediate_cb.pop_front(out_block_num_tiles);
}

// The pre-migration kernel had a private `matmul_blocks(...)` that walked an
// in0/in1 K-block, packed each output sub-block via per-tile pack_tile<true> at
// absolute offsets, and the kernel below wrapped that in a K-loop with
// L1_ACC accumulation into a single pre-reserved intermediate buffer. The K-loop
// helper now covers all of that via:
//   LastBlockTarget::Interm + OutputCBLayout::TileRowMajor → pack_subblock_row_major_strided
//     packs each subblock at absolute row positions inside the row group;
//   packer_l1_acc + pack_last_to_interm → llk_pack_reconfig_l1_acc(0 → 1) per K-block
//     so block 0 writes fresh and blocks 1..N-1 accumulate into the same L1 cells;
//   (TileRowMajor + packer_l1_acc + Interm) → the helper packs in place: it does ONE
//     internal reserve_back before the K-loop and ONE push_back after (plus the final
//     pack_reconfig_l1_acc(0)), skipping its own per-block reserve/push/drain. The caller
//     therefore does no reserve/push of its own around the helper call.
// in0_policy controls the outer-loop in0 reuse: when reusing across the next
// n_block_iter we keep in0 fronted on the last K-block (WaitAndRetainOnLastBlock);
// on the final n iter we pop it to free the slot (WaitAndPopPerKBlock). The
// runtime if-else picks between two helper template instantiations (one per
// in0_policy value).

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

#ifdef FUSE_TERNARY
    const uint32_t fused_ternary_scalar_uint = get_common_arg_val<uint32_t>(0);
    const uint32_t broadcast_ternary_b = get_common_arg_val<uint32_t>(1);
#else
    // Default value when ternary is not fused (not used, helps compiler optimize)
    constexpr uint32_t fused_ternary_scalar_uint = 0;
    constexpr uint32_t broadcast_ternary_b = 1;
#endif

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in2_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t ternary_a_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t ternary_b_cb_id = tt::CBIndex::c_6;

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);
    CircularBuffer intermediate_cb(intermediate_cb_id);
    CircularBuffer in2_cb(in2_cb_id);
    CircularBuffer ternary_a_cb(ternary_a_cb_id);
    CircularBuffer ternary_b_cb(ternary_b_cb_id);

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    using namespace compute_kernel_lib;

    // Boot-time matmul init: compute_kernel_hw_startup does the one hw_configure MMIO, then
    // matmul_block_init sets up unpack/math matmul state (mm_block_init is deprecated). The helper
    // invocation below uses InitMode::None, so the per-(m,n) matmul_block_init below is the only
    // re-init that fires.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, intermediate_cb_id);
    matmul_block_init(in0_cb_id, in1_cb_id, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

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

            matmul_block_init(
                in0_cb_id,
                in1_cb_id,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            reconfig_data_format(in1_cb_id, in0_cb_id);
            pack_reconfig_data_format(intermediate_cb_id);
            // The (TileRowMajor + packer_l1_acc + Interm) matmul_block config packs in place: the
            // helper does its own single reserve_back/push_back over the whole output block and
            // manages the per-K-block L1_ACC accumulation, so the caller does no reserve/push here.

            const uint32_t in0_subblocks = current_M_block_tiles / current_subblock_h;
            const uint32_t in1_subblocks = current_N_block_tiles / current_subblock_w;
            const auto shape = MatmulBlockShape::of(
                in0_subblocks,
                in1_subblocks,
                current_subblock_h,
                current_subblock_w,
                K_block_tiles,
                K_num_blocks,
                /*batch=*/1,
                /*in1_per_core_w=*/N_block_tiles,
                /*out_row_width=*/N_block_tiles);

            // in0_policy selection: WaitAndRetainOnLastBlock when reusing in0 across
            // the next n iter (helper skips popping in0 on the last K-block);
            // WaitAndPopPerKBlock on the last n iter so the slot is freed for the
            // next m iter.
            //
            // #44982: in0-reuse is Ring-only. Linear keeps k_forward fixed, so reusing
            // would feed the previous iter's last K-block (= K_num_blocks-1) when the
            // new iter wants K-block 0 — force a fresh read on Linear (always pop in0).
#ifndef IS_LINEAR
            if (n_block_iter < N_blocks_per_core - 1) {
                matmul_block_gathered<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    LastBlockTarget::Interm,
                    OutputCBLayout::TileRowMajor,
                    matmul_config::InitMode::None,
                    InputPolicy::WaitAndRetainOnLastBlock,
                    InputPolicy::WaitAndPopPerKBlock,
                    matmul_config::DataFormatReconfig::InputAndOutput,  // reconfig (was defaulted)
                    NoneActivation,                                     // Activation (was defaulted)
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    NoKBlockInnerDimFn,
                    NoIn0Source,
                    NoIn1BaseOffset>(
                    in0_cb,
                    in1_cb,
                    intermediate_cb,
                    intermediate_cb,
                    shape,
                    NoPostCompute{},
                    NoPreKBlock{},
                    NoPostKBlock{},
                    NoKBlockInnerDimFn{},
                    NoIn0Source{},
                    NoIn1BaseOffset{});
            } else
#endif
            {
                matmul_block_gathered<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    LastBlockTarget::Interm,
                    OutputCBLayout::TileRowMajor,
                    matmul_config::InitMode::None,
                    InputPolicy::WaitAndPopPerKBlock,
                    InputPolicy::WaitAndPopPerKBlock,
                    matmul_config::DataFormatReconfig::InputAndOutput,  // reconfig (was defaulted)
                    NoneActivation,                                     // Activation (was defaulted)
                    NoPostCompute,
                    NoPreKBlock,
                    NoPostKBlock,
                    NoKBlockInnerDimFn,
                    NoIn0Source,
                    NoIn1BaseOffset>(
                    in0_cb,
                    in1_cb,
                    intermediate_cb,
                    intermediate_cb,
                    shape,
                    NoPostCompute{},
                    NoPreKBlock{},
                    NoPostKBlock{},
                    NoKBlockInnerDimFn{},
                    NoIn0Source{},
                    NoIn1BaseOffset{});
            }

            // The helper did the single push_back + pack_reconfig_l1_acc(0) for the accumulated
            // output block itself (TileRowMajor + packer_l1_acc + Interm), so the caller does neither.

            out_cb.reserve_back(out_block_num_tiles);
#ifndef FUSE_TERNARY
            intermediate_cb.wait_front(out_block_num_tiles);
#ifndef FUSE_BIAS
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#else
            in2_cb.wait_front(N_block_tiles);
            add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
            in2_cb.pop_front(N_block_tiles);
#endif  // FUSE_BIAS
            intermediate_cb.pop_front(out_block_num_tiles);

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
