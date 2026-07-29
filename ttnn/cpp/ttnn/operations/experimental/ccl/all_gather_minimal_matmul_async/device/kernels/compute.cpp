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

#ifdef FUSE_SWIGLU
// Fused SwiGLU output stage. The matmul produced an interleaved gate/up block in `in_cb`:
// within each M row, column tile 2p is the gate projection and 2p+1 the up projection (the
// weight was tile-pair interleaved on the host). Each pair emits one output tile = silu(gate)*up,
// so the block shrinks from N_block_tiles to N_block_tiles/2 along N. No extra CB / DRAM
// round-trip. N_block_tiles must be even (enforced host-side). silu and mul_binary are distinct
// SFPU programs, so each is re-initialised right before use.
//
// With FUSE_BIAS: bias is interleaved identically (tile 2p = gate bias, 2p+1 = up bias)
// and added via row-broadcast before silu/mul: out = silu(gate + bias_gate) * (up + bias_up).
void swiglu_block(uint32_t in_cb, uint32_t bias_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
#ifdef FUSE_BIAS
    reconfig_data_format(in_cb, bias_cb);
#else
    reconfig_data_format_srca(in_cb);
#endif
    pack_reconfig_data_format(out_cb);

    constexpr uint32_t GATE_DST = 0;
    constexpr uint32_t UP_DST = 1;
    const uint32_t out_N_block_tiles = N_block_tiles >> 1;

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        const uint32_t row_base = m * N_block_tiles;
        for (uint32_t p = 0; p < out_N_block_tiles; p++) {
            const uint32_t gate_n = p << 1;
            const uint32_t up_n = gate_n + 1;
            const uint32_t gate_tile_id = row_base + gate_n;
            const uint32_t up_tile_id = gate_tile_id + 1;

            tile_regs_acquire();
#ifdef FUSE_BIAS
            add_bcast_rows_init_short(in_cb, bias_cb);
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, gate_tile_id, gate_n, GATE_DST);
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, up_tile_id, up_n, UP_DST);
#else
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, gate_tile_id, GATE_DST);
            copy_tile(in_cb, up_tile_id, UP_DST);
#endif
            silu_tile_init();
            silu_tile(GATE_DST);
            mul_binary_tile_init();
            mul_binary_tile(GATE_DST, UP_DST, GATE_DST);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(GATE_DST, out_cb);
            tile_regs_release();
        }
        cb_push_back(out_cb, out_N_block_tiles);
    }
}
#endif  // FUSE_SWIGLU

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

// Slightly modified from compute_common.hpp
void matmul_blocks(
    CircularBuffer& in0_cb,
    CircularBuffer& in1_cb,
    CircularBuffer& out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb.get_cb_id(),
                    in1_cb.get_cb_id(),
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
                    pack_tile<true>(write_dst_index, out_cb.get_cb_id(), out_tile_id);
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

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, intermediate_cb_id);
    matmul_init(in0_cb_id, in1_cb_id);

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
            // Accumulation buffer
            intermediate_cb.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                in0_cb.wait_front(in0_block_num_tiles);
                in1_cb.wait_front(in1_block_num_tiles);

                matmul_blocks(
                    in0_cb,
                    in1_cb,
                    intermediate_cb,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w);

                if (k_block == K_num_blocks - 1) {
                    /**
                     * On next iteration we might get reuse on in0.
                     * Only valid for Ring: k_forward toggles each n_block_iter so the last
                     * actual_k_block of n=X equals the first of n=X+1. Linear keeps k_forward
                     * fixed, so reusing would feed the previous iter's last K-block (=
                     * K_num_blocks-1) when the new iter wants K-block 0. Force fresh read.
                     */
#ifndef IS_LINEAR
                    if (n_block_iter < N_blocks_per_core - 1) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
                    }
#endif
                }
                if (!reuse_in0_block) {
                    in0_cb.pop_front(in0_block_num_tiles);
                }
                in1_cb.pop_front(in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            intermediate_cb.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

#ifdef FUSE_SWIGLU
            // SwiGLU collapses the interleaved gate/up block to half its N width.
            out_cb.reserve_back(out_block_num_tiles >> 1);
            intermediate_cb.wait_front(out_block_num_tiles);
#ifdef FUSE_BIAS
            in2_cb.wait_front(N_block_tiles);
#endif
            swiglu_block(
                intermediate_cb.get_cb_id(), in2_cb.get_cb_id(), out_cb.get_cb_id(), M_block_tiles, N_block_tiles);
#ifdef FUSE_BIAS
            in2_cb.pop_front(N_block_tiles);
#endif
            intermediate_cb.pop_front(out_block_num_tiles);

#elif !defined(FUSE_TERNARY)
            out_cb.reserve_back(out_block_num_tiles);
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
            out_cb.reserve_back(out_block_num_tiles);
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
