// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

template <BroadcastType bcast_type>
ALWI void unary_bcast_init_short(uint32_t icb, uint32_t call_line = __builtin_LINE()) {
    state_configure<Operand::SRCA>(icb, call_line);

#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A_init<bcast_type, false, EltwiseBinaryReuseDestType::NONE, true>(false, false, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, bcast_type>(icb)));
    } else {
        UNPACK((llk_unpack_A_init<bcast_type, false, EltwiseBinaryReuseDestType::NONE, false>(false, false, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<B2D, DST_ACCUM_MODE, bcast_type>(icb)));
    }
#endif
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

/**
 * Multiply input block by ternary_b and scalar
 * Performs: output = scalar * input * ternary_b
 *
 * Uses mul_tiles for input * ternary_b, then mul_unary_tile for result * scalar
 *
 * ternary_b: full block (M_block_tiles * N_block_tiles in CB), consumed one row at a time
 * input: full block, consumed one row at a time from in_cb
 * scalar_value: scalar multiplier (as uint32_t encoding)
 */
void mul_block(
    uint32_t in_cb,
    uint32_t ternary_b_cb,
    uint32_t scalar_value,
    uint32_t out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles) {
    DPRINT << "COMPUTE: mul_block START - M=" << M_block_tiles << " N=" << N_block_tiles << ENDL();

    // Initialize multiplication operations
    mul_tiles_init(in_cb, ternary_b_cb);
    binop_with_scalar_tile_init();
    pack_reconfig_data_format(out_cb);

    cb_reserve_back(out_cb, N_block_tiles);

    constexpr uint32_t dst_id = 0;

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of input tiles
        DPRINT << "COMPUTE: mul_block waiting for in_cb row " << m << ENDL();
        cb_wait_front(in_cb, N_block_tiles);

        // Wait for one row of ternary_b tiles
        DPRINT << "COMPUTE: mul_block waiting for ternary_b_cb row " << m << ENDL();
        cb_wait_front(ternary_b_cb, N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();

            // Multiply: dst = input[n] * ternary_b[n]
            mul_tiles(in_cb, ternary_b_cb, n, n, dst_id);

            // Multiply by scalar: dst = dst * scalar
            mul_unary_tile(dst_id, scalar_value);

            tile_regs_commit();
            tile_regs_wait();

            DPRINT << "COMPUTE: mul_block[m=" << m << ",n=" << n << "] packing result" << ENDL();
            pack_tile(dst_id, out_cb);

            tile_regs_release();
        }

        // Pop consumed tiles
        cb_pop_front(in_cb, N_block_tiles);
        cb_pop_front(ternary_b_cb, N_block_tiles);

        // Push output row
        DPRINT << "COMPUTE: mul_block pushing row " << m << " to out_cb" << ENDL();
        cb_push_back(out_cb, N_block_tiles);
    }

    DPRINT << "COMPUTE: mul_block complete" << ENDL();
}

void add_bias_and_addcmul_block(
    uint32_t intermediate_cb,
    uint32_t bias_cb,
    uint32_t ternary_a_cb,
    uint32_t ternary_b_cb,
    uint32_t scalar_value,
    uint32_t out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles) {
    // Note: we could also defined addcmul as unary_bcast_tile + addcmul_tile
    // However, unary_cast_tile updates both Unpacker/Math and *Pack*, which makes its setup
    // more difficult alongside copy_tile() and add_bcast_tiles

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
            pack_tile(DST_ID, intermediate_cb, tile_id);
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
    // ============================================

    mul_tiles_init(intermediate_cb, ternary_b_cb);
    binop_with_scalar_tile_init();
    reconfig_data_format(intermediate_cb, ternary_b_cb);
    pack_reconfig_data_format(intermediate_cb);

    // Wait for ALL input data ONCE
    cb_wait_front(intermediate_cb, out_block_num_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        // Wait for one row of ternary_b tiles
        cb_wait_front(ternary_b_cb, N_block_tiles);

        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();

            // Multiply: intermediate[tile_id] * ternary_b[n]
            mul_tiles(intermediate_cb, ternary_b_cb, tile_id, n, DST_ID);

            // Multiply by scalar
            mul_unary_tile(DST_ID, scalar_value);

            // fill_tile_init();
            // fill_tile(DST_ID, 3.0f);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(DST_ID, intermediate_cb, tile_id);

            tile_regs_release();
        }

        cb_pop_front(ternary_b_cb, N_block_tiles);
    }

    // Pop input and push output ONCE at the end
    cb_pop_front(intermediate_cb, out_block_num_tiles);
    // Reserve output buffer ONCE
    cb_reserve_back(intermediate_cb, out_block_num_tiles);
    cb_push_back(intermediate_cb, out_block_num_tiles);

    // ============================================
    // STEP 3: Add ternary_a block
    // Read from intermediate_cb and write to out_cb
    // ============================================

    add_bcast_rows_init_short(intermediate_cb, ternary_a_cb);
    reconfig_data_format(intermediate_cb, ternary_a_cb);
    pack_reconfig_data_format(out_cb);
    unary_op_init_common(intermediate_cb, out_cb);

    // Wait for ALL input data ONCE
    cb_wait_front(intermediate_cb, out_block_num_tiles);
    cb_wait_front(ternary_a_cb, N_block_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            uint32_t tile_id = m * N_block_tiles + n;

            tile_regs_acquire();

            add_tiles_bcast<BroadcastType::ROW>(intermediate_cb, ternary_a_cb, tile_id, n, DST_ID);
            copy_tile_to_dst_init_short(intermediate_cb);
            copy_tile(intermediate_cb, tile_id, DST_ID);

            // DPRINT << "COMPUTE: add_bias_and_addcmul_block[m=" << m << ",n=" << n << ", tile_id = " << tile_id <<"]
            // filling tile with 4.0f" << ENDL();

            // fill_tile_init();
            // fill_tile(DST_ID, 4.0f);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(DST_ID, out_cb, tile_id);

            tile_regs_release();
        }
        cb_push_back(out_cb, N_block_tiles);
    }

    // Pop and push ONCE at the end
    cb_pop_front(intermediate_cb, out_block_num_tiles);

    cb_pop_front(ternary_a_cb, N_block_tiles);
}

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

#ifdef FUSE_TERNARY
    const uint32_t fused_ternary_scalar_uint = get_arg_val<uint32_t>(argidx++);
#else
    // Default value when ternary is not fused (not used, helps compiler optimize)
    constexpr uint32_t fused_ternary_scalar_uint = 0;
#endif

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

#if defined(FUSE_BIAS) || defined(FUSE_TERNARY)

    // Circular buffers for fused operations
    constexpr uint32_t in2_cb = tt::CBIndex::c_4;  // bias input
    constexpr uint32_t ternary_a_cb = tt::CBIndex::c_5;
    constexpr uint32_t ternary_b_cb = tt::CBIndex::c_6;

#ifdef FUSE_BIAS
#ifdef FUSE_TERNARY
    // Need two intermediate buffers: one after bias, one after mul_block
    constexpr uint32_t bias_output_cb = tt::CBIndex::c_7;
    constexpr uint32_t mul_output_cb = tt::CBIndex::c_8;
#endif
#endif

#if !defined(FUSE_BIAS) && defined(FUSE_TERNARY)
    // Need one intermediate buffer after mul_block
    constexpr uint32_t mul_output_cb = tt::CBIndex::c_7;
#endif

#endif  // defined(FUSE_BIAS) || defined(FUSE_TERNARY)

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
                cb_wait_front(in0_cb, in0_block_num_tiles);
                cb_wait_front(in1_cb, in1_block_num_tiles);

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
                     * On next iteration we might get reuse on in0
                     *
                     */
                    if (n_block_iter < N_blocks_per_core - 1) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
                    }
                }
                if (!reuse_in0_block) {
                    cb_pop_front(in0_cb, in0_block_num_tiles);
                }
                cb_pop_front(in1_cb, in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

#ifndef FUSE_TERNARY
            cb_wait_front(intermediate_cb, out_block_num_tiles);
            cb_reserve_back(out_cb, out_block_num_tiles);
#ifndef FUSE_BIAS
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#else
            cb_wait_front(in2_cb, N_block_tiles);
            add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_pop_front(in2_cb, N_block_tiles);
#endif
            cb_pop_front(intermediate_cb, out_block_num_tiles);

#else   // FUSE_TERNARY is set
            cb_reserve_back(out_cb, out_block_num_tiles);
            add_bias_and_addcmul_block(
                intermediate_cb,
                in2_cb,
                ternary_a_cb,
                ternary_b_cb,
                fused_ternary_scalar_uint,
                out_cb,
                M_block_tiles,
                N_block_tiles);
#endif  // FUSE_TERNARY
        }
    }
}
