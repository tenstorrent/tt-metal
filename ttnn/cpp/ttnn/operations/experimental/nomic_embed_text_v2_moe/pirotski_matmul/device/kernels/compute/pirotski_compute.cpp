// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// minimal_matmul's legacy compute kernel as pirotski_matmul was validated against it (the tree of #60402),
// kept here because the op passes its own M range per core: minimal_matmul's copy now derives the range
// from a core index and a common M_tiles argument (#62181). Only the FUSE_BIAS variant is kept.

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

// Renamed from copy_block to avoid an ambiguous overload with ckernel::copy_block (added to
// api/compute/tile_move_copy.h in #49070), which has the identical (uint32_t, uint32_t, uint32_t,
// uint32_t) signature and is in scope here via `using namespace ckernel`. See tt-metal#50386.
void copy_and_pack_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    // Data formats must be reconfigured before the op init: llk_unpack_A_init asserts that the
    // unpacker is already configured for in_cb (see #55052), and the preceding matmul left it
    // configured for its own operands.
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(in_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
            pack_tile(fused_act_dst_id, out_cb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        cb_out.push_back(N_block_tiles);
    }
}

/**
 * Add bias to input block
 * Performs: output = input + bias (row broadcast)
 *
 * stream_output:
 *   - true: Pushes tiles one row at a time (for intermediate output to next stage)
 *   - false: Pushes all tiles at end (for final output)
 */
void add_bias_block(uint32_t in_cb, uint32_t bias_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    reconfig_data_format(in_cb, bias_cb);
    pack_reconfig_data_format(out_cb);
    add_bcast_rows_init(in_cb, bias_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            add_tiles_bcast<BroadcastType::ROW>(in_cb, bias_cb, tile_id, n, fused_act_dst_id /*dst*/);
            pack_tile(fused_act_dst_id, out_cb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        cb_out.push_back(N_block_tiles);
    }
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

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    constexpr uint32_t in2_cb = tt::CBIndex::c_4;

    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_intermediate(intermediate_cb);
    CircularBuffer cb_in2(in2_cb);

    // compute_kernel_hw_startup must be the first compute API call (before SFPU/op inits).
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);

    matmul_init(in0_cb, in1_cb);

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

            // Reconfig before init: on all but the first block the unpackers are still
            // configured for the previous output stage's operands (see #55052).
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermediate_cb);
            matmul_block_init(
                in0_cb,
                in1_cb,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            // Accumulation buffer
            cb_intermediate.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                cb_in0.wait_front(in0_block_num_tiles);
                cb_in1.wait_front(in1_block_num_tiles);

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
                    cb_in0.pop_front(in0_block_num_tiles);
                }
                cb_in1.pop_front(in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_intermediate.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

            cb_out.reserve_back(out_block_num_tiles);
            cb_intermediate.wait_front(out_block_num_tiles);
#ifndef FUSE_BIAS
            copy_and_pack_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#else
            cb_in2.wait_front(N_block_tiles);
            add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_in2.pop_front(N_block_tiles);
#endif  // FUSE_BIAS
            cb_intermediate.pop_front(out_block_num_tiles);
        }
    }
}
