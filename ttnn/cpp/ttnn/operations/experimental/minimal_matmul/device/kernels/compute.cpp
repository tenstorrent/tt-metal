// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include <tt-metalium/constants.hpp>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"

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

// Slightly modified from compute_common.hpp
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t M_num_subblocks,
    const uint32_t N_num_subblocks,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_subblock = 0; M_subblock < M_num_subblocks; ++M_subblock) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_subblock = 0; N_subblock < N_num_subblocks; ++N_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, false, subblock_w, subblock_h, K_block_tiles);
                in0_index++;
                in1_index += N_block_tiles;
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_subblock * subblock_h + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_subblock * subblock_w + w;
                    uint32_t out_tile_id = h_tile_id * N_block_tiles + w_tile_id;
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

void safe_print_full_tile(uint32_t cb_id) {
#if defined(DEBUG_PRINT_ENABLED)
    UNPACK(tt::compute::common::print_full_tile(cb_id));
#endif
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);

    uint32_t argidx = 0;
    const uint32_t M_start_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_block = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;
#ifdef FUSE_BIAS
    constexpr uint32_t in2_cb = tt::CBIndex::c_4;
#endif

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
    bool reuse_in1_block = false;
    for (uint32_t m_block = M_start_block; m_block <= M_end_block; m_block++) {
        for (uint32_t n_block = N_start_block; n_block <= N_end_block; n_block++) {
            mm_block_init_short(
                in0_cb,
                in1_cb,
                false /*transpose*/,
                subblock_w /*ct_dim*/,
                subblock_h /*rt_dim*/,
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
                    M_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    M_num_subblocks,
                    N_num_subblocks,
                    subblock_h,
                    subblock_w);

                if (k_block == K_num_blocks - 1) {
                    /**
                     * On next iteration we're going to get some reuse on in0 or in1
                     * Therefore you should not pop one of them
                     *
                     */
                    if (n_block < N_end_block) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
                    } else {
                        // going to stride on M, so reuse in1
                        reuse_in1_block = true;
                    }
                }
                if (!reuse_in0_block) {
                    cb_pop_front(in0_cb, in0_block_num_tiles);
                }
                if (!reuse_in1_block) {
                    cb_pop_front(in1_cb, in1_block_num_tiles);
                }
                reuse_in0_block = false;
                reuse_in1_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            /**
             * Depending on the direction we're striding, either in0 or in1 DM will write the output.
             * The CB pointers must match each other.
             * Push both of them.
             */

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));
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
        }
    }
}
}  // namespace NAMESPACE
