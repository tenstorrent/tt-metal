// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmul/bias compute primitives for the conv3d compute kernel (conv3d_compute.cpp).
//
// These are the fork's matmul_blocks / add_bias_inplace / add_block_inplace.  Upstream conv3d inlines
// equivalents in its compute.cpp and has since added DEST-batched variants; this fork keeps the simple
// per-tile forms because they pair with the kernel's manual fp32 cross-core reduce (see the compute
// kernel header for why the fork is not re-based).  Keep the matmul subblock math aligned with upstream
// when it changes; the bias/reduce forms are intentionally fork-local.

#pragma once

#include "api/compute/compute_kernel_api.h"
#include <tt-metalium/constants.hpp>

#include "api/compute/untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Slightly modified from compute_common.hpp
inline void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t in0_num_subblocks,
    const uint32_t in1_num_subblocks,
    const uint32_t in0_block_w,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const bool transpose,
    const uint32_t in1_base_tile = 0) {
    // precondition: in0_cb has M*K produced
    // precondition: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;

    reconfig_data_format(in1_cb, in0_cb);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = in1_base_tile;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            tile_regs_commit();

            cb_reserve_back(out_cb, out_subblock_num_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
            cb_push_back(out_cb, out_subblock_num_tiles);
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
}

template <uint32_t rows, uint32_t cols>
void add_bias_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t dst_tiles = 1;

    add_bcast_rows_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, cols);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            tile_regs_acquire();
            // Add jth tile of bias to each column j of in0_cb
            add_tiles_bcast_rows(in0_cb, in1_cb, 0, j, 0);
            tile_regs_commit();
            cb_pop_front(in0_cb, dst_tiles);
            cb_reserve_back(in0_cb, dst_tiles);
            tile_regs_wait();
            pack_tile(0, in0_cb);
            cb_push_back(in0_cb, dst_tiles);
            tile_regs_release();
        }
    }
}

template <uint32_t num_tiles>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_cb has num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    constexpr uint32_t dst_tiles = 1;

    add_tiles_init(in0_cb, in1_cb);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        add_tiles(in0_cb, in1_cb, 0, 0, 0);
        tile_regs_commit();
        cb_pop_front(in0_cb, dst_tiles);
        cb_pop_front(in1_cb, dst_tiles);
        cb_reserve_back(in0_cb, dst_tiles);
        tile_regs_wait();
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, dst_tiles);
        tile_regs_release();
    }
}
