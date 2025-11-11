// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <tools/profiler/kernel_profiler.hpp>

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(1);      // out_subblock_h*in0_block_w
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_w*in0_block_w
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(3);              // out_subblock_w
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(4);           // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(5);           // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(6);   // out_subblock_h * out_subblock_w
    constexpr bool untilize_out = get_compile_time_arg_val(7);                 // untilize output

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;

    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;

    constexpr uint32_t in1_transpose_tile = false;

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

    cb_wait_front(in0_cb_id, in0_block_num_tiles);
    cb_wait_front(in1_cb_id, in1_block_num_tiles);

    tile_regs_acquire();

    // Compute output sub-block
    uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w

    {
        DeviceZoneScopedN("matmul_block");
        for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
            // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
            // accumulation is done by iterating matmul_block across inner dim
            // in0_block_w is passed as inner dim (kt) to matmul_block, internally used to stride in0
            matmul_block(
                in0_cb_id,
                in1_cb_id,
                in0_index,
                in1_index,
                dst_index,
                in1_transpose_tile,
                out_subblock_w,
                out_subblock_h,
                in0_block_w);
            in0_index++;               // stride right by 1
            in1_index += in1_block_w;  // stride down by 1
        }
    }

    tile_regs_commit();
    // Pack out to output buffer
    cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
    tile_regs_wait();

    uint32_t start_dst_index = 0;
    pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

    tile_regs_release();
    cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

    cb_pop_front(in0_cb_id, in0_block_num_tiles);
    cb_pop_front(in1_cb_id, in1_block_num_tiles);
}
}  // namespace NAMESPACE
