// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "mod_div_lib.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/transpose_wh.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"

#define DEBUG_PRINT 0

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

inline void tilize_in(
    uint32_t in_cb_id,
    uint32_t in_subblock_h,
    uint32_t in_block_w,
    uint32_t in_num_subblocks,
    uint32_t out_cb_id
) {
    tilize_init_short(in_cb_id, in_block_w);
    for (uint32_t in_subblock = 0; in_subblock < in_num_subblocks; ++in_subblock) {
        for (uint32_t h = 0; h < in_subblock_h; ++h) {
            cb_wait_front(in_cb_id, in_block_w);
            cb_reserve_back(out_cb_id, in_block_w);
            tilize_block(in_cb_id, in_block_w, out_cb_id);
            cb_push_back(out_cb_id, in_block_w);
            cb_pop_front(in_cb_id, in_block_w);
        }
    }
    tilize_uninit(in_cb_id);
}

inline void eltwise_mul_and_add_block_v2(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t eltwise_mul_partials_cb_cb_id, uint32_t temp_sum_cb, uint32_t out_cb_id, uint32_t block_num_tiles, uint32_t idx, uint32_t total_blocks) {
    uint32_t last_block_idx = total_blocks - 1;
    for(uint32_t i=0; i<block_num_tiles; i++) {
        cb_wait_front(in1_cb_id, 1);
        cb_wait_front(in0_cb_id, 1);
        cb_reserve_back(eltwise_mul_partials_cb_cb_id, 1);
        mul_tiles_init(in0_cb_id, in1_cb_id);
        ACQ();
        mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
        pack_tile(0, eltwise_mul_partials_cb_cb_id);
        REL();
        cb_push_back(eltwise_mul_partials_cb_cb_id, 1);
        cb_pop_front(in0_cb_id, 1);
        cb_pop_front(in1_cb_id, 1);
        if(idx==0){
            copy_tile_to_dst_init_short();
            ACQ();
            cb_wait_front(eltwise_mul_partials_cb_cb_id, 1);
            cb_reserve_back(out_cb_id, 1);
            copy_tile(eltwise_mul_partials_cb_cb_id, 0, 0);
            pack_tile(0, out_cb_id);
            REL();
            cb_push_back(out_cb_id, 1);
            cb_pop_front(eltwise_mul_partials_cb_cb_id, 1);
        }
        else{

            add_tiles_init();
            cb_wait_front(eltwise_mul_partials_cb_cb_id, 1);
            cb_wait_front(out_cb_id, 1);
            ACQ();
            add_tiles(eltwise_mul_partials_cb_cb_id, out_cb_id, 0, 0, 0);
            pack_tile(0, temp_sum_cb);
            REL();
            cb_push_back(temp_sum_cb, 1);
            cb_pop_front(eltwise_mul_partials_cb_cb_id, 1);
            cb_pop_front(out_cb_id, 1);

            copy_tile_to_dst_init_short();
            ACQ();
            cb_wait_front(temp_sum_cb, 1);
            cb_reserve_back(out_cb_id, 1);
            copy_tile(temp_sum_cb, 0, 0);
            pack_tile(0, out_cb_id);
            REL();
            cb_push_back(out_cb_id, 1);
            cb_pop_front(temp_sum_cb, 1);
        }
    }
}

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t in0_block_w            = get_compile_time_arg_val(0); // inner block size in tiles
    constexpr uint32_t in0_num_subblocks      = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles    = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in0_subblock_h         = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks      = get_compile_time_arg_val(5); // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles    = get_compile_time_arg_val(6); //out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w            = get_compile_time_arg_val(7); // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    constexpr uint32_t in0_num_blocks_h       = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_blocks_w       = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_blocks_w       = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h         = get_compile_time_arg_val(11); // inner row block size in tiles
    constexpr uint32_t out_subblock_w         = get_compile_time_arg_val(12); // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13); // out_subblock_h * out_subblock_w;
    constexpr bool tilize_in0                 = get_compile_time_arg_val(14);
    constexpr bool untilize_out               = get_compile_time_arg_val(15);

    constexpr uint32_t out_block_num_tiles    = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    constexpr uint32_t out_block_w = in1_block_w;

    // CB indices
    constexpr uint32_t in0_cb_id                                = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id                                = tt::CB::c_in1;
    constexpr uint32_t in0_pretilize_cb_id                      = tt::CB::c_in6;
    constexpr uint32_t in0_cb_second_reader_id                  = tt::CB::c_in7;
    constexpr uint32_t eltwise_mul_partials_cb                  = tt::CB::c_intermed0;
    constexpr uint32_t tilized_in0_cb_id                        = tt::CB::c_intermed1;
    constexpr uint32_t temp_sum_cb                              = tt::CB::c_intermed3;
    constexpr uint32_t prev_eltwise_cb                          = tt::CB::c_intermed5;
    constexpr uint32_t out_cb_id                                = tt::CB::c_out0;

    constexpr uint32_t in0_num_subblocks_read = in0_num_subblocks;

    constexpr uint32_t num_blocks = in0_num_blocks_h * in0_num_blocks_w; // num_tokens window

    binary_op_init_common(in0_cb_id, in1_cb_id, out_cb_id);


    for(uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {

        for(uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {

            uint32_t i = in0_block_h_i * in0_num_blocks_w + in0_block_w_i;
            if constexpr (tilize_in0) {
                reconfig_data_format_srca(in0_cb_id);
                pack_reconfig_data_format(tilized_in0_cb_id);
                tilize_in(in0_cb_id, in0_subblock_h, in0_block_w, in0_num_subblocks_read, tilized_in0_cb_id);
            }
            reconfig_data_format_srca(tilized_in0_cb_id);
            pack_reconfig_data_format(eltwise_mul_partials_cb);
            eltwise_mul_and_add_block_v2(tilized_in0_cb_id, in1_cb_id, eltwise_mul_partials_cb, temp_sum_cb, out_cb_id, in0_block_num_tiles, i, num_blocks);


        } // for in0_num_blocks_h
    } // for in0_num_blocks_w
} // MAIN
} // NAMESPACE
