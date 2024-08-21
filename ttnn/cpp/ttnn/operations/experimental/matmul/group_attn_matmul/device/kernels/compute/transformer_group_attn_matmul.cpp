// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {
    uint32_t i = 0;

    uint32_t has_work_for_q_heads = get_arg_val<uint32_t>(i++);
    if (has_work_for_q_heads == 0) return;

    uint32_t batch                  = get_arg_val<uint32_t>(i++);
    uint32_t Mt                     = get_arg_val<uint32_t>(i++);
    uint32_t num_kv_heads_skip      = get_arg_val<uint32_t>(i++);
    uint32_t num_kv_heads_remaining = get_arg_val<uint32_t>(i++);

    // matmul params
    uint32_t in0_block_w            = get_arg_val<uint32_t>(i++);
    uint32_t out_subblock_h         = get_arg_val<uint32_t>(i++);
    uint32_t in1_num_subblocks      = get_arg_val<uint32_t>(i++);
    uint32_t in1_num_blocks         = get_arg_val<uint32_t>(i++);
    uint32_t in0_block_num_tiles    = get_arg_val<uint32_t>(i++);
    uint32_t in1_block_num_tiles    = get_arg_val<uint32_t>(i++);
    uint32_t out_num_tiles          = get_arg_val<uint32_t>(i++);

    // matmul inner loop tracking
    uint32_t in0_subblock_num_tiles = get_arg_val<uint32_t>(i++);
    uint32_t in1_per_core_w         = get_arg_val<uint32_t>(i++);


    constexpr uint32_t transpose_hw = get_compile_time_arg_val(0);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(1);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t intermediate_num_tiles = get_compile_time_arg_val(3);


    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_intermed0 = 24;
    constexpr uint32_t cb_intermed1 = 25;
    constexpr uint32_t out_cb_id = 16;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;
    constexpr uint32_t in0_num_blocks_w = 1; // TODO: Generalize

    // need switching between ColMajor and RowMajor for at least 32 times, inefficient
    #ifdef ARCH_GRAYSKULL
    mm_init(cb_in0, cb_in1, cb_intermed0, transpose_hw);
    #else
    // TODO: switch back to matmul block after didt solved
    mm_init(cb_in0, cb_in1, cb_intermed0, transpose_hw);
    // mm_block_init(cb_in0, cb_in1, cb_intermed0, transpose_hw, out_subblock_w, out_subblock_h, in0_block_w);
    #endif

    for (uint32_t b = 0; b < batch; b++) {

        for (uint32_t m = 0; m < Mt; m++) {  // TODO: Must be 1; generalize to support batch > 32 (ie. Mt > 1)
            for (uint32_t in0_block = 0; in0_block < in0_num_blocks_w; in0_block++) { // TODO: Must be 1; generalize to support inner dim blocking
                cb_wait_front(cb_in0, in0_block_num_tiles);

                for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
                    uint32_t in0_index_subblock_offset = 0;
                    for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                        cb_wait_front(cb_in1, in1_block_num_tiles);
                        cb_pop_front(cb_in1, num_kv_heads_skip);

                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) { // TODO: Must be 1; need to review inner dim blocking and the untilizing
                            uint32_t in1_index_subblock_offset = 0;

                            tile_regs_acquire();

                            #ifdef ARCH_GRAYSKULL
                            uint32_t dst_index = 0;
                            uint32_t in0_index_h_offset = 0;
                            for (uint32_t h = 0; h < out_subblock_h; h++) {
                                for (uint32_t w = 0; w < out_subblock_w; w++) {
                                    uint32_t in1_index_inner_dim_offset = 0;
                                    for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                        uint32_t in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                        uint32_t in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                        matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, dst_index, transpose_hw);
                                        in1_index_inner_dim_offset += in1_per_core_w;
                                    }
                                    dst_index++;
                                }
                                in0_index_h_offset += in0_block_w;
                            }
                            #else
                            // TODO: switch back to matmul block after didt solved
                            uint32_t dst_index = 0;
                            uint32_t in0_index_h_offset = 0;
                            for (uint32_t h = 0; h < out_subblock_h; h++) {
                                for (uint32_t w = 0; w < out_subblock_w; w++) {
                                    uint32_t in1_index_inner_dim_offset = 0;
                                    for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                        uint32_t in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                        uint32_t in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                        matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, dst_index, transpose_hw);
                                        in1_index_inner_dim_offset += in1_per_core_w;
                                    }
                                    dst_index++;
                                }
                                in0_index_h_offset += in0_block_w;
                            }
                            // // Compute output sub-block
                            // uint32_t dst_index = 0; // start at 0, each call to matmul_block internally increments dst_index
                            // uint32_t in0_index = in0_index_subblock_offset; // offset into in0 block
                            // uint32_t in1_index = in1_index_subblock_offset; // offset into in1 block
                            // // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w
                            // for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                            //     // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                            //     // accumulation is done by iterating matmul_block across inner dim
                            //     // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                            //     matmul_block(cb_in0, cb_in1, in0_index, in1_index, dst_index, transpose_hw, out_subblock_w, out_subblock_h, in0_block_w);
                            //     in0_index ++;  // stride right by 1
                            //     in1_index += in1_per_core_w; // to stride down by 1 need to stride by in_per_core_w (should be called in1_block_w)
                            // }
                            #endif

                            tile_regs_commit();

                            in1_index_subblock_offset += out_subblock_w;
                        } // in1_num_subblocks loop
                        cb_pop_front(cb_in1, num_kv_heads_remaining);

                        // TODO: Review inner dim blocking, untilizing, and in1_num_subblocks > 1 (with pack_untilize, can only untilize up to dst num tiles)
                        // This should normally be inside subblock loop and we pack out out_subblock_num_tiles
                        pack_untilize_dst_init_short<intermediate_num_tiles>(cb_intermed0);
                        cb_reserve_back(cb_intermed0, intermediate_num_tiles);
                        tile_regs_wait();
                        pack_untilize_dst<intermediate_num_tiles>(cb_intermed0);
                        pack_untilize_uninit();

                        tile_regs_release();
                        cb_push_back(cb_intermed0, intermediate_num_tiles);

                    }  // 32 tiles loop

                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }  // in1_num_blocks loop
            } // in0_num_blocks_w

            // cb_intermed1 comes from reader; untilized row-major tile
            unpack_reconfig_data_format_srca(cb_in1, cb_intermed1);
            pack_reconfig_data_format(cb_intermed0, out_cb_id);
            cb_wait_front(cb_intermed1, out_num_tiles);

            cb_reserve_back(out_cb_id, out_num_tiles);

            // tilize CB::intermed1 and write to CB::c_out0
            tilize_init_short_with_dt(cb_in1, cb_intermed1, out_num_tiles);
            tilize_block(cb_intermed1, out_num_tiles, out_cb_id);
            cb_push_back(out_cb_id, out_num_tiles);

            cb_pop_front(cb_intermed1, out_num_tiles);
            tilize_uninit(cb_intermed1);

            cb_pop_front(cb_in0, in0_block_num_tiles);
        } // Mt loop
    }  // batch

}
} // NAMESPACE
