// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    uint32_t has_work_for_q_heads = get_arg(args::has_work_for_q_heads);
    if (has_work_for_q_heads == 0) return;

    uint32_t batch                  = get_arg(args::batch);
    uint32_t Mt                     = get_arg(args::Mt);
    uint32_t num_kv_heads_skip      = get_arg(args::num_kv_heads_skip);
    uint32_t num_kv_heads_remaining = get_arg(args::num_kv_heads_remaining);

    // matmul params
    uint32_t in0_block_w            = get_arg(args::in0_block_w);
    uint32_t out_subblock_h         = get_arg(args::out_subblock_h);
    uint32_t in1_num_subblocks      = get_arg(args::in1_num_subblocks);
    uint32_t in1_num_blocks         = get_arg(args::in1_num_blocks);
    uint32_t in0_block_num_tiles    = get_arg(args::in0_block_num_tiles);
    uint32_t in1_block_num_tiles    = get_arg(args::in1_block_num_tiles);
    uint32_t out_num_tiles          = get_arg(args::out_num_tiles);

    // matmul inner loop tracking
    uint32_t in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);
    uint32_t in1_per_core_w         = get_arg(args::in1_per_core_w);


    constexpr uint32_t transpose_hw = get_arg(args::transpose_hw);
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);
    constexpr uint32_t intermediate_num_tiles = get_arg(args::intermediate_num_tiles);


    DataflowBuffer cb_in0_obj(dfb::in0);
    DataflowBuffer cb_in1_obj(dfb::in1);
    DataflowBuffer cb_intermed0_obj(dfb::intermed0);
    DataflowBuffer cb_intermed1_obj(dfb::intermed1);
    DataflowBuffer cb_out_obj(dfb::out);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;
    constexpr uint32_t in0_num_blocks_w = 1; // TODO: Generalize

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::intermed0);
    // need switching between ColMajor and RowMajor for at least 32 times, inefficient
    // TODO: switch back to matmul block after didt solved
    matmul_init(dfb::in0, dfb::in1, transpose_hw);
    // matmul_block_init(dfb::in0, dfb::in1, transpose_hw, out_subblock_w, out_subblock_h, in0_block_w);

    for (uint32_t b = 0; b < batch; b++) {

        for (uint32_t m = 0; m < Mt; m++) {  // TODO: Must be 1; generalize to support batch > 32 (ie. Mt > 1)
            for (uint32_t in0_block = 0; in0_block < in0_num_blocks_w; in0_block++) { // TODO: Must be 1; generalize to support inner dim blocking
                cb_in0_obj.wait_front(in0_block_num_tiles);

                for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
                    uint32_t in0_index_subblock_offset = 0;
                    for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                        cb_in1_obj.wait_front(in1_block_num_tiles);
                        cb_in1_obj.pop_front(num_kv_heads_skip);

			// This init changes DEST mapping, hence needs to be called before MATH does any processing, so that it has correct DEST mapping.
                        pack_untilize_dest_init<intermediate_num_tiles>(dfb::intermed0);

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
                                        matmul_tiles(dfb::in0, dfb::in1, in0_index, in1_index, dst_index);
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
                                        matmul_tiles(dfb::in0, dfb::in1, in0_index, in1_index, dst_index);
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
                            // // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                            // for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                            //     // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                            //     // accumulation is done by iterating matmul_block across inner dim
                            //     // in0_block_w is passed as innder dim (kt) to matmul_block, internally used to stride in0
                            //     matmul_block(dfb::in0, dfb::in1, in0_index, in1_index, dst_index, transpose_hw, out_subblock_w, out_subblock_h, in0_block_w);
                            //     in0_index ++;  // stride right by 1
                            //     in1_index += in1_per_core_w; // to stride down by 1 need to stride by in_per_core_w (should be called in1_block_w)
                            // }
                            #endif

                            tile_regs_commit();

                            in1_index_subblock_offset += out_subblock_w;
                        } // in1_num_subblocks loop
                        cb_in1_obj.pop_front(num_kv_heads_remaining);
                        // TODO: Review inner dim blocking, untilizing, and in1_num_subblocks > 1 (with pack_untilize, can only untilize up to dst num tiles)
                        // This should normally be inside subblock loop and we pack out out_subblock_num_tiles
                        cb_intermed0_obj.reserve_back(intermediate_num_tiles);
                        tile_regs_wait();
                        pack_untilize_dest<intermediate_num_tiles>(dfb::intermed0);
                        pack_untilize_uninit(dfb::intermed0);

                        tile_regs_release();
                        cb_intermed0_obj.push_back(intermediate_num_tiles);

                    }  // 32 tiles loop

                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }  // in1_num_blocks loop
            } // in0_num_blocks_w

            // cb_intermed1 comes from reader; untilized row-major tile
            reconfig_data_format_srca(dfb::in1, dfb::intermed1);
            pack_reconfig_data_format(dfb::intermed0, dfb::out);
            cb_intermed1_obj.wait_front(out_num_tiles);

            cb_out_obj.reserve_back(out_num_tiles);

            // tilize intermed1 and write to out
            tilize_init_short_with_dt(dfb::in1, dfb::intermed1, out_num_tiles, dfb::out);
            tilize_block(dfb::intermed1, out_num_tiles, dfb::out);
            cb_out_obj.push_back(out_num_tiles);

            cb_intermed1_obj.pop_front(out_num_tiles);
            tilize_uninit(dfb::intermed1, dfb::out);

            cb_in0_obj.pop_front(in0_block_num_tiles);
        } // Mt loop
    }  // batch

}
