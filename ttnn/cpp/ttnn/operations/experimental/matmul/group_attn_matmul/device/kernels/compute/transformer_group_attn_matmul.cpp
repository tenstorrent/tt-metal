// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul_op.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "experimental/circular_buffer.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
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


    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_intermed0 = tt::CBIndex::c_3;
    constexpr uint32_t cb_intermed1 = tt::CBIndex::c_4;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_5;

    experimental::CircularBuffer cb_in0_obj(cb_in0);
    experimental::CircularBuffer cb_in1_obj(cb_in1);
    experimental::CircularBuffer cb_intermed0_obj(cb_intermed0);
    experimental::CircularBuffer cb_intermed1_obj(cb_intermed1);
    experimental::CircularBuffer cb_out_obj(out_cb_id);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;
    constexpr uint32_t in0_num_blocks_w = 1; // TODO: Generalize

    ckernel::MatmulOpConfig mm_cfg{};
    mm_cfg.in0_cb_id = cb_in0;
    mm_cfg.in1_cb_id = cb_in1;
    mm_cfg.out_cb_id = cb_intermed0;
    mm_cfg.transpose = static_cast<bool>(transpose_hw);
    ckernel::TileMatmulOp mm(mm_cfg);
    mm.init();

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
                        pack_untilize_dest_init<intermediate_num_tiles>(cb_intermed0);

                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) { // TODO: Must be 1; need to review inner dim blocking and the untilizing
                            uint32_t in1_index_subblock_offset = 0;

                            tile_regs_acquire();

                            uint32_t dst_index = 0;
                            uint32_t in0_index_h_offset = 0;
                            for (uint32_t h = 0; h < out_subblock_h; h++) {
                                for (uint32_t w = 0; w < out_subblock_w; w++) {
                                    uint32_t in1_index_inner_dim_offset = 0;
                                    for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                        uint32_t in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                        uint32_t in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                        mm.matmul(in0_index, in1_index, dst_index);
                                        in1_index_inner_dim_offset += in1_per_core_w;
                                    }
                                    dst_index++;
                                }
                                in0_index_h_offset += in0_block_w;
                            }

                            tile_regs_commit();

                            in1_index_subblock_offset += out_subblock_w;
                        } // in1_num_subblocks loop
                        cb_in1_obj.pop_front(num_kv_heads_remaining);
                        // TODO: Review inner dim blocking, untilizing, and in1_num_subblocks > 1 (with pack_untilize, can only untilize up to dst num tiles)
                        // This should normally be inside subblock loop and we pack out out_subblock_num_tiles
                        cb_intermed0_obj.reserve_back(intermediate_num_tiles);
                        tile_regs_wait();
                        pack_untilize_dest<intermediate_num_tiles>(cb_intermed0);
                        pack_untilize_uninit(cb_intermed0);

                        tile_regs_release();
                        cb_intermed0_obj.push_back(intermediate_num_tiles);

                    }  // 32 tiles loop

                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }  // in1_num_blocks loop
            } // in0_num_blocks_w

            // cb_intermed1 comes from reader; untilized row-major tile
            reconfig_data_format_srca(cb_in1, cb_intermed1);
            pack_reconfig_data_format(cb_intermed0, out_cb_id);
            cb_intermed1_obj.wait_front(out_num_tiles);

            cb_out_obj.reserve_back(out_num_tiles);

            // tilize CB::intermed1 and write to CBIndex::c_16
            tilize_init_short_with_dt(cb_in1, cb_intermed1, out_num_tiles, out_cb_id);
            tilize_block(cb_intermed1, out_num_tiles, out_cb_id);
            cb_out_obj.push_back(out_num_tiles);

            cb_intermed1_obj.pop_front(out_num_tiles);
            tilize_uninit(cb_intermed1, out_cb_id);

            cb_in0_obj.pop_front(in0_block_num_tiles);
        } // Mt loop
    }  // batch

}
