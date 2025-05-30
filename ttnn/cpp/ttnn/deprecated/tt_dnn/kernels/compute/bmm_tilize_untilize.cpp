// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "mod_div_lib.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#define DEBUG_PRINT 0

// #include "debug_macros.h"

// SliceRange srt = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 8, .ws = 1};
// SliceRange srr = SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 4};
// SliceRange srr1 = SliceRange{.h0 = 1, .h1 = 2, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
// SliceRange src = SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

inline void kernel_sleep(uint32_t loop_count = 1000) { for (volatile uint32_t i = 0; i < loop_count; ++i); }

inline void tilize_in(
    uint32_t in_cb_id, uint32_t in_subblock_h, uint32_t in_block_w, uint32_t in_num_subblocks, uint32_t out_cb_id) {
    // UNPACK(( kernel_sleep(100) ));
    UNPACK((llk_unpack_reconfig_data_format(1, 0, 0, 0)));
    MATH((llk_math_reconfig_data_format(1, 0, 0, 0)));
    tilize_init(in_cb_id, in_block_w, out_cb_id);
    for (uint32_t in_subblock = 0; in_subblock < in_num_subblocks; ++in_subblock) {
        for (uint32_t h = 0; h < in_subblock_h; ++h) {
            cb_wait_front(in_cb_id, in_block_w);
            cb_reserve_back(out_cb_id, in_block_w);
            tilize_block(in_cb_id, in_block_w, out_cb_id);
            cb_push_back(out_cb_id, in_block_w);
            cb_pop_front(in_cb_id, in_block_w);
        }
    }
    tilize_uninit_with_dt(0, 1, out_cb_id);
}  // tilize_in()

// NOTE: Bias is not supported with the untilize option
#ifndef FUSE_BIAS

inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    uint32_t interm_cb_id,
    uint32_t reblock_cb_id,
    uint32_t out_cb_id) {
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    cb_wait_front(interm_cb_id, num_tiles_in_row_of_subblocks);

    int within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        int block_offset = 0;

        // Reblock
        copy_tile_to_dst_init_short(interm_cb_id);
        cb_reserve_back(reblock_cb_id, out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                tile_regs_acquire();
                copy_tile(interm_cb_id, tile_index, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, reblock_cb_id);
                tile_regs_release();
            }
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(reblock_cb_id, out_block_w);

        // Untilize
        untilize_init_short(reblock_cb_id);
        cb_wait_front(reblock_cb_id, out_block_w);
        cb_reserve_back(out_cb_id, out_block_w);
        untilize_block(reblock_cb_id, out_block_w, out_cb_id);
        cb_pop_front(reblock_cb_id, out_block_w);
        cb_push_back(out_cb_id, out_block_w);
        untilize_uninit(reblock_cb_id);

        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}  // reblock_and_untilize()

#endif

inline void pack_matmul_subblock(uint32_t cb_id, uint32_t out_subblock_num_tiles) {
    cb_reserve_back(cb_id, out_subblock_num_tiles);
    tile_regs_wait();
    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
        pack_tile(i, cb_id);
    }
    tile_regs_release();
    cb_push_back(cb_id, out_subblock_num_tiles);
}

namespace NAMESPACE {
void MAIN {
    uint32_t in0_block_w = get_compile_time_arg_val(0);             // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);       // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);     // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    uint32_t in0_subblock_h = get_compile_time_arg_val(4);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(5);    // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);  // out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_block_w = get_compile_time_arg_val(7);          // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    volatile uint32_t in0_num_blocks_h = get_compile_time_arg_val(8);
    volatile uint32_t in0_num_blocks_w = get_compile_time_arg_val(9);
    volatile uint32_t in1_num_blocks_w = get_compile_time_arg_val(10);
    uint32_t out_subblock_h = get_compile_time_arg_val(11);          // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(12);          // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13);  // out_subblock_h * out_subblock_w;
    bool tilize_in0 = get_compile_time_arg_val(14);
    bool untilize_out = get_compile_time_arg_val(15);

    uint32_t out_block_w = in1_block_w;
    bool spill = in0_num_blocks_w > 1;

    // CB indices
    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t matmul_partials_cb = tt::CBIndex::c_24;
    constexpr uint32_t tilized_in0_cb_id = tt::CBIndex::c_25;
    constexpr uint32_t untilize_mode_final_matmul_partials_cb = tt::CBIndex::c_26;
    constexpr uint32_t untilize_mode_reblock_cb = tt::CBIndex::c_27;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;

#ifdef FUSE_BIAS
    uint32_t bias_ntiles_w = get_compile_time_arg_val(16);
    constexpr uint32_t bias_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t out_for_bias_cb_id = tt::CBIndex::c_28;
    init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(out_for_bias_cb_id, bias_cb_id, out_cb_id);
#endif

    mm_init(in0_cb_id, in1_cb_id, out_cb_id);
    for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
#ifdef FUSE_BIAS
        uint32_t bias_block_offset = 0;
#endif
        for (uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
            bool enable_reload = false;
            for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                bool last_out = (in0_block_w_i == in0_num_blocks_w - 1);
                if (tilize_in0) {
                    tilize_in(in0_cb_id, in0_subblock_h, in0_block_w, in0_num_subblocks, tilized_in0_cb_id);
                    mm_init_short(tilized_in0_cb_id, in1_cb_id);
                    cb_wait_front(tilized_in0_cb_id, in0_block_num_tiles);
                } else {
                    cb_wait_front(in0_cb_id, in0_block_num_tiles);
                }
                cb_wait_front(in1_cb_id, in1_block_num_tiles);
                int in0_index_subblock_offset = 0;
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        tile_regs_acquire();
                        if (enable_reload) {
                            // Reconfigure input
                            copy_tile_to_dst_init_short(matmul_partials_cb);
                            UNPACK((llk_unpack_reconfig_data_format(1, matmul_partials_cb, 0, 0)));
                            MATH((llk_math_reconfig_data_format(1, matmul_partials_cb, 0, 0)));
                            cb_wait_front(matmul_partials_cb, out_subblock_num_tiles);
                            for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                                copy_tile(matmul_partials_cb, i, i);
                            }
                            cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);
                            // Reconfigure srcA back
                            mm_init_short_with_dt(
                                tilize_in0 ? tilized_in0_cb_id : in0_cb_id, in1_cb_id, matmul_partials_cb);
                        }  // enable_reload
                        // Compute output sub-block from in0_subblock x in1_subblock
                        int dst_index = 0;
                        int in0_index_h_offset = 0;
                        for (uint32_t h = 0; h < out_subblock_h; ++h) {
                            for (uint32_t w = 0; w < out_subblock_w; ++w) {
                                int in1_index_inner_dim_offset = 0;
                                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                                    matmul_tiles(
                                        tilize_in0 ? tilized_in0_cb_id : in0_cb_id,                  // in0_cb
                                        in1_cb_id,                                                   // in1_cb
                                        in0_index_subblock_offset + in0_index_h_offset + inner_dim,  // in0 tile
                                        in1_index_subblock_offset + in1_index_inner_dim_offset + w,  // in1 tile
                                        dst_index,                                                   // dst
                                        false);
                                    in1_index_inner_dim_offset += in1_block_w;
                                }  // for in0_block_w
                                ++dst_index;
                            }  // for out_subblock_w
                            in0_index_h_offset += in0_block_w;
                        }  // for out_subblock_h
#ifdef FUSE_BIAS
                           // if bias is to be added, add it to the data in dst before packing into the out cb
                        if (last_out) {
                            tile_regs_commit();
                            // first move the current result from dst to interim CB
                            pack_matmul_subblock(out_for_bias_cb_id, out_subblock_num_tiles);
                            // reconfig unpacker df for src B
                            // reconfig_data_format(out_for_bias_cb_id, bias_cb_id);
                            // bcast add data from bias_cb_id
                            cb_wait_front(bias_cb_id, bias_ntiles_w);
                            cb_wait_front(out_for_bias_cb_id, out_subblock_num_tiles);
                            add_bcast_rows_init_short(out_for_bias_cb_id, bias_cb_id);
                            // reconfig packer df for out
                            // pack_reconfig_data_format(out_cb_id);
                            tile_regs_acquire();
                            uint32_t i = 0;
                            for (uint32_t h = 0; h < out_subblock_h; ++h) {
                                uint32_t bcast_tile_i = bias_block_offset + in1_index_subblock_offset;
                                for (uint32_t w = 0; w < out_subblock_w; ++w) {
                                    add_tiles_bcast_rows(out_for_bias_cb_id, bias_cb_id, i, bcast_tile_i, i);
                                    ++bcast_tile_i;
                                    ++i;
                                }
                            }
                            // do not pop front bias as it may be used again for subsequent blocks
                            cb_pop_front(out_for_bias_cb_id, out_subblock_num_tiles);
                            // reconfig for matmul
                            mm_init_short(tilize_in0 ? tilized_in0_cb_id : in0_cb_id, in1_cb_id);
                            // reconfig unpacker df for srcB
                            // reconfig_data_format(in1_cb_id, in0_cb_id);
                        }
#endif

#ifdef SFPU_OP_INIT_ACTIVATION
                        if (last_out) {
                            SFPU_OP_INIT_ACTIVATION
                            for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                                SFPU_OP_FUNC_ACTIVATION
                            }
                        }
#endif

                        tile_regs_commit();

                        auto curr_matmul_out_cb =
                            last_out ? (untilize_out ? untilize_mode_final_matmul_partials_cb : out_cb_id)
                                     : matmul_partials_cb;
                        pack_matmul_subblock(curr_matmul_out_cb, out_subblock_num_tiles);
                        in1_index_subblock_offset += out_subblock_w;
                    }  // for in1_num_subblocks
#ifndef FUSE_BIAS
                       // untilizing is only supported if there is no bias
                    if (last_out && untilize_out) {
                        reconfig_data_format(
                            untilize_mode_final_matmul_partials_cb, untilize_mode_final_matmul_partials_cb);
                        reblock_and_untilize(
                            in1_num_subblocks,
                            out_subblock_num_tiles,
                            out_subblock_h,
                            out_subblock_w,
                            out_block_w,
                            untilize_mode_final_matmul_partials_cb,
                            untilize_mode_reblock_cb,
                            out_cb_id);
                        mm_init_short(tilize_in0 ? tilized_in0_cb_id : in0_cb_id, in1_cb_id);
                        reconfig_data_format(in1_cb_id, in0_cb_id);
                    }  // last_out
#endif
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }

                if (spill) {
                    enable_reload = true;
                }

                cb_pop_front(tilize_in0 ? tilized_in0_cb_id : in0_cb_id, in0_block_num_tiles);
                cb_pop_front(in1_cb_id, in1_block_num_tiles);
            }  // for in0_num_blocks_w
#ifdef FUSE_BIAS
            bias_block_offset += in1_block_w;
#endif
        }  // for in1_num_blocks_w
    }  // for in0_num_blocks_h
}  // MAIN
}  // namespace NAMESPACE
