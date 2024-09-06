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

inline void tilize_in(
    uint32_t in_cb_id,
    uint32_t in_subblock_h,
    uint32_t in_block_w,
    uint32_t in_num_subblocks,
    uint32_t out_cb_id) {

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
} // tilize_in()

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

        uint32_t within_block_index = 0;
        for (uint32_t h = 0; h < out_subblock_h; h++) {
            uint32_t block_offset = 0;

            // Reblock
            copy_tile_to_dst_init_short();
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
    } // reblock_and_untilize()
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

    constexpr uint32_t in0_block_w            = get_compile_time_arg_val(0); // inner block size in tiles
    constexpr uint32_t in0_num_subblocks      = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles    = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in0_subblock_h         = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks      = get_compile_time_arg_val(5); // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles    = get_compile_time_arg_val(6); //out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_per_core_w         = get_compile_time_arg_val(7); // out_subblock_w*in1_num_subblocks
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

    constexpr uint32_t out_block_w = in1_per_core_w;
    constexpr bool spill = in0_num_blocks_w > 1;

    // CB indices
    constexpr uint32_t in0_cb_id                                = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id                                = tt::CB::c_in1;
    constexpr uint32_t matmul_partials_cb                       = tt::CB::c_intermed0;
    constexpr uint32_t tilized_in0_cb_id                        = tt::CB::c_intermed1;
    constexpr uint32_t untilize_mode_reblock_cb                 = tt::CB::c_intermed2;
    constexpr uint32_t out_cb_id                                = tt::CB::c_out0;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;

    #ifdef FUSE_BIAS
    constexpr uint32_t bias_ntiles_w = get_compile_time_arg_val(16);
    constexpr uint32_t bias_cb_id                           = tt::CB::c_in2;
    constexpr uint32_t mm_out_cb_id = matmul_partials_cb;
    #else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
    #endif

    constexpr uint32_t mm_in0_cb_id = tilize_in0 ? tilized_in0_cb_id : in0_cb_id;

    mm_init(mm_in0_cb_id, in1_cb_id, out_cb_id);

    #ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
    #endif

    cb_wait_front(in1_cb_id, in1_block_num_tiles * in0_num_blocks_w * in1_num_blocks_w); // wait for all weights, in_num_blocks_w == 1

    for(uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        bool enable_reload = false;
        uint32_t in1_index_inner_dim_h_offset = 0;

        #ifdef PACK_RELU
        PACK(( llk_pack_relu_config(ReluType::NO_RELU) ));
        #endif

        uint32_t curr_matmul_out_cb = matmul_partials_cb;
        for(uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) { // inner dim of act (w)
            bool last_out = (in0_block_w_i == in0_num_blocks_w - 1);
            if constexpr (tilize_in0) {
                #if defined PACK_RELU and not defined FUSE_BIAS
                if (last_out) {
                    // if last block we pack the final result with relu enabled
                    PACK(( llk_pack_relu_config(ReluType::NO_RELU) ));
                }
                #endif
                unpack_reconfig_data_format_srca(in1_cb_id, in0_cb_id);
                tilize_in(in0_cb_id, in0_subblock_h, in0_block_w, in0_num_subblocks, tilized_in0_cb_id);
                mm_init_short();
                unpack_reconfig_data_format_srca(in0_cb_id, in1_cb_id);
            }
            cb_wait_front(mm_in0_cb_id, in0_block_num_tiles);

            if (last_out) {
                #if defined PACK_RELU and not defined FUSE_BIAS
                // if last block we pack the final result with relu enabled
                PACK(( llk_pack_relu_config(ReluType::ZERO_RELU) ));
                #endif
                curr_matmul_out_cb = mm_out_cb_id;
            }

            uint32_t in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                uint32_t in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                    if (enable_reload) {
                        // Reconfigure input
                        copy_tile_to_dst_init_short();
                        unpack_reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                        cb_wait_front(matmul_partials_cb, out_subblock_num_tiles);
                        tile_regs_acquire();
                        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                            copy_tile(matmul_partials_cb, i, i);
                        }
                        cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);
                        // Reconfigure srcA back
                        mm_init_short();
                        unpack_reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                    } else {
                        // just acquire
                        tile_regs_acquire();
                    }

                    // Compute output sub-block from in0_subblock x in1_subblock
                    uint32_t dst_index = 0;
                    uint32_t in0_index_h_offset = 0;
                    uint32_t in1_index_offset = in1_index_inner_dim_h_offset + in1_index_subblock_offset;
                    for (uint32_t h = 0; h < out_subblock_h; ++h) {
                        uint32_t in0_index_offset = in0_index_subblock_offset + in0_index_h_offset;
                        for (uint32_t w = 0; w < out_subblock_w; ++w) {
                            uint32_t in1_index_inner_dim_subblock_offset = 0;
                            uint32_t in1_index_offset_w = in1_index_offset + w;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                                matmul_tiles(mm_in0_cb_id,                    // in0_cb
                                                in1_cb_id,                                                     // in1_cb
                                                in0_index_offset + inner_dim,    // in0 tile
                                                in1_index_offset_w + in1_index_inner_dim_subblock_offset,    // in1 tile
                                                dst_index,                                                     // dst
                                                false);
                                in1_index_inner_dim_subblock_offset += in1_per_core_w;
                            } // for in0_block_w
                            ++dst_index;
                        } // for out_subblock_w
                        in0_index_h_offset += in0_block_w;
                    } // for out_subblock_h

                    #if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                    if (last_out) {
                        for (uint32_t i = 0; i < out_subblock_num_tiles; ++ i) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
                    }
                    #endif
                    tile_regs_commit();
                    pack_matmul_subblock(curr_matmul_out_cb, out_subblock_num_tiles);
                    in1_index_subblock_offset += out_subblock_w;
                } // for in1_num_subblocks
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if constexpr (spill) enable_reload = true;

            cb_pop_front(mm_in0_cb_id, in0_block_num_tiles);
            in1_index_inner_dim_h_offset += in1_block_num_tiles;
        } // for in0_num_blocks_w
        #ifdef FUSE_BIAS
        #ifdef PACK_RELU
        PACK(( llk_pack_relu_config(ReluType::ZERO_RELU) ));
        #endif
        add_bcast_rows_init_short();
        unpack_reconfig_data_format(in1_cb_id, matmul_partials_cb, mm_in0_cb_id, bias_cb_id);
        cb_wait_front(bias_cb_id, bias_ntiles_w);
        cb_wait_front(matmul_partials_cb, out_block_num_tiles);
        for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
            uint32_t in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                // reconfig packer df for out
                // pack_reconfig_data_format(out_cb_id);
                tile_regs_acquire();
                uint32_t i = 0;
                for (uint32_t h = 0; h < out_subblock_h; ++ h) {
                    uint32_t bcast_tile_i = in1_index_subblock_offset;
                    for (uint32_t w = 0; w < out_subblock_w; ++ w) {
                        add_tiles_bcast_rows(matmul_partials_cb, bias_cb_id, i, bcast_tile_i, i);
                        ++ bcast_tile_i;
                        ++ i;
                    }
                }
                // reconfig unpacker df for srcB
                // unpack_reconfig_data_format(in1_cb_id, in0_cb_id);

                #ifdef SFPU_OP_INIT_ACTIVATION
                for (uint32_t i = 0; i < out_subblock_num_tiles; ++ i) {
                    SFPU_OP_FUNC_ACTIVATION
                }
                #endif
                tile_regs_commit();
                // do not pop front bias as it may be used again for subsequent blocks
                cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);

                pack_matmul_subblock(untilize_mode_out_cb_id, out_subblock_num_tiles);
                in1_index_subblock_offset += out_subblock_w;
            } // for in1_num_subblocks
        }
        if constexpr(in0_num_blocks_h > 1) {
            if constexpr (!tilize_in0) {
                mm_init_short();
            }
            unpack_reconfig_data_format(matmul_partials_cb, in1_cb_id, bias_cb_id, mm_in0_cb_id);
        }
        #else
        if constexpr(untilize_out) {
            #ifdef PACK_RELU
            PACK(( llk_pack_relu_config(ReluType::NO_RELU) ));
            #endif
            unpack_reconfig_data_format(in1_cb_id, matmul_partials_cb, mm_in0_cb_id, untilize_mode_reblock_cb);
            for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                uint32_t in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                    reblock_and_untilize(
                                in1_num_subblocks,
                                out_subblock_num_tiles,
                                out_subblock_h,
                                out_subblock_w,
                                out_block_w,
                                matmul_partials_cb,
                                untilize_mode_reblock_cb,
                                out_cb_id);
                }
            }
            if constexpr(in0_num_blocks_h > 1) {
                if constexpr (!tilize_in) {
                    mm_init_short();
                }
                unpack_reconfig_data_format(matmul_partials_cb, in1_cb_id, untilize_mode_reblock_cb, mm_in0_cb_id);
            }
        }
        #endif
    } // for in0_num_blocks_h
    cb_pop_front(in1_cb_id, in1_block_num_tiles * in0_num_blocks_w * in1_num_blocks_w);
} // MAIN
} // NAMESPACE
