// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "mod_div_lib.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.

namespace NAMESPACE {

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    // Reconfigure input
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Reconfigure srcA back
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}

template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t interm_cb_id,
    uint32_t out_cb_id) {
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    cb_wait_front(interm_cb_id, num_tiles_in_row_of_subblocks);

    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;

        cb_reserve_back(out_cb_id, out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                copy_tile(interm_cb_id, tile_index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dst<out_subblock_w, out_block_w>(out_cb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(out_cb_id, out_block_w);

        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

void MAIN {
// RUNTIME ARGS
#ifdef MATMUL_DRAM_SHARDED
    const bool is_worker_core = get_arg_val<uint32_t>(0) == 1;
    // if not worker core, skip
    if (not is_worker_core) {
        return;
    }
#endif

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);     // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(13);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_24;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t mm_out_cb_id = mm_partials_cb_id;
#else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
#endif

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                bool enable_reload = false;
                uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

#ifdef PACK_RELU
                // for each batch we start with relu disabled so that intermediate results are not relu'd
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                }
#endif

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    bool last_out = block == (num_blocks_inner_dim - 1);
// Configure packer once for pack out without Bias
#if not defined FUSE_BIAS and defined PACK_RELU
                    if (last_out) {
                        // if last block we pack the final result with relu enabled
                        PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
                    }
#endif

                    cb_wait_front(in0_cb_id, in0_block_num_tiles);
                    cb_wait_front(in1_cb_id, in1_block_num_tiles);

                    int in0_index_subblock_offset = 0;
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                        int in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                            tile_regs_acquire();
                            if (enable_reload) {
                                reload_from_cb_to_dst(
                                    in0_cb_id,
                                    in1_cb_id,
                                    mm_partials_cb_id,
                                    in1_transpose_tile,
                                    out_subblock_num_tiles,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                            }

#ifndef SKIP_COMPUTE
                            // Compute output sub-block
                            uint32_t dst_index =
                                0;  // start at 0, each call to matmul_block internally increments dst_index
                            uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                            uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                            // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w
                            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                                // accumulation is done by iterating matmul_block across inner dim
                                // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride
                                // in0
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
                                in1_index += in1_block_w;  // to stride down by 1 need to stride by in_per_core_w
                                                           // (should be called in1_block_w)
                            }

#endif  // SKIP_COMPUTE

                            if (last_out) {
// If we fuse bias, we will pack out and run bias + optional sfpu in a separate loop
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                    SFPU_OP_FUNC_ACTIVATION
                                }
#endif
                                tile_regs_commit();
                                // Pack out to output buffer
                                cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                                tile_regs_wait();

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                                if (block == 0) {  // no accumulation for first iteration
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#else
                                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif

                                uint32_t start_dst_index = 0;
                                matmul_pack_tile(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

                            } else {
                                tile_regs_commit();
                                // Wait for tiles in output buffer to be written out since interm and output share
                                // memory
                                if (block == 0) {
                                    cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                                    out_num_tiles_to_wait += out_subblock_num_tiles;
                                }
                                // Move partial result to interm buffer
                                cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                                tile_regs_wait();

#ifdef PACKER_L1_ACC
                                if (block == 0) {  // no accumulation for first iteration
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#endif

                                uint32_t start_dst_index = 0;
                                matmul_pack_tile(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                    if (block < num_blocks_inner_dim - 1) {
                        // Wait for l1 accumulation to populate interm buffer,
                        // then pop to update fifo rd pointer
                        cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
                        cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
                    }
                    // never reload when with bias, bias uses interm buffer
                    enable_reload = false;
#else
                    // Last iteration does spill and reload to output buffer
                    if (block < num_blocks_inner_dim - 2) {
                        cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
                        cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
                    }
                    if (block == num_blocks_inner_dim - 2) {
                        enable_reload = true;
                    }  // reload when last iteration
#endif
#else
                    if constexpr (spill) {
                        enable_reload = true;
                    }
#endif

                    cb_pop_front(in0_cb_id, in0_block_num_tiles);
                    cb_pop_front(in1_cb_id, in1_block_num_tiles);
                }

#ifdef FUSE_BIAS
#ifdef PACK_RELU
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                reconfig_data_format(in1_cb_id, mm_partials_cb_id, in0_cb_id, bias_cb_id);
                add_bcast_rows_init_short(mm_partials_cb_id, bias_cb_id);
                // reconfigure unpacker df for src B
                cb_wait_front(bias_cb_id, in1_block_w);
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        // Redundant wait since we know data was just pushed
                        cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
                        tile_regs_acquire();
                        for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                            uint32_t bcast_tile_idx = in1_index_subblock_offset;
                            for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                                add_tiles_bcast_rows(mm_partials_cb_id, bias_cb_id, i, bcast_tile_idx, i);
                                bcast_tile_idx++;
                            }
                        }
// if there's no SFPU fusion, we commit the regs so packer can start packing
#ifndef SFPU_OP_INIT_ACTIVATION
                        tile_regs_commit();
#endif

                        cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);

// sfpu activation
#ifdef SFPU_OP_INIT_ACTIVATION
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
                        tile_regs_commit();
#endif

                        // Pack out to output buffer
                        cb_reserve_back(untilize_mode_out_cb_id, out_subblock_num_tiles);
                        tile_regs_wait();
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, untilize_mode_out_cb_id);
                        }
                        tile_regs_release();
                        cb_push_back(untilize_mode_out_cb_id, out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    }
                }
                if constexpr (num_blocks_w_dim > 1) {
                    cb_pop_front(bias_cb_id, in1_block_w);
                }
#endif  // FUSE_BIAS
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif  // PACK_RELU
#ifndef FUSE_BIAS
                    reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif  // FUSE_BIAS
                    pack_untilize_dst_init_short<out_subblock_w, out_block_w>(out_cb_id);
                    copy_tile_to_dst_init_short();
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            in1_num_subblocks, out_subblock_num_tiles, out_subblock_h, mm_partials_cb_id, out_cb_id);
                    }
                    pack_untilize_uninit(mm_partials_cb_id);
                }
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
                    // reconfigure init for matmul
                    mm_block_init_short(
                        in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
#ifdef FUSE_BIAS
                    // reconfigure unpacker df for src A and src B
                    reconfig_data_format(mm_partials_cb_id, in1_cb_id, bias_cb_id, in0_cb_id);
#else
                    // reconfigure unpacker df for src A
                    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
#endif
                }
            }
        }
    }
}
}  // namespace NAMESPACE
