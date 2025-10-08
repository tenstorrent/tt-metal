// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "mod_div_lib.h"
#include "debug/dprint.h"

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

void MAIN {
    // RUNTIME ARGS
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
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(16);

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_5;
    // Reader will use this CB to pass the number of non-zero (nnz) entries in the sparsity tensor.
    constexpr uint32_t nnz_cb_id = tt::CBIndex::c_25;
    volatile uint32_t* nnz_addr_ptr;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
    constexpr uint32_t in1_transpose_tile = false;

    constexpr bool spill = num_blocks_inner_dim > 1;

    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
            // Check whether this batch is valid
            cb_wait_front(nnz_cb_id, 1);
            tensix_sync();
            cb_get_tile(nnz_cb_id, 0, &nnz_addr_ptr);
            // The first 4 entries have metadata, so we look at the 5th entry
            // for our value pushed from the reader.
            uint32_t nnz = nnz_addr_ptr[4];
            cb_release_tile(nnz_cb_id);
            cb_pop_front(nnz_cb_id, 1);

            if (nnz == 0) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                bool enable_reload = false;
                uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    bool last_out = block == (num_blocks_inner_dim - 1);
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

                            if (last_out) {
                                tile_regs_commit();
                                // Pack out to output buffer
                                cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                                tile_regs_wait();

                                PACK((pack_reconfig_data_format(mm_out_cb_id)));
                                PACK((llk_pack_reconfig_l1_acc(0)));

                                uint32_t start_dst_index = 0;
                                pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

                            } else {
                                tile_regs_commit();
                                // Wait for tiles in output buffer to be written out since interm and output share
                                // memory
                                if (block == 0) {
                                    // cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                                    out_num_tiles_to_wait += out_subblock_num_tiles;
                                }
                                // Move partial result to interm buffer
                                cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                                tile_regs_wait();

                                if (block == 0) {  // no accumulation for first iteration
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }

                                uint32_t start_dst_index = 0;
                                pack_tile_block(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }

                    // Last iteration does spill and reload to output buffer
                    if (block < num_blocks_inner_dim - 2) {
                        cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
                        cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
                    }
                    if (block == num_blocks_inner_dim - 2) {
                        enable_reload = true;
                    }  // reload when last iteration

                    cb_pop_front(in0_cb_id, in0_block_num_tiles);
                    cb_pop_front(in1_cb_id, in1_block_num_tiles);
                }

                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
                    // reconfigure unpacker df for src A
                    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
                    // reconfigure init for matmul
                    mm_block_init_short(
                        in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
}
}  // namespace NAMESPACE
