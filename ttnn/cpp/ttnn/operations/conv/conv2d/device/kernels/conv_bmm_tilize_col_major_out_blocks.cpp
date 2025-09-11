// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "mod_div_lib.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
// #include "debug/dprint.h"
#include "compute_kernel_api/untilize.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#define DEBUG_PRINT 0

#ifdef SPLIT_READER
template <bool init_tilize = true, bool uninit_tilize = true>
__attribute__((noinline)) void tilize_in(
#else
template <bool init_tilize = true, bool uninit_tilize = true>
void tilize_in(
#endif
    uint32_t in_cb_id, uint32_t in_block_w, uint32_t in_num_subblocks, uint32_t out_cb_id) {
    if constexpr (init_tilize) {
        fast_tilize_init_with_dt(in_cb_id, in_block_w, out_cb_id);
    }
    for (uint32_t in_subblock = 0; in_subblock < in_num_subblocks; ++in_subblock) {
        cb_wait_front(in_cb_id, in_block_w);
        cb_reserve_back(out_cb_id, in_block_w);
        fast_tilize_block(in_cb_id, in_block_w, out_cb_id);
        cb_push_back(out_cb_id, in_block_w);
        cb_pop_front(in_cb_id, in_block_w);
    }
    if constexpr (uninit_tilize) {
        fast_tilize_uninit(in_cb_id, out_cb_id);
    }
}  // tilize_in()

#ifdef ACTIVATION_REUSE
template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id>
inline void tilize_single_block() {
    cb_wait_front(in_cb_id, in_block_w);
    fast_tilize_block(in_cb_id, in_block_w, out_cb_id);
    cb_pop_front(in_cb_id, in_block_w);
}

template <uint32_t in_cb_id, uint32_t window_reuse_offset>
inline uint32_t update_in_cb(uint32_t in_cb_addr) {
    UNPACK((get_local_cb_interface(in_cb_id).fifo_rd_ptr = in_cb_addr));
    return in_cb_addr + window_reuse_offset;
}

template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id, uint32_t tilized_cb_row_offset>
inline void tilize_single_block_with_out_cb_update(uint32_t& out_cb_addr) {
    PACK((get_local_cb_interface(out_cb_id).fifo_wr_ptr = out_cb_addr));
    PACK((out_cb_addr += tilized_cb_row_offset));
    tilize_single_block<in_cb_id, in_block_w, out_cb_id>();
}

template <
    uint32_t in_cb_id,
    uint32_t in_block_w,
    uint32_t in_num_subblocks,
    uint32_t out_cb_id,
    uint32_t window_reuse_offset,
    uint32_t image_width_in_tiles>
inline void tilize_in_reuse(uint32_t in_cb_start_addr) {
    fast_tilize_init_with_dt(in_cb_id, in_block_w, out_cb_id);

    constexpr uint32_t num_image_rows = in_num_subblocks / image_width_in_tiles;
    constexpr uint32_t last_row_columns = in_num_subblocks % image_width_in_tiles;
    uint32_t in_cb_addr = in_cb_start_addr;

    // full image rows
    for (uint32_t image_row = 0; image_row < num_image_rows; ++image_row) {
        // each time we start processing a new image row, we need to update the read pointer to the appropriate offset
        // from the start of the CB to match the reader behavior
        in_cb_addr = update_in_cb<in_cb_id, window_reuse_offset>(in_cb_addr);
        for (uint32_t image_col = 0; image_col < image_width_in_tiles; ++image_col) {
            cb_reserve_back(out_cb_id, in_block_w);
            tilize_single_block<in_cb_id, in_block_w, out_cb_id>();
            cb_push_back(out_cb_id, in_block_w);
        }
    }

    // partial last image row
    in_cb_addr = update_in_cb<in_cb_id, window_reuse_offset>(in_cb_addr);
    for (uint32_t image_col = 0; image_col < last_row_columns; ++image_col) {
        cb_reserve_back(out_cb_id, in_block_w);
        tilize_single_block<in_cb_id, in_block_w, out_cb_id>();
        cb_push_back(out_cb_id, in_block_w);
    }

    fast_tilize_uninit(in_cb_id, out_cb_id);
}

template <
    uint32_t in1_cb_id,
    uint32_t in2_cb_id,
    uint32_t in_block_w,
    uint32_t in1_num_subblocks,
    uint32_t in2_num_subblocks,
    uint32_t out_cb_id,
    uint32_t out_cb_tiles,
    uint32_t window_reuse_offset,
    uint32_t tilized_cb_row_offset,
    uint32_t tilized_cb_second_reader_offset,
    uint32_t image_width_in_tiles>
inline void tilize_in_reuse_split_reader(uint32_t act_cb_start_address, uint32_t act_cb_second_reader_start_address) {
    // with activation reuse, the activation buffers are sized to fit one output image width only,
    // so we need to interleave waits and pops on the two buffers to allow parallelization;
    // we reserve back tilized CB to store whole act block h - and then we update write pointers so that
    // we fill in first row of the first hald (NCRISC), first row of the second half (BRISC) and so on
    cb_reserve_back(out_cb_id, out_cb_tiles);
    fast_tilize_init_with_dt(in1_cb_id, in_block_w, out_cb_id);

    uint32_t in1_cb_addr = act_cb_start_address;
    uint32_t in2_cb_addr = act_cb_second_reader_start_address;

    uint32_t out_cb_addr, out_cb_addr_second_reader;
    PACK((out_cb_addr = get_local_cb_interface(out_cb_id).fifo_wr_ptr));
    PACK((out_cb_addr_second_reader = out_cb_addr + tilized_cb_second_reader_offset));

    constexpr uint32_t min_num_subblocks =
        in1_num_subblocks > in2_num_subblocks ? in2_num_subblocks : in1_num_subblocks;
    constexpr uint32_t min_num_image_rows = min_num_subblocks / image_width_in_tiles;
    constexpr uint32_t leftover_in1 = in1_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t leftover_in2 = in2_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t max_leftover = leftover_in1 > leftover_in2 ? leftover_in1 : leftover_in2;

    // process minimum number of image rows for both readers in the same loop
    for (uint32_t image_row = 0; image_row < min_num_image_rows; ++image_row) {
        // each time we start processing a new image row, we need to update the read pointer to the appropriate offset
        // from the start of the CB to match the reader behavior
        in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
        in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
        for (uint32_t image_col = 0; image_col < image_width_in_tiles; ++image_col) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                out_cb_addr);
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                out_cb_addr_second_reader);
        }
    }

    // leftover image rows if one reader had more rows than the other
    in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
    in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
    for (uint32_t image_col = 0; image_col < max_leftover; ++image_col) {
        if (image_col < leftover_in1) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                out_cb_addr);

            if (image_col == image_width_in_tiles - 1) {
                in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
            }
        }

        if (image_col < leftover_in2) {
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                out_cb_addr_second_reader);

            if (image_col == image_width_in_tiles - 1) {
                in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
            }
        }
    }

    cb_push_back(out_cb_id, out_cb_tiles);
    fast_tilize_uninit(in2_cb_id, out_cb_id);
}
#endif

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
            pack_untilize_dest<out_subblock_w, out_block_w>(out_cb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(out_cb_id, out_block_w);
        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t reader_num_h_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(5);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(6);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(7);  // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    constexpr uint32_t in0_num_blocks_h = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_blocks_w = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_blocks_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(11);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13);  // out_subblock_h * out_subblock_w;
    constexpr bool height_sharded = get_compile_time_arg_val(14);
    constexpr bool untilize_out = get_compile_time_arg_val(15);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in0_pretilize_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t in0_cb_second_reader_id = get_compile_time_arg_val(21);
    constexpr uint32_t matmul_partials_cb = get_compile_time_arg_val(22);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(23);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(24);
    constexpr bool partials_cb_uses_output = get_compile_time_arg_val(26);
    constexpr uint32_t in0_nblocks_w_tilize = get_compile_time_arg_val(27);
    constexpr bool check_skip_compute = get_compile_time_arg_val(28);

#ifdef ACTIVATION_REUSE
    constexpr uint32_t image_width_in_tiles = get_compile_time_arg_val(29);
    constexpr uint32_t window_reuse_offset = get_compile_time_arg_val(30);
#ifdef SPLIT_READER
    constexpr uint32_t tilized_cb_row_offset = get_compile_time_arg_val(31);
    constexpr uint32_t tilized_cb_second_reader_offset = get_compile_time_arg_val(32);
#endif
#endif

    constexpr uint32_t out_block_num_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    constexpr uint32_t out_block_w = in1_block_w;
    constexpr bool spill = in0_num_blocks_w > 1;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;

#ifdef FUSE_BIAS
    constexpr uint32_t bias_ntiles_w = get_compile_time_arg_val(16);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(17);
    uint32_t bias_block_offset = 0;
    constexpr uint32_t mm_out_cb_id = matmul_partials_cb;
#else
    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
#endif

    constexpr uint32_t mm_in0_cb_id = height_sharded ? tilized_in0_cb_id : in0_cb_id;

#ifdef SPLIT_READER
    constexpr bool split_reader = true;
    constexpr uint32_t in0_num_subblocks_read_last = reader_num_h_subblocks / 2;
    constexpr uint32_t in0_num_subblocks_read = reader_num_h_subblocks - in0_num_subblocks_read_last;
#else
    constexpr bool split_reader = false;
    constexpr uint32_t in0_num_subblocks_read = reader_num_h_subblocks;
#endif

    // For block sharded conv2d, compute kernels may be scheduled on cores that only need tilize
    // operations (input grid) while actual matmul occurs on different cores (output grid).
    // Skip dummy compute operations when possible to reduce di/dt issues, but allow dummy
    // tilize operations on cores without input data for code simplicity.
    bool skip_compute = false;
    if constexpr (check_skip_compute) {
        skip_compute = (bool)get_arg_val<uint32_t>(0);
    }
#ifdef ACTIVATION_REUSE
    // if activation reuse is enabled, we need to update read pointers of the act buffers
    // each time we pass on to the new output image row to match the reader behavior
    uint32_t act_cb_start_address = get_local_cb_interface(in0_cb_id).fifo_rd_ptr;
#ifdef SPLIT_READER
    const uint32_t out_cb_tiles = in0_block_w * (in0_num_subblocks_read + in0_num_subblocks_read_last);
    const uint32_t tilized_cb_start_address = get_local_cb_interface(tilized_in0_cb_id).fifo_wr_ptr;
    const uint32_t act_cb_second_reader_start_address = get_local_cb_interface(in0_cb_second_reader_id).fifo_rd_ptr;
#endif
#endif

    mm_block_init(mm_in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif
    UNPACK(uint32_t partials_cb_read_ptr = get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr;)
    PACK(uint32_t partials_cb_write_ptr = get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr;)
    // in1 num blocks w is the outer loop. Output blocks are computed in col major order.
    for (uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
        for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
            bool enable_reload = false;

#ifdef PACK_RELU
            // for each output block we start we relu disabled so that intermediate results are not relu'd
            PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif
            if constexpr (partials_cb_uses_output) {
                UNPACK(partials_cb_read_ptr = get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr;)
                PACK(partials_cb_write_ptr = get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr;)
            }
            uint32_t curr_matmul_out_cb = matmul_partials_cb;
            for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                bool last_out = (in0_block_w_i == in0_num_blocks_w - 1);
                if constexpr (!height_sharded) {
                    if (in0_block_w_i % in0_nblocks_w_tilize == 0) {
#if defined PACK_RELU and not defined FUSE_BIAS
                        if (last_out) {
                            // if last block we pack the final result with relu enabled
                            PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                        }
#endif
#ifdef PACKER_L1_ACC
                        pack_reconfig_data_format(curr_matmul_out_cb, tilized_in0_cb_id);
                        pack_reconfig_l1_acc(0);
#endif

                        tilize_in(in0_pretilize_cb_id, in0_block_w, reader_num_h_subblocks, tilized_in0_cb_id);

                        mm_block_init_short_with_both_dt(
                            in0_cb_id,
                            in1_cb_id,
                            in0_pretilize_cb_id,
                            in0_pretilize_cb_id,
                            false,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                    }
                } else {
#if defined PACK_RELU and not defined FUSE_BIAS
                    if (last_out) {
                        // if last block we pack the final result with relu enabled
                        PACK((llk_pack_relu_config(ReluType::NO_RELU)));
                    }
#endif
#ifdef PACKER_L1_ACC
                    pack_reconfig_data_format(curr_matmul_out_cb, tilized_in0_cb_id);
                    pack_reconfig_l1_acc(0);
#endif

#ifndef ACTIVATION_REUSE
                    tilize_in<true, !split_reader>(in0_cb_id, in0_block_w, in0_num_subblocks_read, tilized_in0_cb_id);
#ifdef SPLIT_READER
                    tilize_in<false, true>(
                        in0_cb_second_reader_id, in0_block_w, in0_num_subblocks_read_last, tilized_in0_cb_id);
#endif
#else
#ifndef SPLIT_READER
                    tilize_in_reuse<
                        in0_cb_id,
                        in0_block_w,
                        in0_num_subblocks_read,
                        tilized_in0_cb_id,
                        window_reuse_offset,
                        image_width_in_tiles>(act_cb_start_address);
#else
                    PACK((get_local_cb_interface(tilized_in0_cb_id).fifo_wr_ptr = tilized_cb_start_address));
                    tilize_in_reuse_split_reader<
                        in0_cb_id,
                        in0_cb_second_reader_id,
                        in0_block_w,
                        in0_num_subblocks_read,
                        in0_num_subblocks_read_last,
                        tilized_in0_cb_id,
                        out_cb_tiles,
                        window_reuse_offset,
                        tilized_cb_row_offset,
                        tilized_cb_second_reader_offset,
                        image_width_in_tiles>(act_cb_start_address, act_cb_second_reader_start_address);
#endif
#endif

                    mm_block_init_short_with_both_dt(
                        mm_in0_cb_id,
                        in1_cb_id,
                        in0_cb_id,
                        in0_cb_id,
                        false,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                }

                cb_wait_front(mm_in0_cb_id, in0_block_num_tiles);

                uint32_t in0_index_subblock_offset = 0;
                if constexpr (check_skip_compute) {
                    if (skip_compute) {
                        cb_pop_front(mm_in0_cb_id, in0_block_num_tiles);
                        continue;
                    }
                }

                cb_wait_front(in1_cb_id, in1_block_num_tiles);

                if (last_out) {
#if defined PACK_RELU and not defined FUSE_BIAS
                    // if last block we pack the final result with relu enabled
                    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
#ifndef FUSE_BIAS
                    curr_matmul_out_cb = mm_out_cb_id;
#endif
                }

#ifdef PACKER_L1_ACC
                pack_reconfig_data_format(curr_matmul_out_cb);
#endif
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        if (enable_reload) {
                            // Reconfigure input
                            copy_tile_to_dst_init_short_with_dt(in1_cb_id, matmul_partials_cb);
                            cb_wait_front(matmul_partials_cb, out_subblock_num_tiles);
                            tile_regs_acquire();

                            uint32_t start_dst_index = 0;
                            uint32_t start_tile_index = 0;
                            copy_block_matmul_partials(
                                matmul_partials_cb, start_tile_index, start_dst_index, out_subblock_num_tiles);

                            cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);
                            // Reconfigure srcA back
                            mm_block_init_short_with_dt(
                                mm_in0_cb_id,
                                in1_cb_id,
                                matmul_partials_cb,
                                false,
                                out_subblock_w,
                                out_subblock_h,
                                in0_block_w);
                        } else {
                            // just acquire
                            tile_regs_acquire();
                        }

                        // Compute output sub-block
                        uint32_t dst_index =
                            0;  // start at 0, each call to matmul_block internally increments dst_index
                        uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                        uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                        // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                        for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; inner_dim_idx++) {
                            // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                            // accumulation is done by iterating matmul_block across inner dim
                            // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                            matmul_block(
                                mm_in0_cb_id,
                                in1_cb_id,
                                in0_index,
                                in1_index,
                                dst_index,
                                false,
                                out_subblock_w,
                                out_subblock_h,
                                in0_block_w);
                            in0_index++;               // stride right by 1
                            in1_index += in1_block_w;  // to stride down by 1 need to stride by in_per_core_w (should be
                                                       // called in1_block_w)
                        }

#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
                        if (last_out) {
                            for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                                SFPU_OP_FUNC_ACTIVATION
                            }
                        }
#endif
                        tile_regs_commit();
                        cb_reserve_back(curr_matmul_out_cb, out_subblock_num_tiles);
                        tile_regs_wait();

#ifdef PACKER_L1_ACC
                        // no accumulation for first iteration, last iteration
                        // accumulation happens with copying tiles to dst
                        if (in0_block_w_i == 0) {
                            pack_reconfig_l1_acc(0);
                        } else if (last_out) {
// Fuse bias always uses intermediate buffer
// no need to spill and reload last iteration
#ifdef FUSE_BIAS
                            pack_reconfig_l1_acc(1);
#else
                            pack_reconfig_l1_acc(0);
#endif
                        } else {
                            pack_reconfig_l1_acc(1);
                        }
#endif

                        uint32_t start_dst_index = 0;
                        pack_tile_block(start_dst_index, curr_matmul_out_cb, out_subblock_num_tiles);

                        tile_regs_release();
                        cb_push_back(curr_matmul_out_cb, out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    }  // for in1_num_subblocks
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }
                if (curr_matmul_out_cb == matmul_partials_cb) {
                    if constexpr (!partials_cb_uses_output) {
                        UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr;)
                        PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr;)
                    }
                }
#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                if (in0_block_w_i < in0_num_blocks_w - 1) {
                    // Wait for l1 accumulation to populate interm buffer,
                    // then pop to update fifo rd pointer
                    cb_wait_front(matmul_partials_cb, out_block_num_tiles);
                    cb_pop_front(matmul_partials_cb, out_block_num_tiles);
                    if constexpr (spill) {
                        UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
                        PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr);
                    }
                }
                // never reload when with bias, bias uses interm buffer
                enable_reload = false;
#else
                // Last iteration does spill and reload to output buffer
                if (in0_block_w_i < in0_num_blocks_w - 2) {
                    cb_wait_front(matmul_partials_cb, out_block_num_tiles);
                    cb_pop_front(matmul_partials_cb, out_block_num_tiles);
                    if constexpr (spill) {
                        UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
                        PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr);
                    }
                }
                if (in0_block_w_i == in0_num_blocks_w - 2) {
                    enable_reload = true;
                }
#endif
#else
                if constexpr (spill) {
                    enable_reload = true;

#ifdef FUSE_BIAS
                    if (!last_out) {
                        UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
                        PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr);
                    }
#else
                    if (!last_out) {
                        UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
                    }
                    if (in0_block_w_i < in0_num_blocks_w - 2) {
                        PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr);
                    }
#endif
                }
#endif

                cb_pop_front(mm_in0_cb_id, in0_block_num_tiles);
                cb_pop_front(in1_cb_id, in1_block_num_tiles);
            }  // for in0_num_blocks_w
            if constexpr (matmul_partials_cb == mm_out_cb_id && partials_cb_uses_output) {
                UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
            }
            if constexpr (check_skip_compute) {
                if (skip_compute) {
                    continue;
                }
            }
#ifdef FUSE_BIAS
#ifdef PACK_RELU
            // if last block we pack the final result with relu enabled
            PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
            pack_reconfig_data_format(matmul_partials_cb, untilize_mode_out_cb_id);
#ifdef PACKER_L1_ACC
            pack_reconfig_l1_acc(0);
#endif
            reconfig_data_format(in1_cb_id, matmul_partials_cb, mm_in0_cb_id, bias_cb_id);
            add_bcast_rows_init_short(matmul_partials_cb, bias_cb_id);

            cb_wait_front(bias_cb_id, bias_ntiles_w);
            cb_wait_front(matmul_partials_cb, out_block_num_tiles);
            for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                uint32_t in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                    tile_regs_acquire();
                    uint32_t i = 0;
                    for (uint32_t h = 0; h < out_subblock_h; ++h) {
                        uint32_t bcast_tile_i = bias_block_offset + in1_index_subblock_offset;
                        for (uint32_t w = 0; w < out_subblock_w; ++w) {
                            add_tiles_bcast_rows(matmul_partials_cb, bias_cb_id, i, bcast_tile_i, i);
                            ++bcast_tile_i;
                            ++i;
                        }
                    }

#ifdef SFPU_OP_INIT_ACTIVATION
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                        SFPU_OP_FUNC_ACTIVATION
                    }
#endif
                    tile_regs_commit();
                    // do not pop front bias as it may be used again for subsequent blocks
                    cb_pop_front(matmul_partials_cb, out_subblock_num_tiles);

                    cb_reserve_back(untilize_mode_out_cb_id, out_subblock_num_tiles);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, untilize_mode_out_cb_id);
                    }
                    tile_regs_release();
                    cb_push_back(untilize_mode_out_cb_id, out_subblock_num_tiles);

                    in1_index_subblock_offset += out_subblock_w;
                }  // for in1_num_subblocks
            }  // in0_num_subblocks
            if constexpr (untilize_out) {
                UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
                PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr);
            }
#endif
            if constexpr (untilize_out) {
#if defined PACKER_L1_ACC
                pack_reconfig_data_format(matmul_partials_cb, out_cb_id);
                pack_reconfig_l1_acc(0);
#endif
#ifdef PACK_RELU
                PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif
#ifndef FUSE_BIAS
                reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
#endif

#ifdef PACKER_UNTILIZE
                pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
                copy_tile_to_dst_init_short(matmul_partials_cb);
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    reblock_and_untilize<out_subblock_w, out_block_w>(
                        in1_num_subblocks, out_subblock_num_tiles, out_subblock_h, matmul_partials_cb, out_cb_id);
                }
                pack_untilize_uninit(matmul_partials_cb);
#else
                untilize_init(matmul_partials_cb);
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    for (uint32_t out_block_h_i = 0; out_block_h_i < out_subblock_h; ++out_block_h_i) {
                        cb_wait_front(matmul_partials_cb, out_block_w);
                        cb_reserve_back(out_cb_id, out_block_w);
                        untilize_block(matmul_partials_cb, out_block_w, out_cb_id);
                        cb_push_back(out_cb_id, out_block_w);
                        cb_pop_front(matmul_partials_cb, out_block_w);
                    }
                }
                untilize_uninit(matmul_partials_cb);
#endif
            }
            if constexpr ((in1_num_blocks_w > 1 || in0_num_blocks_h > 1)) {
#ifdef FUSE_BIAS
                reconfig_data_format(matmul_partials_cb, in1_cb_id, bias_cb_id, mm_in0_cb_id);
#else
                reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
#endif
            }
        }  // for in0_num_blocks_h
#ifdef FUSE_BIAS
        bias_block_offset += in1_block_w;
#endif
    }  // for in1_num_blocks_w
}  // MAIN
}  // namespace NAMESPACE
