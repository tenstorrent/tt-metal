// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include <tt-metalium/constants.hpp>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"

// Slightly modified from compute_common.hpp
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t in0_num_subblocks,
    const uint32_t in1_num_subblocks,
    const uint32_t in0_block_w,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const bool transpose) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;

    reconfig_data_format(in1_cb, in0_cb);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            tile_regs_commit();

            cb_reserve_back(out_cb, out_subblock_num_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
            cb_push_back(out_cb, out_subblock_num_tiles);
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
}

template <uint32_t rows, uint32_t cols>
void add_bias_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t dst_tiles = 1;

    add_bcast_rows_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, cols);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            tile_regs_acquire();
            // Add jth tile of bias to each column j of in0_cb
            add_tiles_bcast_rows(in0_cb, in1_cb, 0, j, 0);
            tile_regs_commit();
            cb_pop_front(in0_cb, dst_tiles);
            cb_reserve_back(in0_cb, dst_tiles);
            tile_regs_wait();
            pack_tile(0, in0_cb);
            cb_push_back(in0_cb, dst_tiles);
            tile_regs_release();
        }
    }
}

template <uint32_t num_tiles>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_cb has num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    constexpr uint32_t dst_tiles = 1;

    add_tiles_init(in0_cb, in1_cb);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        add_tiles(in0_cb, in1_cb, 0, 0, 0);
        tile_regs_commit();
        cb_pop_front(in0_cb, dst_tiles);
        cb_pop_front(in1_cb, dst_tiles);
        cb_reserve_back(in0_cb, dst_tiles);
        tile_regs_wait();
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, dst_tiles);
        tile_regs_release();
    }
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_vol2col_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_vol2col_tiled = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight_tiled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias_tiled = get_compile_time_arg_val(3);
    constexpr uint32_t cb_matmul_interm_tiled = get_compile_time_arg_val(4);
    constexpr uint32_t cb_matmul_result_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_reduction_tiled = get_compile_time_arg_val(6);
    constexpr uint32_t cb_worker_ack_back = get_compile_time_arg_val(7);

    constexpr uint32_t num_patches = get_compile_time_arg_val(8);
    constexpr uint32_t matmul_M_t = get_compile_time_arg_val(9);
    constexpr uint32_t matmul_K_t = get_compile_time_arg_val(10);
    constexpr uint32_t matmul_N_t = get_compile_time_arg_val(11);

    constexpr bool use_bias = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t T_out = get_compile_time_arg_val(13);
    constexpr uint32_t H_out = get_compile_time_arg_val(14);
    constexpr uint32_t W_out = get_compile_time_arg_val(15);
    constexpr uint32_t T_block_size = get_compile_time_arg_val(16);
    constexpr uint32_t H_block_size = get_compile_time_arg_val(17);
    constexpr uint32_t W_block_size = get_compile_time_arg_val(18);
    constexpr uint32_t C_out_num_blocks = get_compile_time_arg_val(19);

    // matmul parameters
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(21);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(22);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(23);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(24);

    constexpr uint32_t semaphore_id = get_compile_time_arg_val(25);

    constexpr uint32_t patch_tiles = matmul_M_t * matmul_K_t;
    constexpr uint32_t weight_tiles = matmul_K_t * matmul_N_t;
    constexpr uint32_t output_tiles = matmul_M_t * matmul_N_t;

    mm_init(cb_vol2col_tiled, cb_weight_tiled, cb_matmul_interm_tiled);

    // Load range parameters
    uint32_t argidx = 0;
    const uint32_t c_in_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_in_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_reducer = get_arg_val<uint32_t>(argidx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(argidx++);

    for (uint32_t c_in_block = c_in_block_start; c_in_block < c_in_block_end; c_in_block++) {
        // Process only assigned C_out blocks
        for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
            // Wait for new weights and bias
            cb_wait_front(cb_weight_tiled, weight_tiles);

            if constexpr (use_bias) {
                if (is_reducer) {
                    cb_wait_front(cb_bias_tiled, matmul_N_t);
                }
            }

            // 3D blocking loops over assigned ranges:
            for (uint32_t t_block = t_out_start; t_block < t_out_end; t_block += T_block_size) {
                for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                    for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                        // Tilize row-major patches
                        uint32_t patch_rows_left = num_patches;
                        tilize_init(cb_vol2col_rm, matmul_K_t, cb_vol2col_tiled);
                        for (uint32_t patch_t = 0; patch_t < matmul_M_t; patch_t++) {
                            // Reader produces row pages, which may not be tile aligned. Wait on the correct number of
                            // rows.
                            uint32_t current_patch_rows = patch_rows_left < tt::constants::TILE_HEIGHT
                                                              ? patch_rows_left
                                                              : tt::constants::TILE_HEIGHT;
                            cb_wait_front(cb_vol2col_rm, current_patch_rows);
                            cb_reserve_back(cb_vol2col_tiled, matmul_K_t);
                            tilize_block(cb_vol2col_rm, matmul_K_t, cb_vol2col_tiled);
                            cb_push_back(cb_vol2col_tiled, matmul_K_t);
                            cb_pop_front(cb_vol2col_rm, current_patch_rows);
                            patch_rows_left -= current_patch_rows;
                        }
                        tilize_uninit(cb_vol2col_rm, cb_vol2col_tiled);

                        // Apply matmul blocks
                        cb_wait_front(cb_vol2col_tiled, patch_tiles);
                        matmul_blocks(
                            cb_vol2col_tiled,
                            cb_weight_tiled,
                            cb_matmul_interm_tiled,
                            matmul_M_t,
                            matmul_N_t,
                            matmul_K_t,
                            in0_num_subblocks,
                            in1_num_subblocks,
                            in0_block_w,
                            subblock_h,
                            subblock_w,
                            false /* transpose */);
                        cb_pop_front(cb_vol2col_tiled, patch_tiles);

                        // Stall on matmul/bias to finish
                        cb_wait_front(cb_matmul_interm_tiled, output_tiles);

                        if (!is_reducer) {
                            // not reducer implies that we are a worker and there are multiple workers in this reduction
                            // group

                            // Signal to writer that we have partial results
                            cb_reserve_back(cb_reduction_tiled, output_tiles);
                            cb_push_back(cb_reduction_tiled, output_tiles);

                            // Wait for writer to ack that our data has been used
                            cb_wait_front(cb_worker_ack_back, 1);
                            cb_pop_front(cb_worker_ack_back, 1);

                            // Clear our partial results and continue
                            cb_pop_front(cb_matmul_interm_tiled, output_tiles);
                        } else {
                            // We are a reducer core. Note that num_workers can be 0, in which case there is no
                            // reduction.
                            for (uint32_t i = 0; i < num_workers; i++) {
                                // Wait for writer to populate reduction buffer
                                cb_wait_front(cb_reduction_tiled, output_tiles);

                                // Add partial results from workers and pop them
                                add_block_inplace<output_tiles>(cb_matmul_interm_tiled, cb_reduction_tiled);

                                // By freeing the reduction buffer, we signal to the writer that we have used the
                                // partial results. This is done inside add_block_inplace.
                            }

                            // Apply bias only if we are a reducer, and do it after reduction
                            if constexpr (use_bias) {
                                add_bias_inplace<matmul_M_t, matmul_N_t>(cb_matmul_interm_tiled, cb_bias_tiled);
                            }

                            // After reduction (if any), untilize result
                            cb_wait_front(cb_matmul_interm_tiled, output_tiles);
                            untilize_init_short(cb_matmul_interm_tiled);
                            for (uint32_t patch_t = 0; patch_t < matmul_M_t; patch_t++) {
                                cb_reserve_back(cb_matmul_result_rm, matmul_N_t);
                                untilize_block(cb_matmul_interm_tiled, matmul_N_t, cb_matmul_result_rm);
                                cb_push_back(cb_matmul_result_rm, matmul_N_t);
                                cb_pop_front(cb_matmul_interm_tiled, matmul_N_t);
                            }
                            untilize_uninit(cb_matmul_interm_tiled);
                        }
                    }
                }
            }
            // Free space for next block of weights
            cb_pop_front(cb_weight_tiled, weight_tiles);
            if constexpr (use_bias) {
                if (is_reducer) {
                    cb_pop_front(cb_bias_tiled, matmul_N_t);
                }
            }
        }
    }
}
}  // namespace NAMESPACE
