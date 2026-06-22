// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include <algorithm>
#include <tt-metalium/constants.hpp>

#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

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
    // precondition: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    matmul_block_init(
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
            cb_push_back(out_cb, out_subblock_num_tiles);
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
}

ALWI void pack_tile_with_wh_destination_wait(uint32_t tile, uint32_t out_cb, uint32_t pack_sequence_idx) {
#if defined(ARCH_WORMHOLE)
    if (pack_sequence_idx != 0) {
        // Workaround for https://github.com/tenstorrent/tt-metal/issues/44077:
        // WH pack_tile reprograms the packer L1 destination. Wait before the
        // next tile rewrites that address while the previous pack is in flight.
        PACK(TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK));
    }
#endif
    pack_tile(tile, out_cb);
}

template <uint32_t rows, uint32_t cols, uint32_t add_dst_tiles>
void add_bias_inplace(uint32_t inout_cb, uint32_t bias_cb) {
    constexpr uint32_t num_tiles = rows * cols;

    add_bcast_rows_init_short(inout_cb, bias_cb);
    cb_wait_front(inout_cb, num_tiles);
    cb_wait_front(bias_cb, cols);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t col_start = 0; col_start < cols; col_start += add_dst_tiles) {
            const uint32_t cols_cur = std::min(add_dst_tiles, cols - col_start);
            tile_regs_acquire();
            for (uint32_t j = 0; j < cols_cur; ++j) {
                add_tiles_bcast_rows(inout_cb, bias_cb, j, col_start + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(inout_cb, cols_cur);
            cb_reserve_back(inout_cb, cols_cur);
            for (uint32_t j = 0; j < cols_cur; ++j) {
                pack_tile_with_wh_destination_wait(j, inout_cb, i * cols + col_start + j);
            }
            cb_push_back(inout_cb, cols_cur);
            tile_regs_release();
        }
    }
}

template <uint32_t num_tiles, uint32_t add_dst_tiles>
void add_block_inplace_math(uint32_t inout_cb, uint32_t add_cb) {
    add_tiles_init(inout_cb, add_cb);
    for (uint32_t i = 0; i < num_tiles; i += add_dst_tiles) {
        const uint32_t tiles_cur = std::min(add_dst_tiles, num_tiles - i);
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < tiles_cur; ++tile) {
            add_tiles(inout_cb, add_cb, tile, tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(inout_cb, tiles_cur);
        cb_pop_front(add_cb, tiles_cur);
        cb_reserve_back(inout_cb, tiles_cur);
        for (uint32_t tile = 0; tile < tiles_cur; ++tile) {
            pack_tile_with_wh_destination_wait(tile, inout_cb, i + tile);
        }
        cb_push_back(inout_cb, tiles_cur);
        tile_regs_release();
    }
}

template <uint32_t rows, uint32_t cols, bool use_fp32_partials, uint32_t in_cb, uint32_t out_cb>
void untilize_block() {
    constexpr auto untilize_reconfig_mode =
        use_fp32_partials ? compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure
                          : compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure;
    compute_kernel_lib::untilize<
        cols,
        in_cb,
        out_cb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        untilize_reconfig_mode,
        compute_kernel_lib::untilize_config::RemapMode::AssumeConfigured>(rows);
}

template <
    uint32_t rows,
    uint32_t cols,
    bool use_fp32_partials,
    bool use_bias,
    uint32_t inout_cb,
    uint32_t bias_cb,
    uint32_t out_cb>
void bias_untilize_fullblock_math() {
    cb_wait_front(inout_cb, rows * cols);
    if constexpr (use_bias) {
        if constexpr (use_fp32_partials) {
            reconfig_data_format(inout_cb, bias_cb);
        }
        add_bias_inplace<rows, cols, compute_kernel_lib::DEST_AUTO_LIMIT>(inout_cb, bias_cb);
    }
    untilize_block<rows, cols, use_fp32_partials, inout_cb, out_cb>();
}

template <uint32_t rows, uint32_t cols, bool use_fp32_partials, uint32_t local_cb, uint32_t remote_cb>
void reduce_fullblock_inplace_math(uint32_t num_workers) {
    constexpr uint32_t num_tiles = rows * cols;

    cb_wait_front(local_cb, num_tiles);

    if constexpr (use_fp32_partials) {
        reconfig_data_format(local_cb, remote_cb);
        pack_reconfig_data_format(local_cb);
    }
    for (uint32_t i = 0; i < num_workers; i++) {
        cb_wait_front(remote_cb, num_tiles);
        add_block_inplace_math<num_tiles, compute_kernel_lib::DEST_AUTO_LIMIT>(local_cb, remote_cb);
    }
}

template <
    uint32_t rows,
    uint32_t cols,
    bool use_fp32_partials,
    bool use_bias,
    uint32_t local_cb,
    uint32_t remote_cb,
    uint32_t bias_cb,
    uint32_t out_cb>
void reduce_bias_untilize_fullblock(uint32_t num_workers) {
    if (num_workers > 0) {
        reduce_fullblock_inplace_math<rows, cols, use_fp32_partials, local_cb, remote_cb>(num_workers);
    }
    bias_untilize_fullblock_math<rows, cols, use_fp32_partials, use_bias, local_cb, bias_cb, out_cb>();
}

void kernel_main() {
    constexpr uint32_t cb_vol2col_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_vol2col_tiled = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight_tiled = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bias_tiled = get_compile_time_arg_val(3);
    constexpr uint32_t cb_matmul_interm_tiled = get_compile_time_arg_val(4);
    constexpr uint32_t cb_matmul_result_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_reduction_tiled = get_compile_time_arg_val(6);
    constexpr uint32_t cb_worker_ack_back = get_compile_time_arg_val(7);
    constexpr uint32_t N = get_compile_time_arg_val(8);

    constexpr uint32_t num_patches = get_compile_time_arg_val(9);
    constexpr uint32_t matmul_M_t = get_compile_time_arg_val(10);
    constexpr uint32_t matmul_K_t = get_compile_time_arg_val(11);
    constexpr uint32_t matmul_N_t = get_compile_time_arg_val(12);

    constexpr bool use_bias = get_compile_time_arg_val(13) == 1;
    constexpr uint32_t T_out = get_compile_time_arg_val(14);
    constexpr uint32_t H_out = get_compile_time_arg_val(15);
    constexpr uint32_t W_out = get_compile_time_arg_val(16);
    constexpr uint32_t T_block_size = get_compile_time_arg_val(17);
    constexpr uint32_t H_block_size = get_compile_time_arg_val(18);
    constexpr uint32_t W_block_size = get_compile_time_arg_val(19);
    constexpr uint32_t C_out_num_blocks = get_compile_time_arg_val(20);

    // matmul parameters
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(21);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(22);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(23);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(25);

    constexpr uint32_t semaphore_id = get_compile_time_arg_val(26);
    constexpr bool use_fp32_partials = get_compile_time_arg_val(27) == 1;
    // Stream final single-tile C_out rows through bias/untilize when the writer can overlap the compute tail.
    constexpr bool enable_streaming_output = get_compile_time_arg_val(28) == 1;

    constexpr uint32_t weight_tiles = matmul_K_t * matmul_N_t;
    constexpr uint32_t output_tiles = matmul_M_t * matmul_N_t;
    constexpr uint32_t batch_tiles = subblock_h * matmul_K_t;
    constexpr uint32_t subblock_tiles = subblock_h * matmul_N_t;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_vol2col_tiled, cb_weight_tiled, cb_matmul_interm_tiled);
    matmul_init(cb_vol2col_tiled, cb_weight_tiled);
    MATH((llk_math_reconfig_remap(true)));

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

    // Process each batch element
    for (uint32_t batch_idx = 0; batch_idx < N; batch_idx++) {
        for (uint32_t c_in_block = c_in_block_start; c_in_block < c_in_block_end; c_in_block++) {
            // Process only assigned C_out blocks
            for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
                // Bias must be ready before the first spatial block's reduction.
                // Weight wait is deferred to right before matmul so the first
                // tilize overlaps with BRISC's DRAM weight read.
                if constexpr (use_bias) {
                    if (is_reducer) {
                        cb_wait_front(cb_bias_tiled, matmul_N_t);
                    }
                }

                // 3D blocking loops over assigned ranges:
                for (uint32_t t_block = t_out_start; t_block < t_out_end; t_block += T_block_size) {
                    for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                        for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                            // Fused tilize+matmul: tilize subblock_h rows, then
                            // matmul the batch. Repeat matmul_M_t/subblock_h times.
                            // Saves (M_t - subblock_h) * K_t tiles of L1 vs full M_t*K_t.
                            {
                                uint32_t patches_left = num_patches;
                                for (uint32_t m_start = 0; m_start < matmul_M_t; m_start += subblock_h) {
                                    // Phase 1: tilize subblock_h rows into cb_vol2col_tiled
                                    for (uint32_t m = 0; m < subblock_h; m++) {
                                        const uint32_t patches_this_row = (patches_left >= tt::constants::TILE_HEIGHT)
                                                                              ? tt::constants::TILE_HEIGHT
                                                                              : patches_left;
                                        if constexpr (use_fp32_partials) {
                                            pack_reconfig_data_format(cb_vol2col_tiled);
                                            reconfig_data_format_srca(cb_vol2col_rm);
                                        }
                                        compute_kernel_lib::tilize<
                                            matmul_K_t,
                                            cb_vol2col_rm,
                                            cb_vol2col_tiled,
                                            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                                            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                                            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::
                                                NoReconfigure,
                                            compute_kernel_lib::tilize_config::Fp32Mode::Fast,
                                            compute_kernel_lib::tilize_config::RemapMode::AssumeConfigured>(
                                            1, patches_this_row);
                                        patches_left -= patches_this_row;
                                    }

                                    if constexpr (use_fp32_partials) {
                                        pack_reconfig_data_format(cb_matmul_interm_tiled);
                                    }

                                    // Wait for weights — deferred so tilize overlaps with BRISC's DRAM read.
                                    cb_wait_front(cb_weight_tiled, weight_tiles);

                                    // Phase 2: matmul the batch
                                    cb_wait_front(cb_vol2col_tiled, batch_tiles);
                                    matmul_blocks(
                                        cb_vol2col_tiled,
                                        cb_weight_tiled,
                                        cb_matmul_interm_tiled,
                                        subblock_h,
                                        matmul_N_t,
                                        matmul_K_t,
                                        in0_num_subblocks,
                                        in1_num_subblocks,
                                        in0_block_w,
                                        subblock_h,
                                        subblock_w,
                                        false /* transpose */);
                                    cb_pop_front(cb_vol2col_tiled, batch_tiles);

                                    if constexpr (enable_streaming_output) {
                                        // Streaming emits subblocks before cb_matmul_interm_tiled is physically full,
                                        // so bias uses math add and untilizes immediately.
                                        cb_wait_front(cb_matmul_interm_tiled, subblock_tiles);

                                        if constexpr (use_bias) {
                                            if constexpr (use_fp32_partials) {
                                                reconfig_data_format(cb_matmul_interm_tiled, cb_bias_tiled);
                                            }
                                            add_bias_inplace<
                                                subblock_h,
                                                matmul_N_t,
                                                compute_kernel_lib::DEST_AUTO_LIMIT>(
                                                cb_matmul_interm_tiled, cb_bias_tiled);
                                        }

                                        constexpr auto untilize_reconfig_mode_sb =
                                            use_fp32_partials ? compute_kernel_lib::untilize_config::
                                                                    ReconfigureRegisterDatatypeMode::UnpackReconfigure
                                                              : compute_kernel_lib::untilize_config::
                                                                    ReconfigureRegisterDatatypeMode::NoReconfigure;
                                        compute_kernel_lib::untilize<
                                            matmul_N_t,
                                            cb_matmul_interm_tiled,
                                            cb_matmul_result_rm,
                                            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                                            compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
                                            untilize_reconfig_mode_sb,
                                            compute_kernel_lib::untilize_config::RemapMode::AssumeConfigured>(
                                            subblock_h);
                                    }
                                }
                            }

                            if constexpr (!enable_streaming_output) {
                                // Stall on matmul/bias to finish
                                cb_wait_front(cb_matmul_interm_tiled, output_tiles);

                                if (!is_reducer) {
                                    // not reducer implies that we are a worker and there are multiple workers in this
                                    // reduction group

                                    // Signal to writer that we have partial results
                                    cb_reserve_back(cb_reduction_tiled, output_tiles);
                                    cb_push_back(cb_reduction_tiled, output_tiles);

                                    // Wait for writer to ack that our data has been used
                                    cb_wait_front(cb_worker_ack_back, 1);
                                    cb_pop_front(cb_worker_ack_back, 1);

                                    // Clear our partial results and continue
                                    cb_pop_front(cb_matmul_interm_tiled, output_tiles);
                                } else {
                                    // We are a reducer core.
                                    reduce_bias_untilize_fullblock<
                                        matmul_M_t,
                                        matmul_N_t,
                                        use_fp32_partials,
                                        use_bias,
                                        cb_matmul_interm_tiled,
                                        cb_reduction_tiled,
                                        cb_bias_tiled,
                                        cb_matmul_result_rm>(num_workers);
                                }
                            }  // end if constexpr (!enable_streaming_output)
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
}
