// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include <tt-metalium/constants.hpp>

#include "api/compute/untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

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
    constexpr uint32_t cb_zero_tiled = get_compile_time_arg_val(28);

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

    // Process each batch element
    for (uint32_t batch_idx = 0; batch_idx < N; batch_idx++) {
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
                            // When using fp32 partials, switch packer to bf16 for tilize, then back to fp32 for
                            // matmul. Also reconfigure unpacker srcA from Float32 (left by previous untilize) to
                            // bf16 so the tilize correctly reads bf16 vol2col data.
                            if constexpr (use_fp32_partials) {
                                pack_reconfig_data_format(cb_vol2col_tiled);
                                reconfig_data_format_srca(cb_vol2col_rm);
                            }

                            // Tilize row-major patches
                            compute_kernel_lib::tilize<
                                matmul_K_t,
                                cb_vol2col_rm,
                                cb_vol2col_tiled,
                                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                                matmul_M_t, num_patches);

                            if constexpr (use_fp32_partials) {
                                // Reconfigure packer for fp32 output after tilize left it in bf16.
                                // mm_block_init_short_with_both_dt is not needed: the matmul_block
                                // helper calls mm_block_init() internally.
                                pack_reconfig_data_format(cb_matmul_interm_tiled);
                            }

                            // Apply matmul blocks using library helper.
                            // NoWaitNoPop: we manage input CBs externally — vol2col is
                            // waited/popped here, weights persist across spatial patches.
                            cb_wait_front(cb_vol2col_tiled, patch_tiles);
                            compute_kernel_lib::matmul_block<
                                cb_vol2col_tiled,
                                cb_weight_tiled,
                                cb_matmul_interm_tiled,
                                cb_matmul_interm_tiled,
                                compute_kernel_lib::matmul_block_config::InitUninitMode::InitAndUninit,
                                compute_kernel_lib::matmul_block_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
                                compute_kernel_lib::matmul_block_config::WaitPopMode::NoWaitNoPop>(
                                {.block_w = in0_block_w,
                                 .num_subblocks = in0_num_subblocks,
                                 .block_num_tiles = matmul_M_t * in0_block_w,
                                 .subblock_num_tiles = subblock_h * in0_block_w},
                                {.num_subblocks = in1_num_subblocks,
                                 .block_num_tiles = in0_block_w * matmul_N_t,
                                 .per_core_w = matmul_N_t},
                                1,  // num_blocks (single K-block, no spill/reload)
                                {.h = subblock_h, .w = subblock_w, .num_tiles = subblock_h * subblock_w});
                            cb_pop_front(cb_vol2col_tiled, patch_tiles);

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
                                // We are a reducer core. Note that num_workers can be 0, in which case there is no
                                // reduction.
                                if constexpr (use_fp32_partials) {
                                    cb_wait_front(cb_zero_tiled, 1);
                                    reconfig_data_format_srca(cb_matmul_interm_tiled);
                                    // pack_reconfig not needed — packer already fp32 from pre-matmul reconfig
                                }
                                for (uint32_t i = 0; i < num_workers; i++) {
                                    cb_wait_front(cb_reduction_tiled, output_tiles);

                                    if constexpr (use_fp32_partials) {
                                        for (uint32_t t = 0; t < output_tiles; t++) {
                                            tile_regs_acquire();
                                            // Re-init before each op: copy_tile and add_tiles
                                            // share the MATH unit config, so each needs its own
                                            // init per tile iteration.
                                            copy_tile_init(cb_matmul_interm_tiled);
                                            copy_tile(cb_matmul_interm_tiled, 0, 0);
                                            add_tiles_init(cb_reduction_tiled, cb_zero_tiled, true);
                                            add_tiles(cb_reduction_tiled, cb_zero_tiled, 0, 0, 0);
                                            tile_regs_commit();

                                            cb_pop_front(cb_matmul_interm_tiled, 1);
                                            cb_pop_front(cb_reduction_tiled, 1);
                                            cb_reserve_back(cb_matmul_interm_tiled, 1);
                                            tile_regs_wait();
                                            pack_tile(0, cb_matmul_interm_tiled);
                                            cb_push_back(cb_matmul_interm_tiled, 1);
                                            tile_regs_release();
                                        }
                                    } else {
                                        add_block_inplace<output_tiles>(cb_matmul_interm_tiled, cb_reduction_tiled);
                                    }
                                }

                                // Apply bias only if we are a reducer, and do it after reduction
                                if constexpr (use_bias) {
                                    if constexpr (use_fp32_partials) {
                                        reconfig_data_format(cb_matmul_interm_tiled, cb_bias_tiled);
                                    }
                                    add_bias_inplace<matmul_M_t, matmul_N_t>(cb_matmul_interm_tiled, cb_bias_tiled);
                                }

                                // Untilize result — reconfigure unpacker when fp32 partials need
                                // format conversion back to bf16
                                constexpr auto untilize_reconfig_mode =
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
                                    untilize_reconfig_mode>(matmul_M_t);
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
}
