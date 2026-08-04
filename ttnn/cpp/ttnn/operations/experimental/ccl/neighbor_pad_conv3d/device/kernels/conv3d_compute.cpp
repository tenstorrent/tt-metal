// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Conv3d compute kernel for the fused NP op.  This is INTENTIONALLY the legacy fork of upstream
// conv3d's compute (ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp), not a
// re-base.  It carries NO NP-specific control flow — the halo overlap lives entirely in the reader's
// phase loop + the NP gate; compute just consumes cb_vol2col in whatever order the reader produces it.
// It diverges from upstream conv3d compute on two axes (both required by the fused cross-core
// reduction path on these shapes):
//   1. Manual per-tile fp32 reduction (copy_tile + add_tiles loop, use_fp32_partials branch below)
//      instead of upstream's DEST-batched reduce_fullblock_inplace_math.  This is FASTER, not just
//      fp32-accurate: swapping in upstream's reduce_fullblock_inplace_math regresses s4_res halo_last
//      device-FW ~1.5-3% (measured; not L1-recoverable).  Do NOT replace with the upstream reduce.
//   2. NO streaming output.  Upstream streams single-tile C_out rows through bias/untilize to overlap
//      the writer; that path is gated on C_in_num_blocks==1 && C_out_block<=32, which ALL these
//      shapes fail (they split C_in for the cross-core reduce + use C_out_block 128/256).  The
//      reduction-gated full-block untilize is structurally incompatible with per-subblock streaming.
// Everything else (fused tilize+matmul subblock loop, matmul_blocks, add_bias_inplace) should track
// upstream conv3d's compute semantics; keep the matmul/tilize structure aligned when upstream changes.
#include "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/conv3d_compute_lib.hpp"

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

    constexpr uint32_t weight_tiles = matmul_K_t * matmul_N_t;
    constexpr uint32_t output_tiles = matmul_M_t * matmul_N_t;
    constexpr uint32_t batch_tiles = subblock_h * matmul_K_t;

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
                                                NoReconfigure>(1, patches_this_row);
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
                                }
                            }

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
                                // Untilize result
                                {
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
