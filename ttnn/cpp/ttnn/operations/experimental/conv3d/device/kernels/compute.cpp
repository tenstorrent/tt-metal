// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include <tt-metalium/constants.hpp>

#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

template <uint32_t rows, uint32_t cols>
void add_bias_inplace(uint32_t inout_cb, uint32_t bias_cb) {
    // Math-side broadcast add (`add_tiles_bcast_rows`): used when inout_cb is not
    // physically full, so we cannot rely on pop+reserve returning the same L1 slots
    // and the pack-side L1-acc path in `add_inplace_l1_acc` is unsafe.

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t max_dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;

    add_bcast_rows_init_short(inout_cb, bias_cb);
    cb_wait_front(inout_cb, num_tiles);
    cb_wait_front(bias_cb, cols);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t col_start = 0; col_start < cols; col_start += max_dst_tiles) {
            const uint32_t cols_cur = (cols - col_start) < max_dst_tiles ? (cols - col_start) : max_dst_tiles;

            tile_regs_acquire();
            for (uint32_t j = 0; j < cols_cur; ++j) {
                add_tiles_bcast_rows(inout_cb, bias_cb, j, col_start + j, j);
            }
            tile_regs_commit();
            cb_pop_front(inout_cb, cols_cur);
            cb_reserve_back(inout_cb, cols_cur);
            tile_regs_wait();
            pack_tile_block(0, inout_cb, cols_cur);
            cb_push_back(inout_cb, cols_cur);
            tile_regs_release();
        }
    }
}

template <uint32_t rows, uint32_t cols, bool consume_add_cb>
void add_inplace_l1_acc(uint32_t inout_cb, uint32_t add_cb) {
    // Pack-side L1 accumulation (`pack_reconfig_l1_acc(1)` + indexed pack): cheaper
    // than the math add in `add_bias_inplace` because the add fuses into the pack,
    // but requires inout_cb to be physically full — pop+reserve must return the
    // same L1 slots so the indexed pack lands on top of the existing tiles.
    // consume_add_cb=false: add_cb is a single bias row reused for every output row.
    // consume_add_cb=true:  add_cb is a full block consumed tile-for-tile (reduction).
    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t add_tiles = consume_add_cb ? num_tiles : cols;
    constexpr uint32_t max_dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;
    static_assert(rows > 0 && cols > 0);

    cb_wait_front(inout_cb, num_tiles);
    cb_wait_front(add_cb, add_tiles);

    copy_tile_to_dst_init_short_with_dt(inout_cb, add_cb);
    pack_reconfig_data_format(inout_cb);
    pack_reconfig_l1_acc(1);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t col_start = 0; col_start < cols; col_start += max_dst_tiles) {
            const uint32_t cols_cur = (cols - col_start) < max_dst_tiles ? (cols - col_start) : max_dst_tiles;
            const uint32_t add_offset = consume_add_cb ? 0 : col_start;

            tile_regs_acquire();
            for (uint32_t j = 0; j < cols_cur; ++j) {
                copy_tile(add_cb, add_offset + j, j);
            }
            tile_regs_commit();
            cb_pop_front(inout_cb, cols_cur);
            if constexpr (consume_add_cb) {
                cb_pop_front(add_cb, cols_cur);
            }
            cb_reserve_back(inout_cb, cols_cur);
            tile_regs_wait();
            for (uint32_t j = 0; j < cols_cur; ++j) {
#if defined(ARCH_WORMHOLE)
                // tt-metal #44077: WH pack_tile reprograms the packer L1 destination.
                // Without this stall, the next pack_tile can rewrite the destination
                // address while the previous pack is still in flight, corrupting the
                // L1_ACC into a different tile. Restored from PR #44079 (originally
                // a kernel-local pack_tile_with_wh_destination_wait wrapper, lost
                // during the matmul-helper migration).
                if (j != 0) {
                    PACK(TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK));
                }
#endif
                pack_tile<true>(j, inout_cb, j);
            }
            cb_push_back(inout_cb, cols_cur);
            tile_regs_release();
        }
    }
    pack_reconfig_l1_acc(0);
}

template <uint32_t rows, uint32_t cols, bool use_fp32_partials, bool use_bias, uint32_t inout_cb, uint32_t bias_cb>
void add_bias_inplace_l1_acc_if_needed() {
    if constexpr (use_bias) {
        if constexpr (use_fp32_partials) {
            reconfig_data_format(inout_cb, bias_cb);
        }
        add_inplace_l1_acc<rows, cols, false>(inout_cb, bias_cb);
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
void bias_untilize_fullblock() {
    cb_wait_front(inout_cb, rows * cols);
    if constexpr (rows == 1 && cols == 1) {
        if constexpr (use_bias) {
            if constexpr (use_fp32_partials) {
                reconfig_data_format(inout_cb, bias_cb);
            }
            add_bias_inplace<rows, cols>(inout_cb, bias_cb);
        }
    } else {
        add_bias_inplace_l1_acc_if_needed<rows, cols, use_fp32_partials, use_bias, inout_cb, bias_cb>();
    }
    untilize_block<rows, cols, use_fp32_partials, inout_cb, out_cb>();
}

template <uint32_t rows, uint32_t cols, bool use_fp32_partials, uint32_t local_cb, uint32_t remote_cb>
void reduce_fullblock_inplace(uint32_t num_workers) {
    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t max_dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;

    cb_wait_front(local_cb, num_tiles);
    if (num_workers == 0) {
        return;  // no worker partials to reduce (C_in_num_blocks == 1)
    }

    // Pack-side L1 accumulation, but with the block-invariant reconfigs HOISTED out of the
    // per-worker loop. The prior add_inplace_l1_acc<> re-issued copy_tile_to_dst_init +
    // pack_reconfig_data_format + pack_reconfig_l1_acc(1)/(0) on EVERY worker; that reconfig
    // churn scaled with num_workers and dominated the frequent small-tile reductions of the
    // skinny-Cout conv3d upsamplers (Cin1024->Cout32-block, C_in_num_blocks=16, num_workers~15)
    // — measured +2.5% on Blackhole. The datacopy srcA init, packer format, and L1-acc mode
    // are identical for every worker (nothing between workers dirties them), so issue them
    // ONCE here and keep L1-acc enabled across all workers (each worker accumulates onto the
    // running sum in local_cb). This keeps L1-acc's fused-pack benefit while removing the churn.
    if constexpr (use_fp32_partials) {
        reconfig_data_format(local_cb, remote_cb);
    }
    copy_tile_to_dst_init_short_with_dt(local_cb, remote_cb);
    pack_reconfig_data_format(local_cb);
    pack_reconfig_l1_acc(1);
    for (uint32_t w = 0; w < num_workers; w++) {
        cb_wait_front(remote_cb, num_tiles);
        // local_cb is physically full: pop_front + reserve_back returns the same L1 slots so
        // the indexed L1-acc pack lands on top of the existing partials. Chunk by DST capacity.
        for (uint32_t i = 0; i < num_tiles; i += max_dst_tiles) {
            const uint32_t tiles_cur = (num_tiles - i) < max_dst_tiles ? (num_tiles - i) : max_dst_tiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < tiles_cur; ++j) {
                copy_tile(remote_cb, j, j);
            }
            tile_regs_commit();
            cb_pop_front(local_cb, tiles_cur);
            cb_pop_front(remote_cb, tiles_cur);
            cb_reserve_back(local_cb, tiles_cur);
            tile_regs_wait();
            for (uint32_t j = 0; j < tiles_cur; ++j) {
#if defined(ARCH_WORMHOLE)
                // tt-metal #44077: WH pack_tile reprograms the packer L1 destination; stall so
                // the next pack can't rewrite the address while the previous pack is in flight.
                if (j != 0) {
                    PACK(TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK));
                }
#endif
                pack_tile<true>(j, local_cb, j);
            }
            cb_push_back(local_cb, tiles_cur);
            tile_regs_release();
        }
    }
    pack_reconfig_l1_acc(0);
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
    reduce_fullblock_inplace<rows, cols, use_fp32_partials, local_cb, remote_cb>(num_workers);
    bias_untilize_fullblock<rows, cols, use_fp32_partials, use_bias, local_cb, bias_cb, out_cb>();
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
    constexpr uint32_t subblock_tiles = subblock_h * matmul_N_t;

    // CircularBuffer wrappers for compute_kernel_lib helpers.
    CircularBuffer cb_vol2col_tiled_buf(cb_vol2col_tiled);
    CircularBuffer cb_weight_tiled_buf(cb_weight_tiled);
    CircularBuffer cb_matmul_interm_tiled_buf(cb_matmul_interm_tiled);

    mm_init(cb_vol2col_tiled, cb_weight_tiled, cb_matmul_interm_tiled);
    // Configure Blackhole DEST swizzle_32b remap once at boot (no-op on Wormhole). The
    // tilize/untilize helpers below run with RemapMode::AssumeConfigured, so they rely on
    // this one-time configuration instead of reprogramming the remap per call.
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

                                    // Phase 2: matmul the batch.
                                    // Helper waits in0/in1 internally (mirrors the deferred weight
                                    // wait — wait_front lands inside the helper, after the tilize
                                    // above, preserving tilize/DRAM-read overlap).
                                    // in1_policy=WaitAndRetainOnLastBlock: weights stay across all
                                    // matmul_M_t/subblock_h invocations within this output block
                                    // (popped at the c_out_block level, see end of c_out_block loop).
                                    // InitMode::Short, reconfig gated on use_fp32_partials:
                                    //   - kernel's boot mm_init at the top of kernel_main owns
                                    //     hw_configure (the only place hw_configure is safe);
                                    //   - the helper's per-call Short does reconfig_data_format
                                    //     (in1, in0) + matmul_block_init, restoring matmul-mode
                                    //     unpack/math state after the tilize above (always needed).
                                    //   - the pack (output) reconfig is needed ONLY when partials
                                    //     are fp32: then interm is fp32 while the preceding tilize
                                    //     left the packer at the bf16 vol2col format, so it must be
                                    //     reprogrammed (reconfig=InputAndOutput). For bf16 partials
                                    //     interm == vol2col format == out, so the packer is already
                                    //     correct and pack_reconfig_data_format is pure per-call
                                    //     overhead — main's hand-written matmul_blocks gated it the
                                    //     same way (pack reconfig only `if constexpr (use_fp32_partials)`).
                                    //     Since this call runs once per (Cin-block × Cout-block ×
                                    //     spatial-block × M_t/subblock_h), that redundant reconfig
                                    //     scaled with the reduction-block count and drove the conv3d
                                    //     regressions on high-Cin-block shapes; reconfig=Input drops it.
                                    // interm_buf = out_buf because num_k_blocks==1: interm is unused.
                                    constexpr auto mm_reconfig =
                                        use_fp32_partials
                                            ? compute_kernel_lib::matmul_config::DataFormatReconfig::InputAndOutput
                                            : compute_kernel_lib::matmul_config::DataFormatReconfig::Input;
                                    compute_kernel_lib::matmul_block<
                                        /*transpose=*/false,
                                        /*packer_l1_acc=*/false,
                                        compute_kernel_lib::LastBlockTarget::Out,
                                        compute_kernel_lib::OutputCBLayout::SubblockMajor,
                                        compute_kernel_lib::matmul_config::InitMode::Short,
                                        compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
                                        compute_kernel_lib::InputPolicy::WaitAndRetainOnLastBlock,
                                        compute_kernel_lib::NoPostCompute,
                                        compute_kernel_lib::NoPreKBlock,
                                        compute_kernel_lib::NoPostKBlock,
                                        compute_kernel_lib::NoKBlockInnerDimFn,
                                        compute_kernel_lib::NoIn0Source,
                                        compute_kernel_lib::NoIn1BaseOffset,
                                        compute_kernel_lib::NoneActivation,
                                        mm_reconfig>(
                                        cb_vol2col_tiled_buf,
                                        cb_weight_tiled_buf,
                                        cb_matmul_interm_tiled_buf,
                                        cb_matmul_interm_tiled_buf,
                                        compute_kernel_lib::MatmulBlockShape::of(
                                            in0_num_subblocks,
                                            in1_num_subblocks,
                                            subblock_h,
                                            subblock_w,
                                            in0_block_w,
                                            /*num_k_blocks=*/1));

                                    if constexpr (enable_streaming_output) {
                                        // Streaming emits subblocks before cb_matmul_interm_tiled is physically full,
                                        // so bias uses math add and untilizes immediately.  The full-block path below
                                        // waits for the whole block and can use L1 pack accumulation instead.
                                        cb_wait_front(cb_matmul_interm_tiled, subblock_tiles);

                                        if constexpr (use_bias) {
                                            if constexpr (use_fp32_partials) {
                                                reconfig_data_format(cb_matmul_interm_tiled, cb_bias_tiled);
                                            }
                                            add_bias_inplace<subblock_h, matmul_N_t>(
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
