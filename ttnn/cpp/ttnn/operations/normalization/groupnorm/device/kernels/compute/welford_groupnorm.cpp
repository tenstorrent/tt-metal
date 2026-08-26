// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/welford.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"

// The optional fused activation is applied to the final output tile in DEST, right before it is
// packed into dfb_out (see "Write out the final output" below). The UNTILIZE_OUT branch routes the
// output through a different CB chain (dfb_untilize_in_id), so that insertion point would not be
// the last touch of the data — fail the build rather than fuse into the wrong place.
#if defined(UNTILIZE_OUT) && defined(SFPU_OP_INIT_ACTIVATION)
#error "fused activation + UNTILIZE_OUT not wired: untilize consumes dfb_untilize_in, not dfb_out"
#endif

void kernel_main() {
    /*
     * Definitions
     * block_h: This the length of the row we wish to processes in terms of tiles
     *
     * out_block_...: This is the length of our Circular Buffer, sometimes the length of out
     *                tensors(block_h) are larger than L1 space, so we have to process chunks of
     *                this data at a time. This chunk is called an out_block.
     *
     * num_out_blocks: This is the number of chunks specified by the use, such that a CBs
     *                (length defined by out_block) fit in L1.
     *                Users should minimize the number of num_out_blocks for better perf.
     *
     * ...normal:  If num_out_blocks evenly divides block_h, then all chunks are the size normal.
     * ...last: If num_out_blocks does not divides block_h, the leftovers are put into a chunk of
     *          length last.
     *
     * This is a high level description of the stages of this kernel, tags will be added to show
     * where in the code each stage starts and ends.
     *
     * Two-pass Statistics for Group Normalization:
     * The first pass computes each group's mean and the second accumulates centered squared
     * residuals. Both passes use FP32 SFPU accumulation.
     *
     * Batch Loop:
     *   Input tiles:
     *       For each group, accumulate its mean and centered M2 across all local channels
     *       Intermediate results are stored in dst registers between tiles
     *   Statistics Aggregation:
     *       Local aggregation per core:
     *           Convert accumulated centered M2 to variance
     *           Store per-group statistics in cb_ex_partial_id for inter-core communication
     *       Global reduction across cores:
     *           Reader kernels aggregate local statistics from all cores into cb_ex_global_id
     *           Designated sender core produces final global mean and variance per group
     *   Normalization Factor Calculation:
     *       Add epsilon to variance: Var + eps
     *       Compute reciprocal square root: 1/sqrt(Var + eps)
     *       Store normalization factor in cb_ex2pe_id for use in final calculation
     *   Final Normalization:
     *       Center inputs: x - μ (mean subtraction)
     *       Apply input mask to handle padding/ignored elements
     *       Scale by normalization factor: (x - μ) * (1/sqrt(Var + eps))
     *       Optional affine transform:
     *         Gamma scaling: result * γ
     *         Beta shift: result + β
     *
     * To find code sections, search for "Start LABEL" or "End LABEL" comments
     * Examples: "Start Statistics Aggregation" or "End Statistics Aggregation"
     */

    constexpr uint32_t do_gamma = get_named_compile_time_arg_val("do_gamma");
    constexpr uint32_t do_beta = get_named_compile_time_arg_val("do_beta");
    constexpr uint32_t num_cores_per_mcast_group = get_named_compile_time_arg_val("num_cores_per_mcast_group");

    constexpr uint32_t num_batches = get_named_compile_time_arg_val("batch");
    constexpr uint32_t num_groups = get_named_compile_time_arg_val("group");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");

    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    constexpr uint32_t per_core_MN = get_named_compile_time_arg_val("per_core_MN");

    constexpr uint32_t single_tile_size_bytes = get_named_compile_time_arg_val("single_tile_size_bytes");
    constexpr uint32_t num_tiles_input_mask = get_named_compile_time_arg_val("num_tiles_input_mask");
    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");

    // These are numbers in absolute terms, on a per group, per batch without tiling
    constexpr uint32_t num_channels_per_group = get_named_compile_time_arg_val("num_channels_per_group");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // input cbs
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_in_id = tt::CBIndex::c_29;
    // Welford-fp32 alias for dfb_in0 (non-TILIZE_IN path). Shares L1 memory with dfb_in0 but has
    // its own buffer index configured with unpack_to_dest_mode=UnpackToDestFp32
    // dfb_in0 is in Default mode so the final-stage sub_tiles_bcast_scalar (FPU on SrcA) keeps working.
    constexpr uint32_t dfb_in0_welford_id = get_named_compile_time_arg_val("cb_in0_welford");
    // Boolean indicating whether the welford kernel uses the alias CB.
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    // True when the statistics intake CB is configured with UnpackToDestFp32.
    constexpr bool welford_unpack_fp32_active = get_named_compile_time_arg_val("welford_unpack_fp32_active") != 0;
    // True when a reconfig-relevant operand is fp32: the per-tile reconfig_data_format calls below
    // are then required. All-bf16 compiles them out (no-ops). See program factory.
    constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;
    constexpr bool sfpu_two_pass_l1_replay = get_named_compile_time_arg_val("sfpu_two_pass_l1_replay") != 0;
    constexpr uint32_t sfpu_two_pass_reciprocal = get_named_compile_time_arg_val("sfpu_two_pass_reciprocal");
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_28;

    // interim cbs
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t dfb_x_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_xmm_id = tt::CBIndex::c_25;
    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_27;

    // interim cbs reuse
    constexpr uint32_t dfb_reread_write_out_id = tt::CBIndex::c_22;

    // output cb
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t dfb_out_id = (do_gamma or do_beta) ? dfb_out0_id : dfb_reread_write_out_id;
#endif

#ifdef UNTILIZE_OUT
    constexpr int dfb_outgamma_id = dfb_in_id;
    constexpr int dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_reread_write_out_id;
    constexpr int dfb_outbeta_id = do_gamma ? dfb_out_id : dfb_in_id;
    constexpr int dfb_untilize_in_id = (do_gamma and not do_beta) ? dfb_outgamma_id
                                       : do_beta                  ? dfb_outbeta_id
                                                                  : dfb_reread_write_out_id;
    constexpr int dfb_untilize_out_id =
#ifdef READER_REPACK
        dfb_repack_out_id;
#else
        dfb_out0_id;
#endif
#else
    constexpr int dfb_outgamma_id = do_beta ? dfb_in_id : dfb_out0_id;
    constexpr int dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_reread_write_out_id;
    constexpr int dfb_outbeta_id = dfb_out0_id;
#endif

    DataflowBuffer dfb_beta(dfb_beta_id);
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_gamma(dfb_gamma_id);
    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_in0_welford(dfb_in0_welford_id);
    DataflowBuffer dfb_input_mask(dfb_input_mask_id);
    DataflowBuffer dfb_out(dfb_out_id);
    DataflowBuffer dfb_x(dfb_x_id);
    DataflowBuffer dfb_xmm(dfb_xmm_id);

// tilize input from RM to tile layout
#ifdef TILIZE_IN
    compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in_id);
// Tilize in0 -> in (row-major to tiled)
#ifdef READER_REPACK
    constexpr uint32_t dfb_in_rm_id = dfb_repack_id;
    compute_kernel_lib::tilize<
        per_core_N,
        dfb_in_rm_id,
        dfb_in_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#else
    constexpr uint32_t dfb_in_rm_id = dfb_in0_id;
    compute_kernel_lib::tilize<
        per_core_N,
        dfb_in_rm_id,
        dfb_in_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::NoWait,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
    dfb_in.wait_front(per_core_MN);
#else
    compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in0_id);
#endif

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        num_out_blocks_padded++;
        out_block_h_last = (block_h % num_out_blocks);
    }

    dfb_eps.wait_front(1);
    dfb_input_mask.wait_front(num_tiles_input_mask);

    if constexpr (do_gamma) {
        dfb_gamma.wait_front(per_core_N);
    }
    if constexpr (do_beta) {
        dfb_beta.wait_front(per_core_N);
    }

    for (uint32_t b = 0; b < num_batches; ++b) {
        // Aim the unpacker at the welford intake CB; the tail stage leaves SrcA on dfb_x, so this
        // repeats every batch. Only the fp32 path needs the reconfig, and it must precede the init:
        // transpose_init's LLK assert checks the unpack config registers against the operand it is given.
        if constexpr (welford_unpack_fp32_active) {
#ifdef TILIZE_IN
            reconfig_data_format_srca(dfb_in_id);
#else
            reconfig_data_format_srca(dfb_in0_welford_id);
#endif
        }
#ifdef TILIZE_IN
        transpose_init(dfb_in_id);
#else
        transpose_init(dfb_in0_welford_id);
#endif
        dfb_ex_partial.reserve_back(2);
        tile_regs_acquire();
        two_pass_stats_init_shifted();
        two_pass_stats_clear();
        uint32_t active_group = 0;

        for (uint32_t g = 1; g < num_groups; ++g) {
            welford_save_state(mean_dst, g);
        }

        // First statistics traversal: accumulate sums relative to each group's first value.
        for (uint32_t mt = 0; mt < block_h; ++mt) {
            if (mt > 0) {
                welford_save_state(mean_dst, active_group);
                active_group = 0;
                welford_restore_state(mean_dst, active_group);
            }

            uint32_t min_group = 0;
            uint32_t channels_left = num_channels_per_group;

            for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                const uint32_t stats_input_tile = mt * per_core_N + nt;
                dfb_in0.wait_front(sfpu_two_pass_l1_replay ? stats_input_tile + 1 : 1);
                if constexpr (welford_fp32_alias) {
                    dfb_in0_welford.wait_front(sfpu_two_pass_l1_replay ? stats_input_tile + 1 : 1);
                }
#ifdef TILIZE_IN
                transpose_init(dfb_in_id);
                transpose_tile(dfb_in_id, 0, input_dst);
#else
                transpose_init(dfb_in0_welford_id);
                transpose_tile(dfb_in0_welford_id, sfpu_two_pass_l1_replay ? stats_input_tile : 0, input_dst);
#endif

                uint32_t group_offset = 0;
                for (uint32_t g = min_group; g < num_groups; ++g) {
                    const uint32_t cols_available = tile_width - group_offset;
                    const uint32_t cols_consumed = std::min(cols_available, channels_left);

                    if (mt == 0 && channels_left == num_channels_per_group) {
                        two_pass_stats_update_shifted_rows<false, true, num_groups == 1>(
                            input_dst, group_offset, cols_consumed);
                    } else {
                        two_pass_stats_update_shifted_rows<false, false, num_groups == 1>(
                            input_dst, group_offset, cols_consumed);
                    }

                    channels_left -= cols_consumed;
                    group_offset += cols_consumed;
                    if (channels_left > 0) {
                        break;
                    }

                    ++min_group;
                    if (min_group < num_groups) {
                        welford_save_state(mean_dst, active_group);
                        active_group = min_group;
                        welford_restore_state(mean_dst, active_group);
                    }
                    channels_left = num_channels_per_group;
                    if (group_offset == tile_width) {
                        break;
                    }
                }
                if constexpr (!sfpu_two_pass_l1_replay) {
                    dfb_in0.pop_front(1);
                    if constexpr (welford_fp32_alias) {
                        dfb_in0_welford.pop_front(1);
                    }
                }
            }
        }

        // Convert each group's shifted sum into its mean, leaving group zero
        // resident for the centered-M2 traversal.
        two_pass_stats_finish_shifted_mean<num_groups == 1>(sfpu_two_pass_reciprocal);
        if constexpr (num_groups > 1) {
            welford_save_state(mean_dst, active_group);
        }
        for (uint32_t g = 1; g + 1 < num_groups; ++g) {
            welford_restore_state(mean_dst, g);
            two_pass_stats_finish_shifted_mean<num_groups == 1>(sfpu_two_pass_reciprocal);
            welford_save_state(mean_dst, g);
        }
        if constexpr (num_groups > 1) {
            welford_restore_state(mean_dst, 0);
            two_pass_stats_finish_shifted_mean<num_groups == 1>(sfpu_two_pass_reciprocal);
        }

        // The reader streams the same local input tiles a second time.
        active_group = 0;
        for (uint32_t mt = 0; mt < block_h; ++mt) {
            if (mt > 0) {
                welford_save_state(mean_dst, active_group);
                active_group = 0;
                welford_restore_state(mean_dst, active_group);
            }
            uint32_t min_group = 0;
            uint32_t channels_left = num_channels_per_group;
            for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                const uint32_t stats_input_tile = mt * per_core_N + nt;
                dfb_in0.wait_front(sfpu_two_pass_l1_replay ? stats_input_tile + 1 : 1);
                if constexpr (welford_fp32_alias) {
                    dfb_in0_welford.wait_front(sfpu_two_pass_l1_replay ? stats_input_tile + 1 : 1);
                }
#ifdef TILIZE_IN
                transpose_init(dfb_in_id);
                transpose_tile(dfb_in_id, 0, input_dst);
#else
                transpose_init(dfb_in0_welford_id);
                transpose_tile(dfb_in0_welford_id, sfpu_two_pass_l1_replay ? stats_input_tile : 0, input_dst);
#endif
                uint32_t group_offset = 0;
                for (uint32_t g = min_group; g < num_groups; ++g) {
                    const uint32_t cols_available = tile_width - group_offset;
                    const uint32_t cols_consumed = std::min(cols_available, channels_left);
                    two_pass_stats_update_rows<true, num_groups == 1>(input_dst, group_offset, cols_consumed);
                    channels_left -= cols_consumed;
                    group_offset += cols_consumed;
                    if (channels_left > 0) {
                        break;
                    }
                    ++min_group;
                    if (min_group < num_groups) {
                        welford_save_state(mean_dst, active_group);
                        active_group = min_group;
                        welford_restore_state(mean_dst, active_group);
                    }
                    channels_left = num_channels_per_group;
                    if (group_offset == tile_width) {
                        break;
                    }
                }
                if constexpr (!sfpu_two_pass_l1_replay) {
                    dfb_in0.pop_front(1);
                    if constexpr (welford_fp32_alias) {
                        dfb_in0_welford.pop_front(1);
                    }
                }
            }
        }

#ifdef WELFORD_SFPU_LOCAL_COMBINE
        two_pass_stats_finalize_and_combine_to_face<num_groups == 1>(mean_dst, active_group, sfpu_two_pass_reciprocal);
#else
        two_pass_stats_finalize_to_face<num_groups == 1>(mean_dst, active_group, sfpu_two_pass_reciprocal);
#endif
        for (uint32_t g = 0; g + 1 < num_groups; ++g) {
            welford_restore_state(mean_dst, g);
#ifdef WELFORD_SFPU_LOCAL_COMBINE
            two_pass_stats_finalize_and_combine_to_face<false>(mean_dst, g, sfpu_two_pass_reciprocal);
#else
            two_pass_stats_finalize_to_face<false>(mean_dst, g, sfpu_two_pass_reciprocal);
#endif
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_block(mean_dst, dfb_ex_partial_id, 2);
        tile_regs_release();
        dfb_ex_partial.push_back(2);
        // End Statistics Aggregation

        // Start Normalization Factor Calculation
        // Wait for final welford values in cb_ex_global_id
        dfb_ex_global.wait_front(2 * num_groups);
        dfb_ex2pe.reserve_back(num_groups);
        // (Var + eps)
        // fp32: dfb_ex_global is fp32 (var), dfb_eps is bf16; the welford intake left SrcA on the fp32 input alias.
        // Reset both srcs so they match the operands read below. no-op for bf16.
        if constexpr (enable_fp32_reconfig) {
            reconfig_data_format_srca(dfb_ex_global_id);
        }
        reconfig_data_format_srcb(dfb_eps_id);
        add_init(dfb_ex_global_id, dfb_eps_id);
        for (uint32_t g = 0; g < num_groups; ++g) {
            tile_regs_acquire();
            add_tiles(dfb_ex_global_id, dfb_eps_id, 1 + (g << 1), 0, dst0);

            // 1/[sqrt(Var + eps)]
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_ex2pe_id);
            tile_regs_release();
        }
        dfb_ex2pe.push_back(num_groups);
        // End Normalization Factor Calculation

        dfb_ex2pe.wait_front(num_groups);

        // Start Final Normalization
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual = out_block_h_normal;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
            }

            for (uint32_t mt = 0; mt < out_block_h_actual; ++mt) {
                // This indicates the smallest group that is yet to be processed for this block
                // As we iterate over nt, some of the groups will be completed, and we will update
                // this variable
                uint32_t min_group = 0;

                // This indicates the number of channels left to be processed for the min_group
                // As we iterate over nt, some of the channels will be completed, and we will
                // update this variable
                // It is mainly used when we move from one tile to the next, if there are channels
                // left to be processed for the min_group, we will process them in the next tile
                uint32_t channels_left = num_channels_per_group;

                // This tracks the correct index to use for the mask.
                // For each group, there are block_w number of mask tiles. As we iterate over nt,
                // we will update this variable to track the correct index to use for the mask.
                uint32_t block_w_index = 0;

                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    dfb_in0.wait_front(1);
                    if constexpr (welford_fp32_alias) {
                        // The reader pushes dfb_in0 and dfb_in0_welford in lockstep; wait on the
                        // alias so its rd_ptr advance (via pop_front below) is synchronized with
                        // the reader's wr_ptr advance on the alias.
                        dfb_in0_welford.wait_front(1);
                    }

                    uint32_t group_offset = 0;
                    for (uint32_t g = min_group; g < num_groups; ++g) {
                        dfb_xmm.reserve_back(1);

                        // // Now let us do the actual computation for the current group here
                        // // a. x-u
                        // fp32: SrcA needs dfb_in0 (fp32 input), SrcB needs dfb_ex_global (fp32 mean); the prior
                        // group's mul_tiles(dfb_xmm) left SrcA on dfb_xmm. Use the unconditional 1-arg form: the old
                        // 2-arg srcb(dfb_eps -> dfb_ex_global) never reset SrcA at all.
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_in0_id);
                            reconfig_data_format_srcb(dfb_ex_global_id);
                        } else {
                            reconfig_data_format_srcb(dfb_eps_id, dfb_ex_global_id);
                        }
                        sub_bcast_scalar_init(dfb_in0_id, dfb_ex_global_id);

                        tile_regs_acquire();
                        sub_tiles_bcast_scalar(dfb_in0_id, dfb_ex_global_id, 0, 0 + (g << 1), dst0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(dst0, dfb_xmm_id);
                        tile_regs_release();
                        dfb_xmm.push_back(1);

                        // // b. (x - u) * 1/[sqrt(Var + eps)]
                        dfb_xmm.wait_front(1);
                        // fp32: reset SrcA to dfb_xmm (fp32); stage a left SrcA on dfb_in0. The reconfig has to
                        // precede the init: the init's LLK assert checks that the unpack config registers already
                        // describe these operands.
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_in0_id, dfb_xmm_id);
                        }
                        reconfig_data_format_srcb(dfb_ex_global_id, dfb_ex2pe_id);
                        mul_bcast_scalar_init(dfb_xmm_id, dfb_ex2pe_id);
                        tile_regs_acquire();
                        mul_tiles_bcast_scalar(dfb_xmm_id, dfb_ex2pe_id, 0, g, dst0);
                        tile_regs_commit();
                        dfb_xmm.pop_front(1);
                        dfb_xmm.reserve_back(1);
                        tile_regs_wait();
                        pack_tile(dst0, dfb_xmm_id);
                        tile_regs_release();
                        dfb_xmm.push_back(1);

                        // // c. [(x - u) * rsqrt] * mask
                        const uint32_t mask_offset = g * block_w;
                        const uint32_t mask_index = mask_offset + block_w_index;

                        dfb_xmm.wait_front(1);
                        reconfig_data_format_srcb(dfb_ex2pe_id, dfb_input_mask_id);
                        mul_bcast_rows_init(dfb_xmm_id, dfb_input_mask_id);
                        tile_regs_acquire();
                        mul_tiles_bcast_rows(dfb_xmm_id, dfb_input_mask_id, 0, mask_index, dst0);
                        dfb_xmm.pop_front(1);

                        // // d. Accumulate into cb_x_id.
                        if (group_offset != 0) {
                            // Not the first group for this tile: add what is already in cb_x.
                            reconfig_data_format_srca(dfb_x_id);
                            binary_dest_reuse_tiles_init<
                                EltwiseBinaryType::ELWADD,
                                EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_x_id);
                            dfb_x.wait_front(1);
                            binary_dest_reuse_tiles<
                                EltwiseBinaryType::ELWADD,
                                EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb_x_id, 0, dst0);
                            dfb_x.pop_front(1);
                        }
                        tile_regs_commit();

                        // Then we pack the result into cb_x_id
                        dfb_x.reserve_back(1);
                        tile_regs_wait();
                        pack_tile(dst0, dfb_x_id);
                        tile_regs_release();
                        dfb_x.push_back(1);

                        // The blocks after this loop assume srcb still carries cb_xmm's format.
                        reconfig_data_format_srcb(dfb_xmm_id);

                        uint32_t cols_available = tile_width - group_offset;
                        uint32_t cols_consumed = std::min(cols_available, channels_left);
                        channels_left -= cols_consumed;
                        group_offset += cols_consumed;

                        // There are still channels left to be processed for the current group
                        // This can only be done in the next tile. So we don't do any more groups
                        // for this tile.
                        if (channels_left > 0) {
                            // For the next tile, we need to use the next mask index
                            ++block_w_index;
                            break;
                        }

                        // Since we know that channels_left is 0, it also means that we have
                        // processed all the channels for the current group.
                        // We update the min_group so we never revisit this group again.
                        ++min_group;
                        channels_left = num_channels_per_group;
                        block_w_index = 0;

                        // All available columns have been used for this tile, so we don't do any
                        // more groups for this tile.
                        if (group_offset == tile_width) {
                            break;
                        }
                    }
                    dfb_in0.pop_front(1);
                    if constexpr (welford_fp32_alias) {
                        dfb_in0_welford.pop_front(1);
                    }

                    if constexpr (do_gamma) {
                        // fp32: reset SrcA to dfb_x (fp32); the prior mask/accumulate step left SrcA on a bf16 format.
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_x_id);
                            reconfig_data_format_srcb(dfb_xmm_id, dfb_gamma_id);
                            mul_bcast_rows_init(dfb_x_id, dfb_gamma_id);
                        } else {
                            reconfig_data_format_srcb(dfb_xmm_id, dfb_gamma_id);
                            mul_bcast_rows_init(dfb_x_id, dfb_gamma_id);
                        }

                        dfb_x.wait_front(1);
                        tile_regs_acquire();
                        mul_tiles_bcast_rows(dfb_x_id, dfb_gamma_id, 0, nt, dst0);
                        tile_regs_commit();
                        dfb_x.pop_front(1);
                        dfb_x.reserve_back(1);
                        tile_regs_wait();
                        pack_tile(dst0, dfb_x_id);
                        tile_regs_release();
                        dfb_x.push_back(1);
                    }

                    if constexpr (do_beta) {
                        // fp32: reset SrcA to dfb_x (fp32), same as the gamma step above.
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_x_id);
                            reconfig_data_format_srcb(do_gamma ? dfb_gamma_id : dfb_xmm_id, dfb_beta_id);
                            add_bcast_rows_init(dfb_x_id, dfb_beta_id);
                        } else {
                            reconfig_data_format_srcb(do_gamma ? dfb_gamma_id : dfb_xmm_id, dfb_beta_id);
                            add_bcast_rows_init(dfb_x_id, dfb_beta_id);
                        }

                        dfb_x.wait_front(1);
                        tile_regs_acquire();
                        add_tiles_bcast_rows(dfb_x_id, dfb_beta_id, 0, nt, dst0);
                        tile_regs_commit();
                        dfb_x.pop_front(1);
                        dfb_x.reserve_back(1);
                        tile_regs_wait();
                        pack_tile(dst0, dfb_x_id);
                        tile_regs_release();
                        dfb_x.push_back(1);
                    }

                    // Write out the final output
                    if constexpr (enable_fp32_reconfig) {
                        reconfig_data_format_srca(dfb_x_id);
                    }
                    reconfig_data_format_srcb(do_beta ? dfb_beta_id : dfb_xmm_id, dfb_x_id);
                    copy_init(dfb_x_id);

                    dfb_x.wait_front(1);
                    tile_regs_acquire();
                    copy_tile(dfb_x_id, 0, dst0);
#ifdef SFPU_OP_INIT_ACTIVATION
                    // Optional fused unary activation (e.g. SiLU). Applied here, after gamma and
                    // beta and on the final output tile still in DEST, so the result equals
                    // <act>(group_norm(x)) with the activation consuming the fp32 DEST value rather
                    // than the rounded output a standalone activation op would read back.
                    // Safe w.r.t. welford's SFPU replay state: this is the POST stage, all welford
                    // accumulation for the batch is already finished.
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                    tile_regs_commit();
                    dfb_x.pop_front(1);
                    dfb_out.reserve_back(1);
                    tile_regs_wait();
#ifndef UNTILIZE_OUT
                    // Packer was last set for bf16 dfb_x; reconfigure to dfb_out_id (may be fp32) before pack, restore
                    // after. Only needed when out differs from dfb_x (fp32 path); no-op gated out for bf16.
                    if constexpr (enable_fp32_reconfig) {
                        pack_reconfig_data_format(dfb_out_id);
                    }
#endif
                    pack_tile(dst0, dfb_out_id);
#ifndef UNTILIZE_OUT
                    if constexpr (enable_fp32_reconfig) {
                        pack_reconfig_data_format(dfb_x_id);
                    }
#endif
                    tile_regs_release();
                    dfb_out.push_back(1);
                }
            }

#ifdef UNTILIZE_OUT
            // untilize - DEST capacity auto-detected
            compute_kernel_lib::untilize<
                per_core_N,
                dfb_untilize_in_id,
                dfb_untilize_out_id,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
        }
        // End Final Normalization

        dfb_ex_global.pop_front(2 * num_groups);
        dfb_ex2pe.pop_front(num_groups);
    }

    dfb_eps.pop_front(1);
    dfb_input_mask.pop_front(num_tiles_input_mask);

    // Pop all the cb_beta_id and cb_gamma_id if used
    if constexpr (do_beta) {
        dfb_beta.pop_front(per_core_N);
    }
    if constexpr (do_gamma) {
        dfb_gamma.pop_front(per_core_N);
    }
}
