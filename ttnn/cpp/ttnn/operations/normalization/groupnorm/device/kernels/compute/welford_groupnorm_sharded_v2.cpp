// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/welford.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);

    constexpr uint32_t num_batches = get_compile_time_arg_val(4);
    constexpr uint32_t num_groups = get_compile_time_arg_val(5);

    constexpr uint32_t block_h = get_compile_time_arg_val(7);
    constexpr uint32_t block_w = get_compile_time_arg_val(8);
    constexpr uint32_t block_hw = get_compile_time_arg_val(9);

    constexpr uint32_t per_core_M = get_compile_time_arg_val(12);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(13);
    constexpr uint32_t per_core_MN = get_compile_time_arg_val(14);

    constexpr uint32_t num_tiles_input_mask = get_compile_time_arg_val(19);
    constexpr uint32_t num_channels_per_group = get_compile_time_arg_val(24);
    constexpr uint32_t tile_width = get_compile_time_arg_val(25);

    // Welford-fp32 alias args. When the alias is active, dfb_in0_welford_id points
    // to c_29 (shares SRAM with c_0) and dfb_in_welford_id points to c_31 (shares SRAM with c_1).
    // Both alias indices are configured with unpack_to_dest_mode=UnpackToDestFp32 so
    // transpose_tile preserves FP32 precision for the SFPU Welford.
    // The final-stage sub_tiles_bcast_scalar reads c_0 / c_1 (Default SrcA path).
    //
    // Unlike the mcast / no_mcast groupnorm kernels, no separate
    // welford_unpack_fp32_active flag is needed here. Both the TILIZE_IN and
    // non-TILIZE_IN branches route the welford intake transpose through an alias
    // DFB (dfb_in_welford_id or dfb_in0_welford_id), so the unpack-to-DEST fp32
    // path is active on both branches iff the alias is active. In the
    // mcast/no_mcast kernels the TILIZE_IN branch tilizes directly into the
    // unpack-fp32 DFB without an alias, so those kernels need the unpack-fp32
    // state and the alias gating to be tracked independently.
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    constexpr uint32_t dfb_in0_welford_id = get_named_compile_time_arg_val("cb_in0_welford");
    constexpr uint32_t dfb_in_welford_id = get_named_compile_time_arg_val("cb_in_welford");
    // True when a reconfig-relevant operand is fp32: the per-tile reconfig_data_format calls below
    // are then required. All-bf16 compiles them out (no-ops). See program factory.
    constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // input cbs
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_in_id = tt::CBIndex::c_1;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_7;
#ifdef TILIZE_IN
    constexpr uint32_t dfb_welford_in_id = dfb_in_welford_id;
    constexpr uint32_t dfb_normalization_in_id = dfb_in_id;
#else
    constexpr uint32_t dfb_welford_in_id = dfb_in0_welford_id;
    constexpr uint32_t dfb_normalization_in_id = dfb_in0_id;
#endif

    // interm cbs
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t dfb_x_id = tt::CBIndex::c_13;
    constexpr uint32_t dfb_xmm_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_17;

    // output dfb_id
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t dfb_out_id =
        (do_gamma or do_beta) ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? dfb_in_id : dfb_out0_id)
                              : dfb_out0_id;
#endif

#ifdef UNTILIZE_OUT
    constexpr int dfb_outgamma_id = dfb_in_id;
    constexpr int dfb_outbeta_id = do_gamma ? dfb_out_id : dfb_in_id;
    constexpr int dfb_untilize_in_id = (do_gamma and not do_beta) ? dfb_outgamma_id
                                       : do_beta                  ? dfb_outbeta_id
                                                                  : dfb_out_id;
    constexpr int dfb_untilize_out_id =
#ifdef READER_REPACK
        dfb_repack_out_id;
#else
        dfb_out0_id;
#endif
#else
    constexpr int dfb_outgamma_id = do_beta ? dfb_in_id : dfb_out0_id;
    constexpr int dfb_outbeta_id = dfb_out0_id;
#endif

    constexpr auto offset_scalar_input = [](uint32_t dfb_id, ckl::WaitPolicy wait, ckl::PopPolicy pop) {
        return ckl::input(
            dfb_id,
            wait,
            pop,
            ckl::InputTileMapping::Scalar,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileAddressing::Offset);
    };
    constexpr auto streaming_input = [](uint32_t dfb_id) {
        return ckl::input(dfb_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto streaming_output = [](uint32_t dfb_id) {
        return ckl::output(
            dfb_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    };

    DataflowBuffer dfb_beta(dfb_beta_id);
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_gamma(dfb_gamma_id);
    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer dfb_in_welford(dfb_in_welford_id);
    DataflowBuffer dfb_input_mask(dfb_input_mask_id);

// tilize input from RM to tile layout
#ifdef TILIZE_IN
    compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in_id);
// Tilize in0 -> in (row-major to tiled)
#ifdef READER_REPACK
    constexpr uint32_t dfb_in_rm_id = dfb_repack_id;
    ckl::tilize<
        per_core_N,
        dfb_in_rm_id,
        dfb_in_id,
        ckl::tilize_config::InitUninitMode::InitAndUninit,
        ckl::tilize_config::WaitMode::WaitBlock,
        ckl::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#else
    constexpr uint32_t dfb_in_rm_id = dfb_in0_id;
    ckl::tilize<
        per_core_N,
        dfb_in_rm_id,
        dfb_in_id,
        ckl::tilize_config::InitUninitMode::InitAndUninit,
        ckl::tilize_config::WaitMode::NoWait,
        ckl::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
    dfb_in.wait_front(per_core_MN);
    if constexpr (welford_fp32_alias) {
        // Mirror the tilize push on the alias (c_31, shares SRAM with dfb_in / c_1) so it tracks
        // dfb_in's state. Must be done in compute: the producer of dfb_in is the
        // tilize call above (a compute op), not the reader; the reader never writes dfb_in.
        dfb_in_welford.reserve_back(per_core_MN);
        dfb_in_welford.push_back(per_core_MN);
        dfb_in_welford.wait_front(per_core_MN);
    }
#else
    compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in0_id);
#endif

    // Sharded v2 does not use reciprocal lookup table, so we pass an empty array
    constexpr std::array<uint32_t, 0> empty_reciprocal_lut{};

    dfb_eps.wait_front(1);
    dfb_input_mask.wait_front(num_tiles_input_mask);

    if constexpr (do_gamma) {
        dfb_gamma.wait_front(per_core_N);
    }
    if constexpr (do_beta) {
        dfb_beta.wait_front(per_core_N);
    }

    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t tile_id = b * block_hw;
        dfb_ex_partial.reserve_back(2);
        if constexpr (welford_fp32_alias) {
            // Reconfigure the transpose op for the alias buffer index consumed by the
            // welford loop below.
            transpose_init(dfb_welford_in_id);
        } else {
            transpose_init(dfb_in0_id);
        }
        tile_regs_acquire();
        welford_init();

        uint32_t block_xy_coord = 0;

        for (uint32_t g = 0; g < num_groups; ++g) {
            welford_save_state(mean_dst, g);
        }

        for (uint32_t i = 0; i < block_h; ++i) {
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

            // This tracks the global index of the first element in a given group in a tile.
            // It is used by the Welford's algorithm to scale the running mean and m2.
            // This moves reverse of channels_left, except that it is the global index.
            uint32_t curr_xy_coord = block_xy_coord;

            for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                transpose_init(dfb_welford_in_id);
                transpose_tile(dfb_welford_in_id, tile_id, input_dst);

                // Re-establish the welford SFPU replay buffer state. When transpose_tile
                // takes the unpack-to-DEST fp32 path, transpose_tile calls
                // llk_math_transpose_dest, whose math-side init records slots [16, 32) of
                // the math-thread replay buffer, clobbering welford's LREG2 / LREG3 portions.
                // Without welford_init<WelfordInitMode::PreserveStats>(), welford_update_rows would replay stale
                // transpose-dest ops.
                // When the unpack-to-DEST fp32 path is inactive, transpose_tile routes
                // through SrcA without touching the math-thread replay buffer, so re-init is
                // not needed.
                if constexpr (welford_fp32_alias) {
                    welford_init<WelfordInitMode::PreserveStats>();
                }

                uint32_t group_offset = 0;
                for (uint32_t g = min_group; g < num_groups; ++g) {
                    // Start Welford's Calculation
                    uint32_t cols_available = tile_width - group_offset;
                    uint32_t cols_consumed = std::min(cols_available, channels_left);

                    welford_restore_state(mean_dst, g);
                    welford_update_rows<0>(input_dst, curr_xy_coord, group_offset, cols_consumed, empty_reciprocal_lut);
                    welford_save_state(mean_dst, g);

                    channels_left -= cols_consumed;
                    group_offset += cols_consumed;
                    curr_xy_coord += cols_consumed;

                    // There are still channels left to be processed for the current group
                    // This can only be done in the next tile. So we don't do any more groups
                    // for this tile.
                    if (channels_left > 0) {
                        break;
                    }

                    // Since we know that channels_left is 0, it also means that we have
                    // processed all the channels for the current group.
                    // We update the min_group so we never revisit this group again.
                    ++min_group;
                    channels_left = num_channels_per_group;
                    curr_xy_coord = block_xy_coord;

                    // All available columns have been used for this tile, so we don't do any
                    // more groups for this tile.
                    if (group_offset == tile_width) {
                        break;
                    }
                }
                ++tile_id;
            }
            block_xy_coord += num_channels_per_group;
        }

        for (uint32_t g = 0; g < num_groups; ++g) {
            // Convert M2 to variance
            welford_restore_state(mean_dst, g);
            welford_finalize_to_face<0>(mean_dst, g, block_xy_coord - 1, empty_reciprocal_lut);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_block(mean_dst, dfb_ex_partial_id, 2);
        tile_regs_release();
        dfb_ex_partial.push_back(2);

        // Start Variance Calc
        // Wait for final welford values in dfb_ex_global_id
        dfb_ex_global.wait_front(2 * num_groups);
        // fp32: dfb_ex_global is fp32 (var), dfb_eps is bf16; the welford intake left SrcA on the fp32 input alias.
        if constexpr (enable_fp32_reconfig) {
            reconfig_data_format_srca(dfb_ex_global_id);
        }
        reconfig_data_format_srcb(dfb_eps_id);
        for (uint32_t g = 0; g < num_groups; ++g) {
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    offset_scalar_input(dfb_ex_global_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                    ckl::input(
                        dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled)>{
                    1 + (g << 1), 0u},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
                ckl::PackTile<streaming_output(dfb_ex2pe_id)>{});
        }
        // End Variance Calc

        dfb_ex2pe.wait_front(num_groups);

        // Start Final Val Calc
        tile_id = b * block_hw;
        for (uint32_t i = 0; i < block_h; ++i) {
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
                uint32_t group_offset = 0;
                for (uint32_t g = min_group; g < num_groups; ++g) {
                    // // Now let us do the actual computation for the current group here
                    // // a. x-u
                    reconfig_data_format(dfb_in0_id, dfb_ex_global_id);
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Sub,
                            offset_scalar_input(dfb_normalization_in_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                            ckl::input(
                                offset_scalar_input(dfb_ex_global_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                                ckl::BroadcastDim::Scalar)>{tile_id, g << 1},
                        ckl::PackTile<streaming_output(dfb_xmm_id)>{});

                    // Normalize the centered input: (x - mean) * rsqrt(variance + epsilon).
                    reconfig_data_format(dfb_in0_id, dfb_xmm_id, dfb_ex_global_id, dfb_ex2pe_id);
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Mul,
                            streaming_input(dfb_xmm_id),
                            ckl::input(
                                offset_scalar_input(dfb_ex2pe_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                                ckl::BroadcastDim::Scalar)>{0u, g},
                        ckl::PackTile<streaming_output(dfb_xmm_id)>{});

                    // Apply the current group's row selector.
                    const uint32_t mask_offset = g * block_w;
                    const uint32_t mask_index = mask_offset + block_w_index;
                    reconfig_data_format(dfb_xmm_id, dfb_xmm_id, dfb_ex2pe_id, dfb_input_mask_id);
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Mul,
                            streaming_input(dfb_xmm_id),
                            ckl::input(
                                offset_scalar_input(dfb_input_mask_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                                ckl::BroadcastDim::Row)>{0u, mask_index},
                        ckl::PackTile<streaming_output(dfb_xmm_id)>{});

                    // Accumulate contributions when a tile spans multiple groups.
                    if (group_offset == 0) {
                        ckl::copy<streaming_input(dfb_xmm_id), streaming_output(dfb_x_id)>(
                            ckl::IterationShape::one_tile());
                    } else {
                        // Not the first group for this tile: add what is already in dfb_x.
                        reconfig_data_format_srca(dfb_xmm_id, dfb_x_id);
                        reconfig_data_format_srcb(dfb_input_mask_id, dfb_xmm_id);
                        ckl::add<streaming_input(dfb_x_id), streaming_input(dfb_xmm_id), streaming_output(dfb_x_id)>(
                            ckl::IterationShape::one_tile());
                    }

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
                ++tile_id;

                if constexpr (do_gamma) {
                    // fp32: reset SrcA to dfb_x.
                    if constexpr (enable_fp32_reconfig) {
                        reconfig_data_format_srca(dfb_x_id);
                    }
                    reconfig_data_format_srcb(dfb_xmm_id, dfb_gamma_id);
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Mul,
                            streaming_input(dfb_x_id),
                            ckl::input(
                                offset_scalar_input(dfb_gamma_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                                ckl::BroadcastDim::Row)>{0u, nt},
                        ckl::PackTile<streaming_output(dfb_x_id)>{});
                }

                if constexpr (do_beta) {
                    // fp32: reset SrcA to dfb_x.
                    if constexpr (enable_fp32_reconfig) {
                        reconfig_data_format_srca(dfb_x_id);
                    }
                    reconfig_data_format_srcb(do_gamma ? dfb_gamma_id : dfb_xmm_id, dfb_beta_id);
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Add,
                            streaming_input(dfb_x_id),
                            ckl::input(
                                offset_scalar_input(dfb_beta_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                                ckl::BroadcastDim::Row)>{0u, nt},
                        ckl::PackTile<streaming_output(dfb_x_id)>{});
                }

                // Write out the final output
#ifdef UNTILIZE_OUT
                constexpr auto write_dfb_id = dfb_untilize_in_id;
#else
                constexpr auto write_dfb_id = dfb_out0_id;
#endif
                // fp32: reset SrcA to dfb_x (fp32).
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_x_id);
                }
                reconfig_data_format_srcb(do_beta ? dfb_beta_id : dfb_xmm_id, dfb_x_id);
#ifndef UNTILIZE_OUT
                // The streaming output disables automatic reconfiguration, so select the fp32 output format.
                // Packer was last set for bf16 dfb_xmm; reconfigure to write_dfb_id (may be fp32) before pack, restore
                // after. Gated out for bf16 (no format change).
                if constexpr (enable_fp32_reconfig) {
                    pack_reconfig_data_format(write_dfb_id);
                }
#endif
                ckl::copy<streaming_input(dfb_x_id), streaming_output(write_dfb_id)>(ckl::IterationShape::one_tile());
#ifndef UNTILIZE_OUT
                if constexpr (enable_fp32_reconfig) {
                    pack_reconfig_data_format(dfb_xmm_id);
                }
#endif
            }
        }

        dfb_ex_global.pop_front(2 * num_groups);
        dfb_ex2pe.pop_front(num_groups);
    }

    dfb_eps.pop_front(1);
    dfb_input_mask.pop_front(num_tiles_input_mask);

    // Pop all the dfb_beta_id and dfb_gamma_id if used
    if constexpr (do_beta) {
        dfb_beta.pop_front(per_core_N);
    }
    if constexpr (do_gamma) {
        dfb_gamma.pop_front(per_core_N);
    }

#ifdef UNTILIZE_OUT
    // untilize - DEST capacity auto-detected
    ckl::untilize<
        per_core_N,
        dfb_untilize_in_id,
        dfb_untilize_out_id,
        ckl::untilize_config::InitUninitMode::InitAndUninit,
        ckl::untilize_config::WaitMode::WaitUpfront,
        ckl::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
}
