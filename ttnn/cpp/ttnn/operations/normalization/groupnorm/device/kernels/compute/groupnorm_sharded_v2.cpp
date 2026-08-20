// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr uint32_t is_mcast_sender = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_per_mcast_group = get_compile_time_arg_val(3);
    // True when a reconfig-relevant operand is fp32: the per-group reconfig_data_format calls below
    // are then required. All-bf16 compiles them out (no-ops). See program factory.
    constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;

    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);

    constexpr uint32_t num_cols_per_group = get_compile_time_arg_val(6);

    volatile uint32_t block_h = get_compile_time_arg_val(7);
    constexpr uint32_t block_w = get_compile_time_arg_val(8);
    constexpr uint32_t block_hw = get_compile_time_arg_val(9);

    constexpr uint32_t subblock_w = get_compile_time_arg_val(10);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(11);

    constexpr uint32_t per_core_M = get_compile_time_arg_val(12);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(13);
    constexpr uint32_t per_core_MN = get_compile_time_arg_val(14);

    constexpr uint32_t per_core_N_tile_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t num_groups_per_reset = get_compile_time_arg_val(16);

    constexpr uint32_t single_tile_size_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t num_tiles_per_batch = get_compile_time_arg_val(18);

    constexpr uint32_t num_tiles_input_mask = get_compile_time_arg_val(19);
    constexpr uint32_t block_w_last = get_compile_time_arg_val(20);
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_compile_time_arg_val(21);
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W = get_compile_time_arg_val(22);
    constexpr uint32_t group_row_offset = get_compile_time_arg_val(23);
    constexpr uint32_t tile_width = get_compile_time_arg_val(24);

    // Non-tile-aligned H*W; see compute/groupnorm.cpp. logical_hw / padded_hw only keep two shapes
    // padding to the same size out of one cached program; has_row_mask is what this branches on.
    constexpr uint32_t logical_hw [[maybe_unused]] = get_compile_time_arg_val(25);
    constexpr uint32_t padded_hw [[maybe_unused]] = get_compile_time_arg_val(26);
    constexpr bool has_row_mask = get_compile_time_arg_val(27) == 1;
    // Composed-mask protocol: the writer ships ONE row-0-only column-selector set per (batch,
    // group) unconditionally; under has_row_mask this kernel composes the batch's final
    // row-tile's mask on device as rowvalid (c_18) x column selector -> c_19.
    constexpr uint32_t mask_tiles_per_group = block_w;

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;
    // group_row_offset is the signed difference encoded in uint32_t by the host. Modular
    // addition reconstructs the original group size for both positive and negative offsets.
    constexpr uint32_t full_group_size = (block_w_minus_one * tile_width) + group_row_offset;
    constexpr uint32_t group_start_stride = full_group_size % tile_width;

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // input cbs
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_in_id = tt::CBIndex::c_1;
    constexpr uint32_t dfb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_scaler_global_id = tt::CBIndex::c_4;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_7;
    // Pass-1 operand source: tilized-input CB when the reader tilizes, raw input CB otherwise.
    // (The init calls keep using dfb_in0_id, as before -- only the ops switch.)
#ifdef TILIZE_IN
    constexpr uint32_t dfb_pass1_src_id = dfb_in_id;
#else
    constexpr uint32_t dfb_pass1_src_id = dfb_in0_id;
#endif

    // interm cbs
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t dfb_x_id = tt::CBIndex::c_13;
    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t dfb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t dfb_ex_global_id = num_cores_per_mcast_group == 1 ? dfb_ex_partial_id : tt::CBIndex::c_15;
    constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_17;
    constexpr uint32_t dfb_ones_id = tt::CBIndex::c_26;
    // Composed-mask CBs, created only under pad correction (has_row_mask); aliased to
    // always-present CBs otherwise.
    constexpr uint32_t dfb_rowvalid_id = has_row_mask ? tt::CBIndex::c_18 : tt::CBIndex::c_26;
    constexpr uint32_t dfb_mask_last_id = has_row_mask ? tt::CBIndex::c_19 : tt::CBIndex::c_7;

    // output cb
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    // not used in cases of negative mask
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t dfb_out_id =
        (do_gamma or do_beta) ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? dfb_in_id : dfb_out0_id)
                              : dfb_out0_id;
#endif

    // tile offset
    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
    uint32_t index_w_offset = 0;
    uint32_t index_b_offset = 0;
    uint32_t index_g_offset = 0;
    // data offset
    uint32_t num_datum_per_row_offeset = 0;
    // inplace out cbs
    bool copy_or_add = true;
    uint32_t group_reset_index = 0;
    uint32_t index_block_w = 0;
    uint32_t row_offset = num_cols_per_group;

#ifdef UNTILIZE_OUT
#ifndef FUSE_NEGATIVE_MASK
    constexpr int dfb_outgamma_id = dfb_in_id;
    constexpr int dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_out_id;
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
    constexpr int dfb_outgamma_id = dfb_in_id;
    constexpr int dfb_inbeta_id = dfb_in_id;
    constexpr int dfb_outbeta_id = dfb_in_id;
    constexpr int dfb_untilize_in_id = dfb_in_id;
    constexpr int dfb_untilize_out_id =
#ifdef READER_REPACK
        dfb_repack_out_id;
#else
        dfb_out0_id;
#endif
#endif
#else
    constexpr int dfb_outgamma_id = do_beta ? dfb_in_id : dfb_out0_id;
    constexpr int dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_out_id;
    constexpr int dfb_outbeta_id = dfb_out0_id;
#endif

    // Used in cases of negative mask provided
    constexpr uint32_t dfb_in_negative_mask_id = tt::CBIndex::c_14;

#ifdef FUSE_NEGATIVE_MASK
    constexpr bool use_negative_mask = true;
#else
    constexpr bool use_negative_mask = false;
#endif

    constexpr auto strided_col_input = [](uint32_t dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Col,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileAddressing::Strided);
    };
    constexpr auto strided_block_input = [](uint32_t dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileAddressing::Strided);
    };
    constexpr auto strided_output = [](uint32_t dfb_id) {
        return ckl::output(
            dfb_id,
            ckl::ReservePolicy::None,
            ckl::PushPolicy::None,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileAddressing::Strided);
    };

    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_ex(dfb_ex_id);
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer dfb_in_negative_mask(dfb_in_negative_mask_id);
    DataflowBuffer dfb_input_mask(dfb_input_mask_id);
    DataflowBuffer dfb_mask_last(dfb_mask_last_id);
    DataflowBuffer dfb_out(dfb_out_id);
    DataflowBuffer dfb_outbeta(dfb_outbeta_id);
    DataflowBuffer dfb_outgamma(dfb_outgamma_id);
    DataflowBuffer dfb_rowvalid(dfb_rowvalid_id);
    DataflowBuffer dfb_x(dfb_x_id);

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
#else
    compute_kernel_hw_startup(dfb_in0_id, dfb_input_mask_id, dfb_x_id);
#endif

    if constexpr (has_row_mask) {
        dfb_rowvalid.wait_front(1);
    }

    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        uint32_t group_start_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {
            const uint32_t valid_block_w = (group_start_offset + full_group_size + tile_width - 1) / tile_width;
            const uint32_t synchronized_block_w = ((valid_block_w + subblock_w - 1) / subblock_w) * subblock_w;
            const uint32_t math_block_w = synchronized_block_w == block_w ? valid_block_w : block_w;
            const auto valid_group_shape =
                ckl::IterationShape::grid(block_h, math_block_w).block_size(subblock_w, ckl::BlockTailSync::FullBlock);

            // mask input
            index_h_offset = index_b_offset + index_g_offset;
            reconfig_data_format_srcb(dfb_in0_id, dfb_input_mask_id);
            dfb_input_mask.wait_front(mask_tiles_per_group);
            // Compose the final row-tile's mask: rowvalid[r] * colsel[c] -> dfb_mask_last. The
            // column selector's row 0 is broadcast down the rowvalid tile, so the product
            // varies down the rows.
            if constexpr (has_row_mask) {
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_rowvalid_id);
                    pack_reconfig_data_format(dfb_mask_last_id);
                }
                mul_bcast_rows_init(dfb_rowvalid_id, dfb_input_mask_id);
                dfb_mask_last.reserve_back(block_w);
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        mul_tiles_bcast_rows(dfb_rowvalid_id, dfb_input_mask_id, 0, w + index_subblock_w_offset, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        pack_tile(w, dfb_mask_last_id);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                dfb_mask_last.push_back(block_w);
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_in0_id);
                    pack_reconfig_data_format(dfb_x_id);
                }
            }
            dfb_x.reserve_back(block_hw);
            const uint32_t bcast_row_tiles = has_row_mask ? block_h - 1 : block_h;
            mul_bcast_rows_init(dfb_in0_id, dfb_input_mask_id);
            for (uint32_t i = 0; i < bcast_row_tiles; ++i) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        // When the last group spans fewer than block_w tiles, the index can
                        // exceed the DFB tile count. Clamp it so the read stays in bounds;
                        // the input mask guarantees the result from the clamped tile is zeroed.
                        if (index >= per_core_MN) {
                            index = per_core_MN - 1;
                        }
                        mul_tiles_bcast_rows(
                            dfb_pass1_src_id, dfb_input_mask_id, index, w + index_subblock_w_offset, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, dfb_x_id);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }
            // Peeled final row-tile: full-tile multiply with the composed mask.
            if constexpr (has_row_mask) {
                dfb_mask_last.wait_front(block_w);
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_mask_last_id);
                mul_init(dfb_in0_id, dfb_mask_last_id);
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        if (index >= per_core_MN) {
                            index = per_core_MN - 1;
                        }
                        mul_tiles(dfb_pass1_src_id, dfb_mask_last_id, index, w + index_subblock_w_offset, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, dfb_x_id);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }
            dfb_x.push_back(block_hw);
            reconfig_data_format_srcb(has_row_mask ? dfb_mask_last_id : dfb_input_mask_id, dfb_ones_id);
            // Partial-E[x]
            dfb_x.wait_front(block_hw);
            // Accumulate into dest directly by using mul_tiles (tile * 1 is accumulated into dest)
            // Alternative is to use reduce_tile multiple times, but this showed to be more precise and faster.
            ckl::eltwise_chain(
                valid_group_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    strided_block_input(dfb_x_id),
                    ckl::input(
                        dfb_ones_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
                    ckl::Dst::D0,
                    ckl::DestAccumulation::WholeShape>{ckl::StridedTileRange{0, block_w}},
                ckl::PackTile<ckl::output(
                    dfb_ex2pe_id,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Disabled,
                    ckl::TileAddressing::Direct,
                    ckl::DestAccumulation::WholeShape)>{});

            // reduce only one final tile
            //
            // Note that reader_mcast_sender_unary_sharded_gn_v2.cpp depends on the
            // documented behavior of REDUCE_SCALAR's packer to set every
            // non-result datum of dfb_ex_partial to zero.
            // If this `reduce<…, REDUCE_SCALAR>` pack into dfb_ex_partial is
            // ever replaced by something that does not have the same
            // packer-zero contract (e.g. a `pack_tile` / `pack_block`
            // path like welford_groupnorm_sharded_v2.cpp uses), the sharded
            // reader's "single-tile-overwrite trick" must be adjusted accordingly
            // (e.g. use `zero_whole_cb` from groupnorm_zero_fill.hpp, mirroring the
            // mcast reader). Same applies to the second REDUCE_SCALAR pack into
            // dfb_ex_partial later in this kernel (variance).
            ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, dfb_ex2pe_id, dfb_scaler_id, dfb_ex_partial_id>(
                ckl::ReduceInputBlockShape::single());

            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(ckl::ReduceInputBlockShape::single());
                dfb_ex.reserve_back(1);
                dfb_ex.push_back(1);
            }

            // fp32: reset both srcs so fp32 x/mean aren't read through the partial-E[x] bf16 dfb_ones format.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_x_id);
                reconfig_data_format_srcb(dfb_ex_global_id);
            }
            ckl::sub<
                ckl::input(
                    dfb_x_id,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb_ex_global_id,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb_x_id,
                    ckl::ReservePolicy::PerBlockSize,
                    ckl::PushPolicy::PerBlockSize,
                    ckl::DataFormatReconfig::Disabled)>(valid_group_shape);

            reconfig_data_format_srcb(dfb_ex_global_id, dfb_input_mask_id);
            // Re-mask (x - E[x]); otherwise each padding row is centered to (garbage - E[x])
            // and squared into the variance.
            if constexpr (has_row_mask) {
                dfb_x.wait_front(block_hw);
            }

            if (bcast_row_tiles > 0) {
                ckl::mul<
                    ckl::input(
                        dfb_x_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_input_mask_id,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_x_id,
                        ckl::ReservePolicy::PerBlockSize,
                        ckl::PushPolicy::PerBlockSize,
                        ckl::DataFormatReconfig::Disabled)>(
                    has_row_mask ? ckl::IterationShape::grid(bcast_row_tiles, block_w).block_size(subblock_w)
                                 : valid_group_shape);
            }

            // Peeled final row-tile: full-tile multiply with the composed mask.
            if constexpr (has_row_mask) {
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_mask_last_id);
                ckl::mul<
                    ckl::input(
                        dfb_x_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_mask_last_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Row,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_x_id,
                        ckl::ReservePolicy::PerBlockSize,
                        ckl::PushPolicy::PerBlockSize,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(1, block_w).block_size(subblock_w));
            }
            dfb_input_mask.pop_front(mask_tiles_per_group);
            if constexpr (has_row_mask) {
                dfb_mask_last.pop_front(block_w);
            }
            reconfig_data_format_srcb(has_row_mask ? dfb_mask_last_id : dfb_input_mask_id, dfb_x_id);

            // (x - E[x])^2
            dfb_x.wait_front(block_hw);
            ckl::eltwise_chain(
                valid_group_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    strided_block_input(dfb_x_id),
                    strided_block_input(dfb_x_id),
                    ckl::Dst::D0,
                    ckl::DestAccumulation::WholeShape>{
                    ckl::StridedTileRange{0, block_w}, ckl::StridedTileRange{0, block_w}},
                ckl::PackTile<ckl::output(
                    dfb_ex2pe_id,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Disabled,
                    ckl::TileAddressing::Direct,
                    ckl::DestAccumulation::WholeShape)>{});

            // If modifying this code, see the long comment at the first REDUCE_SCALAR
            // pack into dfb_ex_partial earlier in this kernel.
            // The sharded reader's "single-tile-overwrite trick" depends on
            // this pack also clearing every non-result datum of dfb_ex_partial
            // to exact zero (documented packer behavior for REDUCE_SCALAR).
            ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, dfb_ex2pe_id, dfb_scaler_id, dfb_ex_partial_id>(
                ckl::ReduceInputBlockShape::single());

            dfb_ex_partial.wait_front(1);
            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(ckl::ReduceInputBlockShape::single());
                dfb_ex.reserve_back(1);
                dfb_ex.push_back(1);
            }

            // global reduce results
            dfb_eps.wait_front(1);

            // fp32: reset both srcs so bf16 eps isn't read through the (x-Ex)^2 fp32 format (else garbage var+eps).
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_ex_global_id);
                reconfig_data_format_srcb(dfb_eps_id);
            }
            // The row mask keeps padding out of both sums, so the reduced value is the variance
            // over real rows. Compute 1/sqrt(variance + epsilon).
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ckl::input(
                        dfb_ex_global_id,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled)>{},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(
                    dfb_ex2pe_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>{});
            //  (x - Ex) * 1/[sqrt(Var + eps)]
            // fp32: reset both srcs so fp32 x/rstd aren't read through the (var+eps) bf16 dfb_eps format.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_x_id);
                reconfig_data_format_srcb(dfb_ex2pe_id);
            }
            ckl::mul<
                ckl::input(
                    dfb_x_id,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb_ex2pe_id,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb_x_id,
                    ckl::ReservePolicy::PerBlockSize,
                    ckl::PushPolicy::PerBlockSize,
                    ckl::DataFormatReconfig::Disabled)>(valid_group_shape);
            dfb_x.wait_front(block_hw);
            //  add or copy with previous output results
            uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

            // if we are using negative mask, we are overlapping tilized in and out, otherwise they are 2 separate
            // buffers.
            if constexpr (use_negative_mask == false) {
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    const ckl::StridedTileRange input_range{w, block_w};
                    const ckl::StridedTileRange output_range{index_b_offset + index_g_offset + w, per_core_N};
                    if (copy_or_add) {
                        ckl::eltwise_chain(
                            ckl::IterationShape::col(block_h),
                            ckl::CopyTile<strided_col_input(dfb_x_id)>{input_range},
                            ckl::PackTile<strided_output(dfb_out_id)>{output_range});
                    } else {
                        ckl::eltwise_chain(
                            ckl::IterationShape::col(block_h),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Add,
                                strided_col_input(dfb_out_id),
                                strided_col_input(dfb_x_id)>{output_range, input_range},
                            ckl::PackTile<strided_output(dfb_out_id)>{output_range});
                    }

                    // update group tile offset
                    if (index_block_w >= block_w_curr - 1) {
                        index_block_w = 0;

                        if (group_reset_index == num_groups_per_reset - 1) {
                            copy_or_add = true;

                            group_reset_index = 0;
                        } else {
                            copy_or_add = false;

                            group_reset_index += 1;
                        }
                    } else {
                        copy_or_add = true;
                        index_block_w += 1;
                    }
                }
            } else {
                // zero out values in dfb_tilized_in_id input by multiplying with negative mask for the current group
                dfb_in_negative_mask.wait_front(block_w);
                const ckl::StridedTileRange output_range{index_b_offset + index_g_offset, per_core_N};
                reconfig_data_format_srcb(dfb_x_id, dfb_in_negative_mask_id);
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(block_h, block_w_curr),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        strided_block_input(dfb_in_id),
                        ckl::input(
                            dfb_in_negative_mask_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Row,
                            ckl::DataFormatReconfig::Disabled)>{output_range},
                    ckl::PackTile<strided_output(dfb_in_id)>{output_range});

                // data in dfb_x_id has valid data only for current group
                // dfb_in_id has cleared data for that group
                // just add them together
                reconfig_data_format_srcb(dfb_in_negative_mask_id, dfb_x_id);
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(block_h, block_w_curr),
                    ckl::
                        BinaryFpu<ckl::BinaryFpuOp::Add, strided_block_input(dfb_in_id), strided_block_input(dfb_x_id)>{
                            output_range, ckl::StridedTileRange{0u, block_w}},
                    ckl::PackTile<strided_output(dfb_in_id)>{output_range});
                dfb_in_negative_mask.pop_front(block_w);
            }

            dfb_x.pop_front(block_hw);

            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == tile_width) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > tile_width) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }

            group_start_offset += group_start_stride;
            if (group_start_offset >= tile_width) {
                group_start_offset -= tile_width;
            }
        }
        index_b_offset += num_tiles_per_batch;
    }

    if constexpr (use_negative_mask == false) {
        dfb_out.push_back(per_core_MN);
        dfb_in.pop_front(per_core_MN);

    } else {
        // nothing, for the negative mask implementation, dfb_in_id is the only dfb_id in use, and it already has the
        // data required for the rest of kernel.
    }

    if constexpr (do_gamma) {
        if constexpr (use_negative_mask == false) {
            // fp32: reset both srcs so bf16 gamma isn't read through the normalization loop's fp32 format.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_out_id);
                reconfig_data_format_srcb(dfb_gamma_id);
            }
            ckl::mul<
                ckl::input(
                    dfb_out_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb_gamma_id,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb_outgamma_id,
                    ckl::ReservePolicy::Upfront,
                    ckl::PushPolicy::AtEnd,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::grid(per_core_M, per_core_N));
            dfb_outgamma.wait_front(per_core_MN);
        } else {
            // dfb_in has data required for gamma, so we do it inplace
            // fp32: see non-negative-mask branch above.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_in_id);
                reconfig_data_format_srcb(dfb_gamma_id);
            }
            ckl::mul<
                ckl::input(
                    dfb_in_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb_gamma_id,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb_in_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::grid(per_core_M, per_core_N));
        }
    }

    if constexpr (do_beta) {
        if constexpr (use_negative_mask == false) {
            // fp32: reset both srcs so bf16 beta isn't read as fp32 (matters especially when do_gamma=false).
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_inbeta_id);
                reconfig_data_format_srcb(dfb_beta_id);
            }
            ckl::add<
                ckl::input(
                    dfb_inbeta_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb_beta_id,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb_outbeta_id,
                    ckl::ReservePolicy::Upfront,
                    ckl::PushPolicy::AtEnd,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::grid(per_core_M, per_core_N));
            dfb_outbeta.wait_front(per_core_MN);
        } else {
            // dfb_in_id has data required for beta, so we do it inplace
            // fp32: see non-negative-mask branch above.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_in_id);
                reconfig_data_format_srcb(dfb_beta_id);
            }
            ckl::add<
                ckl::input(
                    dfb_in_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb_beta_id,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb_in_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::grid(per_core_M, per_core_N));
        }
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
