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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr uint32_t is_mcast_sender = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_per_mcast_group = get_compile_time_arg_val(3);

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

    // Non-tile-aligned H*W (#50682); derivation and precision caveat in compute/groupnorm.cpp.
    // Unlike the interleaved kernel this one consumes the mean during centering, so K*E[x]^2 is
    // computed before centering and stashed in dfb_kmsq until the rsqrt step.
    constexpr uint32_t logical_hw = get_compile_time_arg_val(25);
    constexpr uint32_t padded_hw = get_compile_time_arg_val(26);
    constexpr bool has_pad_correction = padded_hw != logical_hw;

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
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_in_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler_global_id = tt::CBIndex::c_4;
    constexpr uint32_t cb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask_id = tt::CBIndex::c_7;

    // interm cbs
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t cb_x_id = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t cb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_global_id = num_cores_per_mcast_group == 1 ? cb_ex_partial_id : tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2pe_id = tt::CBIndex::c_17;
    constexpr uint32_t cb_ones_id = tt::CBIndex::c_26;

    // #50682 pad-correction CBs. dfb_k holds K from the writer; dfb_msq is transient scratch;
    // dfb_kmsq carries K*E[x]^2 across the centering loop.
    constexpr uint32_t dfb_k_id = tt::CBIndex::c_18;
    constexpr uint32_t dfb_msq_id = tt::CBIndex::c_19;
    constexpr uint32_t dfb_kmsq_id = tt::CBIndex::c_20;

    // output cb
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    // not used in cases of negative mask
    constexpr uint32_t cb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out_id =
        (do_gamma or do_beta) ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? cb_in_id : cb_out0_id)
                              : cb_out0_id;
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
    constexpr int cb_outgamma_id = cb_in_id;
    constexpr int cb_inbeta_id = do_gamma ? cb_outgamma_id : cb_out_id;
    constexpr int cb_outbeta_id = do_gamma ? cb_out_id : cb_in_id;
    constexpr int cb_untilize_in_id = (do_gamma and not do_beta) ? cb_outgamma_id : do_beta ? cb_outbeta_id : cb_out_id;
    constexpr int cb_untilize_out_id =
#ifdef READER_REPACK
        cb_repack_out_id;
#else
        cb_out0_id;
#endif
#else
    constexpr int cb_outgamma_id = cb_in_id;
    constexpr int cb_inbeta_id = cb_in_id;
    constexpr int cb_outbeta_id = cb_in_id;
    constexpr int cb_untilize_in_id = cb_in_id;
    constexpr int cb_untilize_out_id =
#ifdef READER_REPACK
        cb_repack_out_id;
#else
        cb_out0_id;
#endif
#endif
#else
    constexpr int cb_outgamma_id = do_beta ? cb_in_id : cb_out0_id;
    constexpr int cb_inbeta_id = do_gamma ? cb_outgamma_id : cb_out_id;
    constexpr int cb_outbeta_id = cb_out0_id;
#endif

    // Used in cases of negative mask provided
    constexpr uint32_t cb_in_negative_mask_id = tt::CBIndex::c_14;

#ifdef FUSE_NEGATIVE_MASK
    constexpr bool use_negative_mask = true;
#else
    constexpr bool use_negative_mask = false;
#endif

    constexpr auto strided_col_input = [](uint32_t cb) {
        return ckl::input(
            cb,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::OperandKind::Col,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileOffset::Strided);
    };
    constexpr auto strided_block_input = [](uint32_t cb) {
        return ckl::input(
            cb,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::OperandKind::Block,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileOffset::Strided);
    };
    constexpr auto strided_output = [](uint32_t cb) {
        return ckl::output(
            cb,
            ckl::ReservePolicy::None,
            ckl::PushPolicy::None,
            ckl::DataFormatReconfig::Disabled,
            ckl::PackRelu::Disabled,
            ckl::L1Accumulation::Disabled,
            ckl::DestAccumulation::Disabled,
            ckl::TileOffset::Strided);
    };

    DataflowBuffer cb_beta(cb_beta_id);
    DataflowBuffer cb_eps(cb_eps_id);
    DataflowBuffer cb_ex(cb_ex_id);
    DataflowBuffer cb_ex2pe(cb_ex2pe_id);
    DataflowBuffer cb_ex_external(cb_ex_external_id);
    DataflowBuffer cb_ex_global(cb_ex_global_id);
    DataflowBuffer cb_ex_partial(cb_ex_partial_id);
    DataflowBuffer cb_gamma(cb_gamma_id);
    DataflowBuffer cb_in(cb_in_id);
    DataflowBuffer cb_in_negative_mask(cb_in_negative_mask_id);
    DataflowBuffer cb_inbeta(cb_inbeta_id);
    DataflowBuffer cb_input_mask(cb_input_mask_id);
    DataflowBuffer cb_ones(cb_ones_id);
    DataflowBuffer cb_out(cb_out_id);
    DataflowBuffer cb_outbeta(cb_outbeta_id);
    DataflowBuffer cb_outgamma(cb_outgamma_id);
    DataflowBuffer cb_scaler(cb_scaler_id);
    DataflowBuffer cb_scaler_global(cb_scaler_global_id);
    DataflowBuffer cb_x(cb_x_id);

// tilize input from RM to tile layout
#ifdef TILIZE_IN
    compute_kernel_hw_startup(cb_in0_id, cb_in0_id, cb_in_id);
// Tilize in0 -> in (row-major to tiled)
#ifdef READER_REPACK
    constexpr uint32_t cb_in_rm_id = cb_repack_id;
    ckl::tilize<
        per_core_N,
        cb_in_rm_id,
        cb_in_id,
        ckl::tilize_config::InitUninitMode::InitAndUninit,
        ckl::tilize_config::WaitMode::WaitBlock,
        ckl::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#else
    constexpr uint32_t cb_in_rm_id = cb_in0_id;
    ckl::tilize<
        per_core_N,
        cb_in_rm_id,
        cb_in_id,
        ckl::tilize_config::InitUninitMode::InitAndUninit,
        ckl::tilize_config::WaitMode::NoWait,
        ckl::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
    cb_in.wait_front(per_core_MN);
#else
    compute_kernel_hw_startup(cb_in0_id, cb_input_mask_id, cb_x_id);
#endif

    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        uint32_t group_start_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {
            const uint32_t valid_block_w = (group_start_offset + full_group_size + tile_width - 1) / tile_width;
            const uint32_t synchronized_block_w = ((valid_block_w + subblock_w - 1) / subblock_w) * subblock_w;
            const uint32_t math_block_w = synchronized_block_w == block_w ? valid_block_w : block_w;
            const auto valid_group_shape =
                ckl::EltwiseShape::grid(block_h, math_block_w, ckl::BlockingSettings{subblock_w});

            // mask input
            index_h_offset = index_b_offset + index_g_offset;
            reconfig_data_format_srcb(cb_in0_id, cb_input_mask_id);
            mul_init(cb_in0_id, cb_input_mask_id);
            cb_x.reserve_back(block_hw);
            cb_input_mask.wait_front(block_w);
            for (uint32_t i = 0; i < block_h; ++i) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        // When the last group spans fewer than block_w tiles, the index can
                        // exceed the CB tile count. Clamp it so the read stays in bounds;
                        // the input mask guarantees the result from the clamped tile is zeroed.
                        if (index >= per_core_MN) {
                            index = per_core_MN - 1;
                        }
                        uint32_t index_mask = w + index_subblock_w_offset;
#ifdef TILIZE_IN
                        mul_tiles(cb_in_id, cb_input_mask_id, index, index_mask, w);
#else
                        mul_tiles(cb_in0_id, cb_input_mask_id, index, index_mask, w);
#endif
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, cb_x_id);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }
            cb_x.push_back(block_hw);
            // Partial-E[x]
            cb_x.wait_front(block_hw);
            reconfig_data_format_srcb(cb_input_mask_id, cb_ones_id);
            // Accumulate into dest directly by using mul_tiles (tile * 1 is accumulated into dest)
            // Alternative is to use reduce_tile multiple times, but this showed to be more precise and faster.
            ckl::eltwise_chain(
                valid_group_shape,
                ckl::BinaryFpu<
                    strided_block_input(cb_x_id),
                    ckl::input(
                        cb_ones_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::WholeShape>{ckl::StridedTileRange{0, block_w}},
                ckl::PackTile<ckl::output(
                    cb_ex2pe_id,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Disabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::WholeShape)>{});

            // reduce only one final tile
            //
            // Note that reader_mcast_sender_unary_sharded_gn_v2.cpp depends on the
            // documented behavior of REDUCE_SCALAR's packer to set every
            // non-result datum of cb_ex_partial to zero.
            // If this `reduce<…, REDUCE_SCALAR>` pack into cb_ex_partial is
            // ever replaced by something that does not have the same
            // packer-zero contract (e.g. a `pack_tile` / `pack_block`
            // path like welford_groupnorm_sharded_v2.cpp uses), the sharded
            // reader's "single-tile-overwrite trick" must be adjusted accordingly
            // (e.g. use `zero_whole_cb` from groupnorm_zero_fill.hpp, mirroring the
            // mcast reader). Same applies to the second REDUCE_SCALAR pack into
            // cb_ex_partial later in this kernel (variance).
            ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, cb_ex2pe_id, cb_scaler_id, cb_ex_partial_id>(
                ckl::ReduceInputBlockShape::single());

            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    cb_ex_external_id,
                    cb_scaler_global_id,
                    cb_ex_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(ckl::ReduceInputBlockShape::single());
                cb_ex.reserve_back(1);
                cb_ex.push_back(1);
            }
            ckl::sub<
                ckl::input(
                    cb_x_id,
                    ckl::WaitPolicy::PerChunk,
                    ckl::PopPolicy::PerChunk,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    cb_ex_global_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_x_id,
                    ckl::ReservePolicy::PerChunk,
                    ckl::PushPolicy::PerChunk,
                    ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::Scalar>(valid_group_shape);

            reconfig_data_format_srcb(cb_ex_global_id, cb_input_mask_id);
            ckl::mul<
                ckl::input(
                    cb_x_id,
                    ckl::WaitPolicy::PerChunk,
                    ckl::PopPolicy::PerChunk,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    cb_input_mask_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_x_id,
                    ckl::ReservePolicy::PerChunk,
                    ckl::PushPolicy::PerChunk,
                    ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::None>(valid_group_shape);
            cb_input_mask.pop_front(block_w);
            // (x - E[x])^2
            cb_x.wait_front(block_hw);
            reconfig_data_format_srcb(cb_input_mask_id, cb_x_id);
            ckl::eltwise_chain(
                valid_group_shape,
                ckl::BinaryFpu<
                    strided_block_input(cb_x_id),
                    strided_block_input(cb_x_id),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::Dst::D0,
                    ckl::DestAccumulation::WholeShape>{
                    ckl::StridedTileRange{0, block_w}, ckl::StridedTileRange{0, block_w}},
                ckl::PackTile<ckl::output(
                    cb_ex2pe_id,
                    ckl::ReservePolicy::PerOuter,
                    ckl::PushPolicy::PerOuter,
                    ckl::DataFormatReconfig::Disabled,
                    ckl::PackRelu::Disabled,
                    ckl::L1Accumulation::Disabled,
                    ckl::DestAccumulation::WholeShape)>{});

            // If modifying this code, see the long comment at the first REDUCE_SCALAR
            // pack into cb_ex_partial earlier in this kernel.
            // The sharded reader's "single-tile-overwrite trick" depends on
            // this pack also clearing every non-result datum of cb_ex_partial
            // to exact zero (documented packer behavior for REDUCE_SCALAR).
            ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, cb_ex2pe_id, cb_scaler_id, cb_ex_partial_id>(
                ckl::ReduceInputBlockShape::single());

            cb_ex_partial.wait_front(1);
            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    cb_ex_external_id,
                    cb_scaler_global_id,
                    cb_ex_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(ckl::ReduceInputBlockShape::single());
                cb_ex.reserve_back(1);
                cb_ex.push_back(1);
            }

            // global reduce results
            cb_eps.wait_front(1);
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_ex_global_id,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        cb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
                    ckl::BinaryFpuOp::Add,
                    ckl::BroadcastDim::None>{},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(
                    cb_ex2pe_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>{});
            ckl::mul<
                ckl::input(
                    cb_x_id,
                    ckl::WaitPolicy::PerChunk,
                    ckl::PopPolicy::PerChunk,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    cb_ex2pe_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_x_id,
                    ckl::ReservePolicy::PerChunk,
                    ckl::PushPolicy::PerChunk,
                    ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::Scalar>(valid_group_shape);
            cb_x.wait_front(block_hw);
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
                            ckl::EltwiseShape::col(block_h),
                            ckl::CopyTile<strided_col_input(cb_x_id)>{input_range},
                            ckl::PackTile<strided_output(cb_out_id)>{output_range});
                    } else {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::col(block_h),
                            ckl::BinaryFpu<
                                strided_col_input(cb_out_id),
                                strided_col_input(cb_x_id),
                                ckl::BinaryFpuOp::Add,
                                ckl::BroadcastDim::None>{output_range, input_range},
                            ckl::PackTile<strided_output(cb_out_id)>{output_range});
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
                // zero out values in cb_tilized_in input by multiplying with negative mask for the current group
                cb_in_negative_mask.wait_front(block_w);
                const ckl::StridedTileRange output_range{index_b_offset + index_g_offset, per_core_N};
                reconfig_data_format_srcb(cb_x_id, cb_in_negative_mask_id);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(block_h, block_w_curr),
                    ckl::BinaryFpu<
                        strided_block_input(cb_in_id),
                        ckl::input(
                            cb_in_negative_mask_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Row,
                            ckl::DataFormatReconfig::Disabled),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None>{output_range},
                    ckl::PackTile<strided_output(cb_in_id)>{output_range});

                // data in cb_x_id has valid data only for current group
                // cb_in_id has cleared data for that group
                // just add them together
                reconfig_data_format_srcb(cb_in_negative_mask_id, cb_x_id);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(block_h, block_w_curr),
                    ckl::BinaryFpu<
                        strided_block_input(cb_in_id),
                        strided_block_input(cb_x_id),
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::None>{output_range, ckl::StridedTileRange{0u, block_w}},
                    ckl::PackTile<strided_output(cb_in_id)>{output_range});
                cb_in_negative_mask.pop_front(block_w);
            }

            cb_x.pop_front(block_hw);

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
        cb_out.push_back(per_core_MN);
        cb_in.pop_front(per_core_MN);

    } else {
        // nothing, for the negative mask implementation, cb_in_id is the only cb in use, and it already has the data
        // required for the rest of kernel.
    }

    if constexpr (do_gamma) {
        if constexpr (use_negative_mask == false) {
            ckl::mul<
                ckl::input(
                    cb_out_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::AtEnd,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    cb_gamma_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_outgamma_id,
                    ckl::ReservePolicy::Upfront,
                    ckl::PushPolicy::AtEnd,
                    ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::grid(per_core_M, per_core_N));
            cb_outgamma.wait_front(per_core_MN);
        } else {
            ckl::mul<
                ckl::input(
                    cb_in_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    cb_gamma_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_in_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::grid(per_core_M, per_core_N));
        }
    }

    if constexpr (do_beta) {
        if constexpr (use_negative_mask == false) {
            ckl::add<
                ckl::input(
                    cb_inbeta_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::AtEnd,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    cb_beta_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_outbeta_id,
                    ckl::ReservePolicy::Upfront,
                    ckl::PushPolicy::AtEnd,
                    ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::grid(per_core_M, per_core_N));
            cb_outbeta.wait_front(per_core_MN);
        } else {
            ckl::add<
                ckl::input(
                    cb_in_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    cb_beta_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Row,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_in_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::grid(per_core_M, per_core_N));
        }
    }

#ifdef UNTILIZE_OUT
    // untilize - DEST capacity auto-detected
    ckl::untilize<
        per_core_N,
        cb_untilize_in_id,
        cb_untilize_out_id,
        ckl::untilize_config::InitUninitMode::InitAndUninit,
        ckl::untilize_config::WaitMode::WaitUpfront,
        ckl::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
}
