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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/groupnorm_constants.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // clang-format off
    // Definitions
    //   block_h: This the length of the row we wish to processes in terms of tiles
    //
    //   out_block_...: This is the length of our Circular Buffer, sometimes the length of out tensors(block_h) are larger than L1 space, so we
    //   have to process chunks of this data at a time
    //   this chunk is called an out_block
    //
    //   num_out_blocks: This is the number of chunks specified by the use, such that a DFBs (length defined by out_block) fit in L1
    //   (Users should minimize the number of num_out_blocks for better perf)
    //
    //   ...normal:  If num_out_blocks evenly divides block_h, then all chunks are the size normal
    //
    //   ...last: If num_out_blocks does not divides block_h, the leftovers are put into a chunk of length last
    //
    //   sender: This refers to a core that does aggregation calculations
    //   for the group of cores
    //
    //   receiver: This the cores that receive the aggregated results from sender, they only do
    //   local computations that they send to the sender for final aggregation
    //
    // This is a high level description of the stages of this kernel, tags will be added to show where in the code each
    // stage starts and ends
    //
    // Batch Loop:
    //   Group Loop:
    //     This is the process which repeats for every group
    //     Average Calc: E[x]
    //       Local Reduce:
    //           First we apply an input mask
    //           This is where we sum up our core's subtensor
    //           After summing up, we pass our scalar tile to dfb_ex_partial_id
    //           The reader kernels then aggregate all of the local scalars into a single tile
    //       Global Reduce:
    //           This single tile (dfb_ex_external_id) is a tile that contains each partial reduce from all the other cores
    //           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
    //           The reader core then sends this data out to all other cores as dfb_ex_global_id
    //
    //     Variance Calc: ∑(x-E[x])^2
    //     This follows the same pattern as the average calculation
    //       Local Reduce:
    //           First we subtract each value from our core's subtensor by the average value
    //           We next apply our input mask to zero our the values we wish to ignore
    //           Next we square our residuals to obtain the squared residuals
    //           After summing up, we pass our scalar tile to dfb_ex2_partial_id
    //           The reader kernels then aggregate all of the local scalars into a single tile
    //       Global Reduce:
    //           This single tile (dfb_ex_external_id) is a tile that contains each partial reduce from all the other cores
    //           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
    //           The reader core then sends this data out to all other cores as dfb_ex2_global_id
    //
    //     dfb_ex2pe_id Calculation:
    //       First we add dfb_ex2_global_id with dfb_eps_id
    //       Then we take the sqrt
    //       Lastly we take the reciprocal and he have the denominator of our calculation
    //     Final Val Calc:
    //       First we subtract each value from our core's subtensor by the average value
    //       We next apply our input mask to zero our the values we wish to ignore
    //       Next we multiply our residual with our denominator
    //       Optional Gamma:
    //           We multiply this value to gamma
    //       Optional Beta:
    //           We add beta to this value
    //
    // We are now done! Nice
    //   To look at where the code starts and stops search for
    //   Start LABEL or End Label
    //   Ex: Start Local Reduce or End Local Reduce
    // clang-format on
    constexpr uint32_t is_mcast_sender = get_named_compile_time_arg_val("is_mcast_sender");
    constexpr uint32_t do_gamma = get_named_compile_time_arg_val("do_gamma");
    constexpr uint32_t do_beta = get_named_compile_time_arg_val("do_beta");
    constexpr uint32_t num_cores_per_mcast_group = get_named_compile_time_arg_val("num_cores_per_mcast_group");
    // True when a reconfig-relevant operand is fp32: the per-group reconfig_data_format calls below
    // are then required. All-bf16 compiles them out (no-ops). See program factory.
    constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;

    constexpr uint32_t batch = get_named_compile_time_arg_val("batch");
    constexpr uint32_t group = get_named_compile_time_arg_val("group");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");
    constexpr uint32_t block_hw = get_named_compile_time_arg_val("block_hw");

    constexpr uint32_t subblock_w = get_named_compile_time_arg_val("subblock_w");
    constexpr uint32_t num_subblocks_w = get_named_compile_time_arg_val("num_subblocks_w");

    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    constexpr uint32_t per_core_MN = get_named_compile_time_arg_val("per_core_MN");

    constexpr uint32_t per_core_N_tile_bytes = get_named_compile_time_arg_val("per_core_N_tile_bytes");
    constexpr uint32_t num_groups_per_reset = get_named_compile_time_arg_val("num_groups_per_reset");

    constexpr uint32_t single_tile_size_bytes = get_named_compile_time_arg_val("single_tile_size_bytes");
    constexpr uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr uint32_t num_tiles_input_mask = get_named_compile_time_arg_val("num_tiles_input_mask");
    constexpr uint32_t num_cols_per_group = get_named_compile_time_arg_val("num_cols_per_group");

    constexpr uint32_t block_w_last = get_named_compile_time_arg_val("block_w_last");
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_named_compile_time_arg_val("GROUP_SIZE_IS_POWER_OF_2");
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W =
        get_named_compile_time_arg_val("GROUP_SIZE_SMALLER_THAN_TILE_W");
    constexpr uint32_t group_row_offset = get_named_compile_time_arg_val("group_row_offset");
    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    // Non-tile-aligned H*W (#50682), L = logical_hw, P = padded_hw, K = P/L - 1. The P - L padding
    // rows are reduced over as data; they hold zeros, so pass 1's sum is right and only the divisor
    // is wrong -- the writer rescales the scaler by sqrt(P/L). Pass 2 centers each padding row to
    // (0 - E[x]) and squares it, biasing the variance by exactly K*E[x]^2, subtracted below.
    // P == L compiles the whole path out. Cost: the subtraction cancels in bfloat16, so accuracy
    // degrades as K and E[x]^2/v grow (real shapes at K <= 0.6 stay in tolerance).
    constexpr uint32_t logical_hw = get_named_compile_time_arg_val("logical_hw");
    constexpr uint32_t padded_hw = get_named_compile_time_arg_val("padded_hw");
    constexpr bool has_pad_correction = padded_hw != logical_hw;

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // input cbs
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_in_id = tt::CBIndex::c_29;
    constexpr uint32_t dfb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_scaler_global_id = tt::CBIndex::c_4;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_28;

    // #50682 pad-correction DFBs, allocated only when has_pad_correction. dfb_k holds K from the
    // writer; dfb_msq / dfb_kmsq are single-tile scratch.
    constexpr uint32_t dfb_k_id = tt::CBIndex::c_1;
    constexpr uint32_t dfb_msq_id = tt::CBIndex::c_7;
    constexpr uint32_t dfb_kmsq_id = tt::CBIndex::c_11;

    // interm cbs
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t dfb_x_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_xmm_id = tt::CBIndex::c_25;
    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex2_partial_id = tt::CBIndex::c_21;
    constexpr uint32_t dfb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t dfb_ex2_id = tt::CBIndex::c_13;
    constexpr uint32_t dfb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t dfb_ex2_global_id = tt::CBIndex::c_14;
    constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_27;

    // interm cbs reuse
    constexpr uint32_t dfb_fusion_id = dfb_xmm_id;
    constexpr uint32_t dfb_reread_out_id = tt::CBIndex::c_23;
    constexpr uint32_t dfb_reread_write_out_id = tt::CBIndex::c_22;

    // output dfb_id
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t dfb_out_id = (do_gamma or do_beta) ? dfb_out0_id : dfb_reread_write_out_id;
#endif

    // tile offset
    uint32_t index_subblock_w_offset = 0;
    uint32_t index_w_offset = 0;
    uint32_t index_b_offset = 0;
    uint32_t index_g_offset = 0;
    uint32_t row_offset = num_cols_per_group;
    // data offset
    uint32_t num_datum_per_row_offeset = 0;
    // inplace out cbs
    bool copy_or_add = true;
    uint32_t group_reset_index = 0;
    uint32_t index_block_w = 0;
    bool apply_gamma_beta[block_w];
    constexpr uint32_t data_per_core_N_per_group = (per_core_N * tile_width / group);

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

    constexpr auto strided_col_input = [](uint32_t dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::OperandKind::Col,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileOffset::Strided);
    };
    constexpr auto offset_scalar_input = [](uint32_t dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::OperandKind::Scalar,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileOffset::Set);
    };
    constexpr auto strided_output = [](uint32_t dfb_id) {
        return ckl::output(
            dfb_id,
            ckl::ReservePolicy::None,
            ckl::PushPolicy::None,
            ckl::DataFormatReconfig::Disabled,
            ckl::PackRelu::Disabled,
            ckl::L1Accumulation::Disabled,
            ckl::DestAccumulation::Disabled,
            ckl::TileOffset::Strided);
    };

    DataflowBuffer dfb_beta(dfb_beta_id);
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_ex(dfb_ex_id);
    DataflowBuffer dfb_ex2(dfb_ex2_id);
    DataflowBuffer dfb_ex2_global(dfb_ex2_global_id);
    DataflowBuffer dfb_ex2_partial(dfb_ex2_partial_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_ex_external(dfb_ex_external_id);
    DataflowBuffer dfb_ex_global(dfb_ex_global_id);
    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_gamma(dfb_gamma_id);
    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_inbeta(dfb_inbeta_id);
    DataflowBuffer dfb_input_mask(dfb_input_mask_id);
    DataflowBuffer dfb_k(dfb_k_id);
    DataflowBuffer dfb_kmsq(dfb_kmsq_id);
    DataflowBuffer dfb_msq(dfb_msq_id);
    DataflowBuffer dfb_outbeta(dfb_outbeta_id);
    DataflowBuffer dfb_outgamma(dfb_outgamma_id);
    DataflowBuffer dfb_reread_out(dfb_reread_out_id);
    DataflowBuffer dfb_reread_write_out(dfb_reread_write_out_id);
    DataflowBuffer dfb_scaler(dfb_scaler_id);
    DataflowBuffer dfb_scaler_global(dfb_scaler_global_id);
    DataflowBuffer dfb_x(dfb_x_id);
    DataflowBuffer dfb_xmm(dfb_xmm_id);

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

    index_b_offset = 0;
    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    constexpr uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    constexpr uint32_t residual = block_h - (num_out_blocks * out_block_h_normal);
    constexpr bool extra_out_block = residual != 0;
    constexpr uint32_t num_out_blocks_padded =
        num_out_blocks + (extra_out_block ? (residual / out_block_h_normal + 1) : 0);
    constexpr uint32_t out_block_h_last = extra_out_block ? residual % out_block_h_normal : out_block_h_normal;
    constexpr uint32_t out_block_hw_last = out_block_h_last * block_w;
    constexpr uint32_t dfb_ex_external_bytes_required_id =
        num_out_blocks_padded * num_cores_per_mcast_group * dfb_ex_external_slot_pitch_bytes;
    constexpr uint32_t dfb_ex_external_tiles_required_id =
        (dfb_ex_external_bytes_required_id + single_tile_size_bytes - 1) / single_tile_size_bytes;

    // Start Batch Loop
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;

        row_offset = num_cols_per_group;
        copy_or_add = true;
        group_reset_index = 0;
        index_block_w = 0;

        // Start Group Loop
        for (uint32_t g = 0; g < group; ++g) {
            // Start Average Calc
            // Start Local Reduce
            dfb_input_mask.wait_front(block_w);
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual = out_block_h_normal;
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        out_block_h_actual = out_block_h_last;
                    }
                }
                dfb_in0.wait_front(out_block_hw_normal);

                reconfig_data_format_srcb(dfb_in0_id, dfb_input_mask_id);
                ckl::mul<
#ifdef TILIZE_IN
                    ckl::input(
                        dfb_in_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::AtEnd,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Disabled),
#else
                    ckl::input(
                        dfb_in0_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Disabled),
#endif
                    ckl::input(
                        dfb_input_mask_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Row,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_x_id,
                        ckl::ReservePolicy::Upfront,
                        ckl::PushPolicy::AtEnd,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(out_block_h_actual, block_w).block_size(subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
#ifndef TILIZE_IN
                        dfb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
#endif
                        dfb_x.reserve_back(out_block_hw_normal - out_block_hw_last);
                        dfb_x.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_scaler_id);

                // Partial/E[x]
                dfb_x.wait_front(out_block_hw_normal);
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_x_id,
                    dfb_scaler_id,
                    dfb_ex_partial_id,
                    ckl::ReduceInputPolicy::NoWaitNoPop,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::of(out_block_h_actual, block_w));
                dfb_x.pop_front(out_block_hw_normal);

                dfb_ex_partial.wait_front(1);
            }
            // End Local Redcue
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::col(dfb_ex_external_tiles_required_id));
                if constexpr (num_cores_per_mcast_group > 1) {
                    dfb_ex.reserve_back(1);
                    dfb_ex.push_back(1);
                }
            }
            // End Global Reduce
            // End Average Calc

            // Start Variance Calc
            // Start Local Reduce
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual = out_block_h_normal;
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        out_block_h_actual = out_block_h_last;
                    }
                }

                dfb_in0.wait_front(out_block_hw_normal);
                dfb_ex_global.wait_front(1);
                // fp32: reset both srcs so fp32 input/mean aren't read through the stale bf16 scaler format.
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_in0_id);
                    reconfig_data_format_srcb(dfb_ex_global_id);
                }
                ckl::sub<
                    ckl::input(
                        dfb_in0_id,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_ex_global_id,
                        ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_xmm_id,
                        ckl::ReservePolicy::Upfront,
                        ckl::PushPolicy::AtEnd,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(out_block_h_actual, block_w).block_size(subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        dfb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                reconfig_data_format_srcb(dfb_ex_global_id, dfb_input_mask_id);
                ckl::mul<
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_input_mask_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Row,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_x_id,
                        ckl::ReservePolicy::Upfront,
                        ckl::PushPolicy::AtEnd,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(out_block_h_actual, block_w).block_size(subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        dfb_xmm.pop_front(out_block_hw_normal - out_block_hw_last);
                        dfb_x.reserve_back(out_block_hw_normal - out_block_hw_last);
                        dfb_x.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                reconfig_data_format_srcb(dfb_input_mask_id, dfb_x_id);
                ckl::square<
                    ckl::input(
                        dfb_x_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_xmm_id,
                        ckl::ReservePolicy::Upfront,
                        ckl::PushPolicy::AtEnd,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(out_block_h_actual, block_w).block_size(subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        dfb_x.pop_front(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                // Partial-Var(x)
                dfb_xmm.wait_front(out_block_hw_normal);
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_xmm_id,
                    dfb_scaler_id,
                    dfb_ex2_partial_id,
                    ckl::ReduceInputPolicy::NoWaitNoPop,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::of(out_block_h_actual, block_w));
                dfb_xmm.pop_front(out_block_hw_normal);
            }
            // End Local Reduce
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex2_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::col(dfb_ex_external_tiles_required_id));
                if constexpr (num_cores_per_mcast_group > 1) {
                    dfb_ex2.reserve_back(1);
                    dfb_ex2.push_back(1);
                }
            }
            // End Global Reduce

            // Start Variance Calc
            //  global reduce results
            dfb_eps.wait_front(1);

            // Padded zero rows become -E[x] after centering and contribute K*E[x]^2 to the
            // variance. Preserve main's correction while using the migrated chain helpers for
            // the final add/rsqrt.
            constexpr uint32_t dfb_var_src_id = has_pad_correction ? dfb_msq_id : dfb_ex2_global_id;
            if constexpr (has_pad_correction) {
                dfb_ex_global.wait_front(1);
                dfb_ex2_global.wait_front(1);
                dfb_k.wait_front(1);

                dfb_msq.reserve_back(1);
                tile_regs_acquire();
                mul_init(dfb_ex_global_id, dfb_ex_global_id);
                mul_tiles(dfb_ex_global_id, dfb_ex_global_id, 0, 0, dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, dfb_msq_id);
                tile_regs_release();
                dfb_msq.push_back(1);

                dfb_msq.wait_front(1);
                dfb_kmsq.reserve_back(1);
                tile_regs_acquire();
                mul_init(dfb_msq_id, dfb_k_id);
                mul_tiles(dfb_msq_id, dfb_k_id, 0, 0, dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, dfb_kmsq_id);
                tile_regs_release();
                dfb_msq.pop_front(1);
                dfb_kmsq.push_back(1);

                dfb_kmsq.wait_front(1);
                dfb_msq.reserve_back(1);
                tile_regs_acquire();
                sub_init(dfb_ex2_global_id, dfb_kmsq_id);
                sub_tiles(dfb_ex2_global_id, dfb_kmsq_id, 0, 0, dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, dfb_msq_id);
                tile_regs_release();
                dfb_kmsq.pop_front(1);
                dfb_msq.push_back(1);
            }

            // fp32: reset both srcs so fp32 variance / bf16 eps aren't read through the stale square/reduce format.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_var_src_id);
                reconfig_data_format_srcb(dfb_eps_id);
            }
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ckl::input(
                        dfb_var_src_id,
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
            if constexpr (has_pad_correction) {
                dfb_ex2_global.pop_front(1);
            }
            // End Variance Calc

            bool start_copy_or_add = copy_or_add;
            uint32_t start_group_reset_index = group_reset_index;
            uint32_t start_index_block_w = index_block_w;

            uint32_t out_block_h_offset = 0;
            // Start Final Val Calc
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual = out_block_h_normal;
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        out_block_h_actual = out_block_h_last;
                    }
                }

                dfb_in0.wait_front(out_block_hw_normal);
                dfb_ex_global.wait_front(1);
                // fp32: reset both srcs so fp32 input/mean aren't read through the stale rsqrt/eps format.
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_in0_id);
                    reconfig_data_format_srcb(dfb_ex_global_id);
                }
                ckl::sub<
                    ckl::input(
                        dfb_in0_id,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_ex_global_id,
                        ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_xmm_id,
                        ckl::ReservePolicy::Upfront,
                        ckl::PushPolicy::AtEnd,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(out_block_h_actual, block_w).block_size(subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        dfb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                reconfig_data_format_srcb(dfb_ex_global_id, dfb_input_mask_id);
                ckl::mul<
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_input_mask_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Row,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_x_id,
                        ckl::ReservePolicy::Upfront,
                        ckl::PushPolicy::AtEnd,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(out_block_h_actual, block_w).block_size(subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        dfb_xmm.pop_front(out_block_hw_normal - out_block_hw_last);
                        dfb_x.reserve_back(out_block_hw_normal - out_block_hw_last);
                        dfb_x.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                dfb_ex2pe.wait_front(1);
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_x_id);
                // fp32: reset both srcs so fp32 x/rstd aren't read through the stale mask/eps format.
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_x_id);
                    reconfig_data_format_srcb(dfb_ex2pe_id);
                }
                ckl::mul<
                    ckl::input(
                        dfb_x_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::OperandKind::Block,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_ex2pe_id,
                        ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::output(
                        dfb_xmm_id,
                        ckl::ReservePolicy::Upfront,
                        ckl::PushPolicy::AtEnd,
                        ckl::DataFormatReconfig::Disabled)>(
                    ckl::IterationShape::grid(out_block_h_actual, block_w).block_size(subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        dfb_x.pop_front(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        dfb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }
                dfb_xmm.wait_front(out_block_hw_normal);

                copy_or_add = start_copy_or_add;
                group_reset_index = start_group_reset_index;
                index_block_w = start_index_block_w;

                // add or copy with previous output results
                uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

                dfb_reread_out.wait_front(out_block_hw_normal);
                dfb_reread_write_out.reserve_back(out_block_hw_normal);
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    const ckl::StridedTileRange input_range{w, block_w};
                    const ckl::StridedTileRange output_range{w, block_w_curr};
                    if (copy_or_add) {
                        ckl::eltwise_chain(
                            ckl::IterationShape::col(out_block_h_actual),
                            ckl::CopyTile<strided_col_input(dfb_xmm_id)>{input_range},
                            ckl::PackTile<strided_output(dfb_reread_write_out_id)>{output_range});
                    } else {
                        ckl::eltwise_chain(
                            ckl::IterationShape::col(out_block_h_actual),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Add,
                                strided_col_input(dfb_reread_out_id),
                                strided_col_input(dfb_xmm_id)>{output_range, input_range},
                            ckl::PackTile<strided_output(dfb_reread_write_out_id)>{output_range});
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

                    bool is_past_end_of_group =
                        (((w + index_g_offset) + 1) * tile_width) > ((g + 1) * data_per_core_N_per_group);
                    apply_gamma_beta[w] = !is_past_end_of_group;
                }
                dfb_xmm.pop_front(out_block_hw_normal);
                dfb_reread_out.pop_front(out_block_hw_normal);
                dfb_reread_write_out.push_back(out_block_hw_normal);

                // Start Optional Gamma:
                if constexpr (do_gamma) {
                    dfb_outgamma.reserve_back(out_block_hw_normal);
                    dfb_gamma.wait_front(per_core_N);
                    dfb_reread_write_out.wait_front(out_block_hw_normal);
                    for (uint32_t j = 0; j < block_w_curr; ++j) {
                        if (apply_gamma_beta[j]) {
                            // fp32: reset both srcs so bf16 gamma isn't read through the reread stage's fp32 format.
                            if constexpr (enable_fp32_reconfig) {
                                reconfig_data_format_srca(dfb_reread_write_out_id);
                                reconfig_data_format_srcb(dfb_gamma_id);
                            }
                            ckl::eltwise_chain(
                                ckl::IterationShape::col(out_block_h_actual),
                                ckl::BinaryFpu<
                                    ckl::BinaryFpuOp::Mul,
                                    strided_col_input(dfb_reread_write_out_id),
                                    ckl::input(offset_scalar_input(dfb_gamma_id), ckl::BroadcastDim::Row)>{
                                    ckl::StridedTileRange{j, block_w_curr}, j + index_g_offset},
                                ckl::PackTile<strided_output(dfb_outgamma_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        } else {
                            ckl::eltwise_chain(
                                ckl::IterationShape::col(out_block_h_actual),
                                ckl::CopyTile<strided_col_input(dfb_reread_write_out_id)>{
                                    ckl::StridedTileRange{j, block_w_curr}},
                                ckl::PackTile<strided_output(dfb_outgamma_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        }
                    }
                    dfb_outgamma.push_back(out_block_hw_normal);
                    dfb_reread_write_out.pop_front(out_block_hw_normal);
                    dfb_outgamma.wait_front(out_block_hw_normal);
                }
                // End Optional Gamma
                //
                // Start Optional Beta
                if constexpr (do_beta) {
                    dfb_outbeta.reserve_back(out_block_hw_normal);
                    dfb_beta.wait_front(per_core_N);
                    for (uint32_t j = 0; j < block_w_curr; ++j) {
                        if (apply_gamma_beta[j]) {
                            // fp32: reset both srcs so bf16 beta isn't read through the fp32 dfb_inbeta format.
                            if constexpr (enable_fp32_reconfig) {
                                reconfig_data_format_srca(dfb_inbeta_id);
                                reconfig_data_format_srcb(dfb_beta_id);
                            }
                            ckl::eltwise_chain(
                                ckl::IterationShape::col(out_block_h_actual),
                                ckl::BinaryFpu<
                                    ckl::BinaryFpuOp::Add,
                                    strided_col_input(dfb_inbeta_id),
                                    ckl::input(offset_scalar_input(dfb_beta_id), ckl::BroadcastDim::Row)>{
                                    ckl::StridedTileRange{j, block_w_curr}, j + index_g_offset},
                                ckl::PackTile<strided_output(dfb_outbeta_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        } else {
                            ckl::eltwise_chain(
                                ckl::IterationShape::col(out_block_h_actual),
                                ckl::CopyTile<strided_col_input(dfb_inbeta_id)>{ckl::StridedTileRange{j, block_w_curr}},
                                ckl::PackTile<strided_output(dfb_outbeta_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        }
                    }
                    dfb_outbeta.push_back(out_block_hw_normal);
                    dfb_inbeta.pop_front(out_block_hw_normal);
                    dfb_outbeta.wait_front(out_block_hw_normal);
                }
                // End Optional Beta

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
            // End Final Val Calc
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
            dfb_ex_global.pop_front(1);
            dfb_ex2pe.pop_front(1);
            dfb_input_mask.pop_front(block_w);
        }
        // End Group Loop
        index_b_offset += num_tiles_per_batch;
    }
    // End Batch Loop
}
