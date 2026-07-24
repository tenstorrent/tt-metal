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
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/groupnorm_constants.hpp"

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
    //   num_out_blocks: This is the number of chunks specified by the use, such that a CBs (length defined by out_block) fit in L1
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
    //           After summing up, we pass our scalar tile to cb_ex_partial_id
    //           The reader kernels then aggregate all of the local scalars into a single tile
    //       Global Reduce:
    //           This single tile (cb_ex_external_id) is a tile that contains each partial reduce from all the other cores
    //           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
    //           The reader core then sends this data out to all other cores as cb_ex_global_id
    //
    //     Variance Calc: ∑(x-E[x])^2
    //     This follows the same pattern as the average calculation
    //       Local Reduce:
    //           First we subtract each value from our core's subtensor by the average value
    //           We next apply our input mask to zero our the values we wish to ignore
    //           Next we square our residuals to obtain the squared residuals
    //           After summing up, we pass our scalar tile to cb_ex2_partial_id
    //           The reader kernels then aggregate all of the local scalars into a single tile
    //       Global Reduce:
    //           This single tile (cb_ex_external_id) is a tile that contains each partial reduce from all the other cores
    //           Only the core designated as the sender reduces this tile to produce the global scalar reduce value.
    //           The reader core then sends this data out to all other cores as cb_ex2_global_id
    //
    //     cb_ex2pe_id Calculation:
    //       First we add cb_ex2_global_id with cb_eps_id
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

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // input cbs
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_in_id = tt::CBIndex::c_29;
    constexpr uint32_t cb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler_global_id = tt::CBIndex::c_4;
    constexpr uint32_t cb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask_id = tt::CBIndex::c_28;

    // interm cbs
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t cb_x_id = tt::CBIndex::c_24;
    constexpr uint32_t cb_xmm_id = tt::CBIndex::c_25;
    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex2_partial_id = tt::CBIndex::c_21;
    constexpr uint32_t cb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t cb_ex2_id = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2_global_id = tt::CBIndex::c_14;
    constexpr uint32_t cb_ex2pe_id = tt::CBIndex::c_27;

    // interm cbs reuse
    constexpr uint32_t cb_fusion_id = cb_xmm_id;
    constexpr uint32_t cb_reread_out_id = tt::CBIndex::c_23;
    constexpr uint32_t cb_reread_write_out_id = tt::CBIndex::c_22;

    // output cb
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t cb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out_id = (do_gamma or do_beta) ? cb_out0_id : cb_reread_write_out_id;
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
    constexpr int cb_outgamma_id = cb_in_id;
    constexpr int cb_inbeta_id = do_gamma ? cb_outgamma_id : cb_reread_write_out_id;
    constexpr int cb_outbeta_id = do_gamma ? cb_out_id : cb_in_id;
    constexpr int cb_untilize_in_id = (do_gamma and not do_beta) ? cb_outgamma_id
                                      : do_beta                  ? cb_outbeta_id
                                                                 : cb_reread_write_out_id;
    constexpr int cb_untilize_out_id =
#ifdef READER_REPACK
        cb_repack_out_id;
#else
        cb_out0_id;
#endif
#else
    constexpr int cb_outgamma_id = do_beta ? cb_in_id : cb_out0_id;
    constexpr int cb_inbeta_id = do_gamma ? cb_outgamma_id : cb_reread_write_out_id;
    constexpr int cb_outbeta_id = cb_out0_id;
#endif

    constexpr auto strided_col_input = [](uint32_t cb) {
        return ckl::input(
            cb,
            ckl::InputLifecycle::CallerManaged,
            ckl::OperandKind::Col,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileOffset::Strided);
    };
    constexpr auto offset_scalar_input = [](uint32_t cb) {
        return ckl::input(
            cb,
            ckl::InputLifecycle::CallerManaged,
            ckl::OperandKind::Scalar,
            ckl::DataFormatReconfig::Disabled,
            ckl::TileOffset::Set);
    };
    constexpr auto strided_output = [](uint32_t cb) {
        return ckl::output(
            cb,
            ckl::OutputLifecycle::CallerManaged,
            ckl::DataFormatReconfig::Disabled,
            ckl::PackRelu::Disabled,
            ckl::L1Accumulation::Disabled,
            ckl::DestAccumulation::Disabled,
            ckl::TileOffset::Strided);
    };

    CircularBuffer cb_beta(cb_beta_id);
    CircularBuffer cb_eps(cb_eps_id);
    CircularBuffer cb_ex(cb_ex_id);
    CircularBuffer cb_ex2(cb_ex2_id);
    CircularBuffer cb_ex2_global(cb_ex2_global_id);
    CircularBuffer cb_ex2_partial(cb_ex2_partial_id);
    CircularBuffer cb_ex2pe(cb_ex2pe_id);
    CircularBuffer cb_ex_external(cb_ex_external_id);
    CircularBuffer cb_ex_global(cb_ex_global_id);
    CircularBuffer cb_ex_partial(cb_ex_partial_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_inbeta(cb_inbeta_id);
    CircularBuffer cb_input_mask(cb_input_mask_id);
    CircularBuffer cb_outbeta(cb_outbeta_id);
    CircularBuffer cb_outgamma(cb_outgamma_id);
    CircularBuffer cb_reread_out(cb_reread_out_id);
    CircularBuffer cb_reread_write_out(cb_reread_write_out_id);
    CircularBuffer cb_scaler(cb_scaler_id);
    CircularBuffer cb_scaler_global(cb_scaler_global_id);
    CircularBuffer cb_x(cb_x_id);
    CircularBuffer cb_xmm(cb_xmm_id);

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
    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    constexpr uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    constexpr uint32_t residual = block_h - (num_out_blocks * out_block_h_normal);
    constexpr bool extra_out_block = residual != 0;
    constexpr uint32_t num_out_blocks_padded =
        num_out_blocks + (extra_out_block ? (residual / out_block_h_normal + 1) : 0);
    constexpr uint32_t out_block_h_last = extra_out_block ? residual % out_block_h_normal : out_block_h_normal;
    constexpr uint32_t out_block_hw_last = out_block_h_last * block_w;
    constexpr uint32_t cb_ex_external_bytes_required =
        num_out_blocks_padded * num_cores_per_mcast_group * cb_ex_external_slot_pitch_bytes;
    constexpr uint32_t cb_ex_external_tiles_required =
        (cb_ex_external_bytes_required + single_tile_size_bytes - 1) / single_tile_size_bytes;

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
            cb_input_mask.wait_front(block_w);
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual = out_block_h_normal;
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        out_block_h_actual = out_block_h_last;
                    }
                }
                cb_in0.wait_front(out_block_hw_normal);

                reconfig_data_format_srcb(cb_in0_id, cb_input_mask_id);
                ckl::mul<
#ifdef TILIZE_IN
                    ckl::input(cb_in_id, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block),
#else
                    ckl::input(cb_in0_id, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
#endif
                    ckl::input(cb_input_mask_id, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Row),
                    ckl::output(cb_x_id, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::grid(out_block_h_actual, block_w, subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
#ifndef TILIZE_IN
                        cb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
#endif
                        cb_x.reserve_back(out_block_hw_normal - out_block_hw_last);
                        cb_x.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }
                reconfig_data_format_srcb(cb_input_mask_id, cb_scaler_id);

                // Partial/E[x]
                cb_x.wait_front(out_block_hw_normal);
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    cb_x_id,
                    cb_scaler_id,
                    cb_ex_partial_id,
                    ckl::ReduceInputPolicy::NoWaitNoPop,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::of(out_block_h_actual, block_w));
                cb_x.pop_front(out_block_hw_normal);

                cb_ex_partial.wait_front(1);
            }
            // End Local Redcue
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    cb_ex_external_id,
                    cb_scaler_global_id,
                    cb_ex_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::col(cb_ex_external_tiles_required));
                if constexpr (num_cores_per_mcast_group > 1) {
                    cb_ex.reserve_back(1);
                    cb_ex.push_back(1);
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

                cb_in0.wait_front(out_block_hw_normal);
                cb_ex_global.wait_front(1);
                ckl::sub<
                    ckl::input(cb_in0_id),
                    ckl::input(cb_ex_global_id, ckl::InputLifecycle::CallerManaged),
                    ckl::output(cb_xmm_id, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::grid(out_block_h_actual, block_w, subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        cb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                reconfig_data_format_srcb(cb_ex_global_id, cb_input_mask_id);
                ckl::mul<
                    ckl::input(cb_xmm_id, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(cb_input_mask_id, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Row),
                    ckl::output(cb_x_id, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::grid(out_block_h_actual, block_w, subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        cb_xmm.pop_front(out_block_hw_normal - out_block_hw_last);
                        cb_x.reserve_back(out_block_hw_normal - out_block_hw_last);
                        cb_x.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                reconfig_data_format_srcb(cb_input_mask_id, cb_x_id);
                ckl::square<
                    ckl::input(cb_x_id, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::output(cb_xmm_id, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>(
                    ckl::EltwiseShape::grid(out_block_h_actual, block_w, subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        cb_x.pop_front(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                // Partial-Var(x)
                cb_xmm.wait_front(out_block_hw_normal);
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    cb_xmm_id,
                    cb_scaler_id,
                    cb_ex2_partial_id,
                    ckl::ReduceInputPolicy::NoWaitNoPop,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::of(out_block_h_actual, block_w));
                cb_xmm.pop_front(out_block_hw_normal);
            }
            // End Local Reduce
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                ckl::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    cb_ex_external_id,
                    cb_scaler_global_id,
                    cb_ex2_global_id,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::NONE>(
                    ckl::ReduceInputBlockShape::col(cb_ex_external_tiles_required));
                if constexpr (num_cores_per_mcast_group > 1) {
                    cb_ex2.reserve_back(1);
                    cb_ex2.push_back(1);
                }
            }
            // End Global Reduce

            // Start Variance Calc
            //  global reduce results
            cb_eps.wait_front(1);
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::BinaryFpu<
                    ckl::input(cb_ex2_global_id),
                    ckl::input(cb_eps_id, ckl::InputLifecycle::CallerManaged),
                    ckl::BinaryFpuOp::Add,
                    ckl::BroadcastDim::None>{},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(
                    cb_ex2pe_id, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
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

                cb_in0.wait_front(out_block_hw_normal);
                cb_ex_global.wait_front(1);
                ckl::sub<
                    ckl::input(cb_in0_id),
                    ckl::input(cb_ex_global_id, ckl::InputLifecycle::CallerManaged),
                    ckl::output(cb_xmm_id, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::grid(out_block_h_actual, block_w, subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        cb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                reconfig_data_format_srcb(cb_ex_global_id, cb_input_mask_id);
                ckl::mul<
                    ckl::input(cb_xmm_id, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(cb_input_mask_id, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Row),
                    ckl::output(cb_x_id, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::grid(out_block_h_actual, block_w, subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        cb_xmm.pop_front(out_block_hw_normal - out_block_hw_last);
                        cb_x.reserve_back(out_block_hw_normal - out_block_hw_last);
                        cb_x.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }

                cb_ex2pe.wait_front(1);
                ckl::mul<
                    ckl::input(cb_x_id, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(cb_ex2pe_id, ckl::InputLifecycle::CallerManaged),
                    ckl::output(cb_xmm_id, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::grid(out_block_h_actual, block_w, subblock_w));
                if constexpr (extra_out_block) {
                    if (out_block_index == (num_out_blocks_padded - 1)) {
                        cb_x.pop_front(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.reserve_back(out_block_hw_normal - out_block_hw_last);
                        cb_xmm.push_back(out_block_hw_normal - out_block_hw_last);
                    }
                }
                cb_xmm.wait_front(out_block_hw_normal);

                copy_or_add = start_copy_or_add;
                group_reset_index = start_group_reset_index;
                index_block_w = start_index_block_w;

                // add or copy with previous output results
                uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

                cb_reread_out.wait_front(out_block_hw_normal);
                cb_reread_write_out.reserve_back(out_block_hw_normal);
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    const ckl::StridedTileRange input_range{w, block_w};
                    const ckl::StridedTileRange output_range{w, block_w_curr};
                    if (copy_or_add) {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::col(out_block_h_actual),
                            ckl::CopyTile<strided_col_input(cb_xmm_id)>{input_range},
                            ckl::PackTile<strided_output(cb_reread_write_out_id)>{output_range});
                    } else {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::col(out_block_h_actual),
                            ckl::BinaryFpu<
                                strided_col_input(cb_reread_out_id),
                                strided_col_input(cb_xmm_id),
                                ckl::BinaryFpuOp::Add,
                                ckl::BroadcastDim::None>{output_range, input_range},
                            ckl::PackTile<strided_output(cb_reread_write_out_id)>{output_range});
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
                cb_xmm.pop_front(out_block_hw_normal);
                cb_reread_out.pop_front(out_block_hw_normal);
                cb_reread_write_out.push_back(out_block_hw_normal);

                // Start Optional Gamma:
                if constexpr (do_gamma) {
                    cb_outgamma.reserve_back(out_block_hw_normal);
                    cb_gamma.wait_front(per_core_N);
                    cb_reread_write_out.wait_front(out_block_hw_normal);
                    for (uint32_t j = 0; j < block_w_curr; ++j) {
                        if (apply_gamma_beta[j]) {
                            ckl::eltwise_chain(
                                ckl::EltwiseShape::col(out_block_h_actual),
                                ckl::BinaryFpu<
                                    strided_col_input(cb_reread_write_out_id),
                                    offset_scalar_input(cb_gamma_id),
                                    ckl::BinaryFpuOp::Mul,
                                    ckl::BroadcastDim::Row>{ckl::StridedTileRange{j, block_w_curr}, j + index_g_offset},
                                ckl::PackTile<strided_output(cb_outgamma_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        } else {
                            ckl::eltwise_chain(
                                ckl::EltwiseShape::col(out_block_h_actual),
                                ckl::CopyTile<strided_col_input(cb_reread_write_out_id)>{
                                    ckl::StridedTileRange{j, block_w_curr}},
                                ckl::PackTile<strided_output(cb_outgamma_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        }
                    }
                    cb_outgamma.push_back(out_block_hw_normal);
                    cb_reread_write_out.pop_front(out_block_hw_normal);
                    cb_outgamma.wait_front(out_block_hw_normal);
                }
                // End Optional Gamma
                //
                // Start Optional Beta
                if constexpr (do_beta) {
                    cb_outbeta.reserve_back(out_block_hw_normal);
                    cb_beta.wait_front(per_core_N);
                    for (uint32_t j = 0; j < block_w_curr; ++j) {
                        if (apply_gamma_beta[j]) {
                            ckl::eltwise_chain(
                                ckl::EltwiseShape::col(out_block_h_actual),
                                ckl::BinaryFpu<
                                    strided_col_input(cb_inbeta_id),
                                    offset_scalar_input(cb_beta_id),
                                    ckl::BinaryFpuOp::Add,
                                    ckl::BroadcastDim::Row>{ckl::StridedTileRange{j, block_w_curr}, j + index_g_offset},
                                ckl::PackTile<strided_output(cb_outbeta_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        } else {
                            ckl::eltwise_chain(
                                ckl::EltwiseShape::col(out_block_h_actual),
                                ckl::CopyTile<strided_col_input(cb_inbeta_id)>{ckl::StridedTileRange{j, block_w_curr}},
                                ckl::PackTile<strided_output(cb_outbeta_id)>{ckl::StridedTileRange{j, block_w_curr}});
                        }
                    }
                    cb_outbeta.push_back(out_block_hw_normal);
                    cb_inbeta.pop_front(out_block_hw_normal);
                    cb_outbeta.wait_front(out_block_hw_normal);
                }
                // End Optional Beta

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
            cb_ex_global.pop_front(1);
            cb_ex2pe.pop_front(1);
            cb_input_mask.pop_front(block_w);
        }
        // End Group Loop
        index_b_offset += num_tiles_per_batch;
    }
    // End Batch Loop
}
