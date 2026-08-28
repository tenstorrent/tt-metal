// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/groupnorm_constants.hpp"
#include "api/dataflow/dataflow_buffer.h"

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
    // True when a reconfig-relevant operand is fp32: the per-group reconfig_data_format calls below
    // are then required. All-bf16 compiles them out (no-ops). See program factory.
    constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;

    constexpr uint32_t batch = get_named_compile_time_arg_val("batch");
    constexpr uint32_t group = get_named_compile_time_arg_val("group");

    // fp32 bits of the statistics divisors, applied once on DST after each SUM reduce. The old
    // route -- bf16(1/sqrt(N)) in the scaler tile, applied twice by REDUCE_SCALAR -- made the
    // effective divisor bf16(1/sqrt(N))^2, inexact unless sqrt(N) is a power of two (#53846).
    // mean_recip_bits is pad-corrected on host when H*W is not tile-aligned.
    constexpr uint32_t mean_recip_bits = get_named_compile_time_arg_val("mean_recip_bits");
    constexpr uint32_t global_recip_bits = get_named_compile_time_arg_val("global_recip_bits");

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

    // Non-tile-aligned H*W: the tile-padding rows are excluded from both accumulation passes by
    // switching to a second, row-masked set of mask tiles on the batch's final row-tile. The writer
    // gives cores that do not hold that row-tile a copy of the normal set, so the switch here is
    // unconditional. The divisor is corrected separately, in the reduce scaler.
    // logical_hw / padded_hw are carried only so two shapes padding to the same size cannot share a
    // cached program; has_row_mask is what this kernel branches on.
    constexpr uint32_t logical_hw [[maybe_unused]] = get_named_compile_time_arg_val("logical_hw");
    constexpr uint32_t padded_hw [[maybe_unused]] = get_named_compile_time_arg_val("padded_hw");
    constexpr bool has_row_mask = get_named_compile_time_arg_val("has_row_mask") == 1;
    constexpr uint32_t mask_tiles_per_group = has_row_mask ? 2 * block_w : block_w;
    constexpr uint32_t last_row_tile = block_h - 1;

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // input cbs
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_in_id = tt::CBIndex::c_29;
#ifdef TILIZE_IN
    // Holds the whole per-core group, tilized once and kept in L1 for all three passes.
    constexpr uint32_t dfb_in_resident_id = tt::CBIndex::c_17;
#endif
    constexpr uint32_t dfb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_scaler_global_id = tt::CBIndex::c_4;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_28;

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
#ifdef UNTILIZE_OUT
    // Scratch for the row-major output reread; tilized into c_23 below.
    constexpr uint32_t dfb_reread_rm_id = tt::CBIndex::c_20;
#endif

    // output cb
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t dfb_out_id = (do_gamma or do_beta) ? dfb_out0_id : dfb_reread_write_out_id;
#endif

    // tile offset
    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
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

    constexpr int dfb_outgamma_id = do_beta ? dfb_in_id : dfb_out0_id;
    constexpr int dfb_inbeta_id = do_gamma ? dfb_outgamma_id : dfb_reread_write_out_id;
    constexpr int dfb_outbeta_id = dfb_out0_id;
#ifdef UNTILIZE_OUT
    // Untilize the tiled result into the row-major output c_30.
    constexpr int dfb_untilize_in_id = (do_gamma or do_beta) ? dfb_out0_id : dfb_reread_write_out_id;
    constexpr int dfb_untilize_out_id =
#ifdef READER_REPACK
        dfb_repack_out_id;
#else
        dfb_out_id;
#endif
#endif

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
#ifdef TILIZE_IN
    DataflowBuffer dfb_in_resident(dfb_in_resident_id);
#endif
    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_inbeta(dfb_inbeta_id);
    DataflowBuffer dfb_input_mask(dfb_input_mask_id);
    DataflowBuffer dfb_outbeta(dfb_outbeta_id);
    DataflowBuffer dfb_outgamma(dfb_outgamma_id);
    DataflowBuffer dfb_reread_out(dfb_reread_out_id);
    DataflowBuffer dfb_reread_write_out(dfb_reread_write_out_id);
    DataflowBuffer dfb_scaler(dfb_scaler_id);
    DataflowBuffer dfb_scaler_global(dfb_scaler_global_id);
    DataflowBuffer dfb_x(dfb_x_id);
    DataflowBuffer dfb_xmm(dfb_xmm_id);

#ifdef TILIZE_IN
#ifdef READER_REPACK
    constexpr uint32_t dfb_in_rm_id = dfb_repack_id;
#else
    constexpr uint32_t dfb_in_rm_id = dfb_in0_id;
#endif
    // Tilize the whole group once and reuse it for all three passes.
    compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in_resident_id);
    constexpr uint32_t dfb_input_id = dfb_in_resident_id;
#else
    // Already tiled, so feed compute directly.
    compute_kernel_hw_startup(dfb_in0_id, dfb_input_mask_id, dfb_x_id);
    constexpr uint32_t dfb_input_id = dfb_in0_id;
#endif

    index_b_offset = 0;
    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    uint32_t out_block_hw_last = out_block_hw_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        uint32_t residual = block_h - (num_out_blocks * out_block_h_normal);
        num_out_blocks_padded += (residual / out_block_h_normal + 1);
        out_block_h_last = residual % out_block_h_normal;
        out_block_hw_last = out_block_h_last * block_w;
    }
    uint32_t dfb_ex_external_tiles_required =
        num_out_blocks_padded * num_cores_per_mcast_group * dfb_ex_external_slot_pitch_bytes / single_tile_size_bytes;
    if ((num_out_blocks_padded * num_cores_per_mcast_group * dfb_ex_external_slot_pitch_bytes) %
        single_tile_size_bytes) {
        dfb_ex_external_tiles_required++;
    }

    // Start Batch Loop
    // Post-reduce hooks: run on DST after the reduce math, before pack (same pattern as
    // dit_rmsnorm_fused_compute.cpp). One fp32 multiply replaces the doubly-applied bf16 scaler.
    auto scale_by_mean_recip = [](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, mean_recip_bits);
    };
    auto scale_by_global_recip = [](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, global_recip_bits);
    };

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
            dfb_input_mask.wait_front(mask_tiles_per_group);
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                }
#ifdef TILIZE_IN
                // Append this out-block; no pop, so the whole group stays available.
                compute_kernel_lib::tilize<
                    block_w,
                    dfb_in_rm_id,
                    dfb_in_resident_id,
                    compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                    compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                    compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                    out_block_h_normal);
                dfb_in_resident.wait_front((out_block_index + 1) * out_block_hw_normal);
                uint32_t out_block_base = out_block_index * out_block_hw_normal;
#else
                dfb_in0.wait_front(out_block_hw_normal);
                constexpr uint32_t out_block_base = 0;
#endif

                index_h_offset = 0;
                // Row-tile index within the batch; derived from out_block_index because the final
                // out-block can be empty when num_out_blocks does not divide block_h.
                uint32_t row_tile_base = out_block_index * out_block_h_normal;
                reconfig_data_format_srcb(dfb_in0_id, dfb_input_mask_id);
                // mask input
                // The row-masked set varies down the rows of a tile, so it can only be consumed by
                // a full-tile multiply. Row-0-only synthesis and the row broadcast are therefore
                // available exactly when there is no row mask, i.e. on tile-aligned H*W.
                if constexpr (has_row_mask) {
                    mul_init(dfb_input_id, dfb_input_mask_id);
                } else {
                    mul_bcast_rows_init(dfb_input_id, dfb_input_mask_id);
                }
                dfb_x.reserve_back(out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                    // Row-masked set on the batch's final row-tile, so the padding contributes
                    // nothing to E[x]. if constexpr keeps tile-aligned codegen unchanged.
                    uint32_t mask_set_offset = 0;
                    if constexpr (has_row_mask) {
                        mask_set_offset = ((row_tile_base + i) == last_row_tile) ? block_w : 0;
                    }
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; ++w) {
                            uint32_t index = w + index_subblock_w_offset + index_h_offset + out_block_base;
                            if constexpr (has_row_mask) {
                                mul_tiles(
                                    dfb_input_id,
                                    dfb_input_mask_id,
                                    index,
                                    w + index_subblock_w_offset + mask_set_offset,
                                    w);
                            } else {
                                mul_tiles_bcast_rows(
                                    dfb_input_id, dfb_input_mask_id, index, w + index_subblock_w_offset, w);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; ++i) {
                            pack_tile(i, dfb_x_id);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
                // Only the tiled path pops here; the row-major group stays resident.
#ifndef TILIZE_IN
                dfb_in0.pop_front(out_block_hw_normal);
#endif
                dfb_x.push_back(out_block_hw_normal);
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_scaler_id);

                // Partial/E[x]
                dfb_x.wait_front(out_block_hw_normal);
                compute_kernel_lib::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_x_id,
                    dfb_scaler_id,
                    dfb_ex_partial_id,
                    compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                    compute_kernel_lib::ReduceInputBlockShape::of(out_block_h_actual, block_w),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::NoAccumulation{},
                    scale_by_mean_recip);
                dfb_x.pop_front(out_block_hw_normal);

                dfb_ex_partial.wait_front(1);
            }
            // End Local Redcue
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                compute_kernel_lib::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex_global_id,
                    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                    compute_kernel_lib::ReduceInputBlockShape::col(dfb_ex_external_tiles_required),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::NoAccumulation{},
                    scale_by_global_recip);
                if (num_cores_per_mcast_group > 1) {
                    dfb_ex.reserve_back(1);
                    dfb_ex.push_back(1);
                }
            }
            // End Global Reduce
            // End Average Calc

            // Start Variance Calc
            // Start Local Reduce
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                }
                uint32_t row_tile_base = out_block_index * out_block_h_normal;

                // The resident group is already there; only the tiled path waits on new rows.
#ifndef TILIZE_IN
                dfb_in0.wait_front(out_block_hw_normal);
#endif
                // x - E[x]
                // fp32: reset both srcs so fp32 input/mean aren't read through the stale bf16 scaler format.
                // The reconfig has to precede the init: the init's LLK assert checks that the unpack config
                // registers already describe these operands. (The MOP is built from the init's static
                // arguments; the registers themselves are consumed later, by UNPACR.)
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_input_id);
                    reconfig_data_format_srcb(dfb_ex_global_id);
                }
                sub_bcast_scalar_init(dfb_input_id, dfb_ex_global_id);

                dfb_xmm.reserve_back(out_block_hw_normal);
                dfb_ex_global.wait_front(1);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
#ifdef TILIZE_IN
                    uint32_t row_base = out_block_index * out_block_hw_normal + i * block_w;
#else
                    constexpr uint32_t row_base = 0;
#endif
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset + row_base;
                            sub_tiles_bcast_scalar(dfb_input_id, dfb_ex_global_id, index, 0, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, dfb_xmm_id);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
#ifndef TILIZE_IN
                    dfb_in0.pop_front(block_w);
#endif
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
#ifndef TILIZE_IN
                    dfb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
#endif
                }
                dfb_xmm.push_back(out_block_hw_normal);

                // zero out the garbage values by mult mask again
                reconfig_data_format_srcb(dfb_ex_global_id, dfb_input_mask_id);
                if constexpr (has_row_mask) {
                    mul_init(dfb_xmm_id, dfb_input_mask_id);
                } else {
                    mul_bcast_rows_init(dfb_xmm_id, dfb_input_mask_id);
                }
                dfb_x.reserve_back(out_block_hw_normal);
                dfb_xmm.wait_front(out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    // Same switch as pass 1; otherwise each padding row is centered to
                    // (garbage - E[x]) and squared into the variance.
                    uint32_t mask_set_offset = 0;
                    if constexpr (has_row_mask) {
                        mask_set_offset = ((row_tile_base + i) == last_row_tile) ? block_w : 0;
                    }
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; ++w) {
                            uint32_t index = w + index_subblock_w_offset;
                            if constexpr (has_row_mask) {
                                mul_tiles(dfb_xmm_id, dfb_input_mask_id, index, index + mask_set_offset, w);
                            } else {
                                mul_tiles_bcast_rows(dfb_xmm_id, dfb_input_mask_id, index, index, w);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; ++i) {
                            pack_tile(i, dfb_x_id);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    dfb_xmm.pop_front(block_w);
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    dfb_xmm.pop_front(out_block_hw_normal - out_block_hw_last);
                }
                dfb_x.push_back(out_block_hw_normal);

                reconfig_data_format_srcb(dfb_input_mask_id, dfb_x_id);
                // (x - E[x])^2
                index_h_offset = 0;
                mul_init(dfb_x_id, dfb_x_id);
                dfb_xmm.reserve_back(out_block_hw_normal);
                dfb_x.wait_front(out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset + index_h_offset;
                            mul_tiles(dfb_x_id, dfb_x_id, index, index, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, dfb_xmm_id);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
                dfb_x.pop_front(out_block_hw_normal);
                dfb_xmm.push_back(out_block_hw_normal);

                // Partial-Var(x)
                dfb_xmm.wait_front(out_block_hw_normal);
                compute_kernel_lib::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_xmm_id,
                    dfb_scaler_id,
                    dfb_ex2_partial_id,
                    compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                    compute_kernel_lib::ReduceInputBlockShape::of(out_block_h_actual, block_w),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::NoAccumulation{},
                    scale_by_mean_recip);
                dfb_xmm.pop_front(out_block_hw_normal);
            }
            // End Local Reduce
            // Start Global Reduce
            if constexpr (is_mcast_sender) {
                compute_kernel_lib::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_SCALAR,
                    dfb_ex_external_id,
                    dfb_scaler_global_id,
                    dfb_ex2_global_id,
                    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                    compute_kernel_lib::ReduceInputBlockShape::col(dfb_ex_external_tiles_required),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::NoAccumulation{},
                    scale_by_global_recip);
                if (num_cores_per_mcast_group > 1) {
                    dfb_ex2.reserve_back(1);
                    dfb_ex2.push_back(1);
                }
            }
            // End Global Reduce

            // Start Variance Calc
            //  global reduce results
            dfb_eps.wait_front(1);
            dfb_ex2_global.wait_front(1);
            dfb_ex2pe.reserve_back(1);

            // The row mask keeps the padding out of both sums, so this is already the variance over
            // the real rows; no back-correction needed.
            // (Var + eps)
            tile_regs_acquire();
            // fp32: reset both srcs so fp32 variance / bf16 eps aren't read through the stale square/reduce format.
            if constexpr (enable_fp32_reconfig) {
                reconfig_data_format_srca(dfb_ex2_global_id);
                reconfig_data_format_srcb(dfb_eps_id);
            }
            add_init(dfb_ex2_global_id, dfb_eps_id);
            add_tiles(dfb_ex2_global_id, dfb_eps_id, 0, 0, dst0);
            tile_regs_wait();
            // 1/[sqrt(Var + eps)]
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_ex2pe_id);
            tile_regs_release();
            dfb_ex2pe.push_back(1);
            dfb_ex2_global.pop_front(1);
            // End Variance Calc

            bool start_copy_or_add = copy_or_add;
            uint32_t start_group_reset_index = group_reset_index;
            uint32_t start_index_block_w = index_block_w;

            uint32_t out_block_h_offset = 0;
            // Start Final Val Calc
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
                uint32_t out_block_h_actual;
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    out_block_h_actual = out_block_h_last;
                } else {
                    out_block_h_actual = out_block_h_normal;
                }

#ifndef TILIZE_IN
                dfb_in0.wait_front(out_block_hw_normal);
#endif
                // x - E[x]
                // fp32: reset both srcs so fp32 input/mean aren't read through the stale rsqrt/eps format.
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_input_id);
                    reconfig_data_format_srcb(dfb_ex_global_id);
                }
                sub_bcast_scalar_init(dfb_input_id, dfb_ex_global_id);
                dfb_xmm.reserve_back(out_block_hw_normal);
                dfb_ex_global.wait_front(1);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
#ifdef TILIZE_IN
                    uint32_t row_base = out_block_index * out_block_hw_normal + i * block_w;
#else
                    constexpr uint32_t row_base = 0;
#endif
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset + row_base;
                            sub_tiles_bcast_scalar(dfb_input_id, dfb_ex_global_id, index, 0, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, dfb_xmm_id);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
#ifndef TILIZE_IN
                    dfb_in0.pop_front(block_w);
#endif
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
#ifndef TILIZE_IN
                    dfb_in0.pop_front(out_block_hw_normal - out_block_hw_last);
#endif
                }
                dfb_xmm.push_back(out_block_hw_normal);

                // zero out the garbage values by mult mask again
                reconfig_data_format_srcb(dfb_ex_global_id, dfb_input_mask_id);
                mul_bcast_rows_init(dfb_xmm_id, dfb_input_mask_id);
                dfb_x.reserve_back(out_block_hw_normal);
                dfb_xmm.wait_front(out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; ++w) {
                            uint32_t index = w + index_subblock_w_offset;
                            uint32_t index_mask = index;
                            mul_tiles_bcast_rows(dfb_xmm_id, dfb_input_mask_id, index, index_mask, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; ++i) {
                            pack_tile(i, dfb_x_id);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    dfb_xmm.pop_front(block_w);
                }
                if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                    dfb_xmm.pop_front(out_block_hw_normal - out_block_hw_last);
                }
                dfb_x.push_back(out_block_hw_normal);
                reconfig_data_format_srcb(dfb_input_mask_id, dfb_x_id);

                // (x - Ex) * 1/[sqrt(Var + eps)]
                index_h_offset = 0;
                // fp32: reset both srcs so fp32 x/rstd aren't read through the stale mask/eps format.
                if constexpr (enable_fp32_reconfig) {
                    reconfig_data_format_srca(dfb_x_id);
                    reconfig_data_format_srcb(dfb_ex2pe_id);
                }
                mul_bcast_scalar_init(dfb_x_id, dfb_ex2pe_id);
                dfb_xmm.reserve_back(out_block_hw_normal);
                dfb_ex2pe.wait_front(1);
                dfb_x.wait_front(out_block_hw_normal);
                for (uint32_t i = 0; i < out_block_h_actual; i++) {
                    index_subblock_w_offset = 0;
                    for (uint32_t j = 0; j < num_subblocks_w; j++) {
                        tile_regs_acquire();
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            uint32_t index = w + index_subblock_w_offset + index_h_offset;
                            mul_tiles_bcast_scalar(dfb_x_id, dfb_ex2pe_id, index, 0, w);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < subblock_w; i++) {
                            pack_tile(i, dfb_xmm_id);
                        }
                        tile_regs_release();
                        index_subblock_w_offset += subblock_w;
                    }
                    index_h_offset += block_w;
                }
                dfb_x.pop_front(out_block_hw_normal);
                dfb_xmm.push_back(out_block_hw_normal);
                dfb_xmm.wait_front(out_block_hw_normal);

                copy_or_add = start_copy_or_add;
                group_reset_index = start_group_reset_index;
                index_block_w = start_index_block_w;

                // add or copy with previous output results
                uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

#ifdef UNTILIZE_OUT
                // Tilize the reread rows so the accumulation below sees tiles.
                compute_kernel_lib::tilize<
                    block_w,
                    dfb_reread_rm_id,
                    dfb_reread_out_id,
                    compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                    compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                    compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                    out_block_h_normal);
#endif

                dfb_reread_out.wait_front(out_block_hw_normal);
                dfb_reread_write_out.reserve_back(out_block_hw_normal);
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    uint32_t index_h_offset = 0;
                    uint32_t index_h1_offset = 0;

                    if (copy_or_add == true) {
                        copy_tile_init(dfb_xmm_id);
                    } else {
                        add_init(dfb_reread_out_id, dfb_xmm_id);
                    }

                    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                        tile_regs_acquire();
                        uint32_t index_reread_out = w + index_h_offset;
                        uint32_t index_xmm = w + index_h1_offset;

                        if (copy_or_add == true) {
                            copy_tile(dfb_xmm_id, index_xmm, dst0);
                        } else {
                            add_tiles(dfb_reread_out_id, dfb_xmm_id, index_reread_out, index_xmm, dst0);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, dfb_reread_write_out_id, index_reread_out);
                        tile_regs_release();

                        index_h_offset += block_w_curr;
                        index_h1_offset += block_w;
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
                    index_h_offset = 0;
                    dfb_outgamma.reserve_back(out_block_hw_normal);
                    dfb_gamma.wait_front(per_core_N);
                    dfb_reread_write_out.wait_front(out_block_hw_normal);
                    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                        for (uint32_t j = 0; j < block_w_curr; ++j) {
                            if (apply_gamma_beta[j]) {
                                // fp32: reset both srcs so bf16 gamma isn't read through the reread stage's fp32
                                // format.
                                if constexpr (enable_fp32_reconfig) {
                                    reconfig_data_format_srca(dfb_reread_write_out_id);
                                    reconfig_data_format_srcb(dfb_gamma_id);
                                }
                                mul_bcast_rows_init(dfb_reread_write_out_id, dfb_gamma_id);
                            } else {
                                copy_tile_init(dfb_reread_write_out_id);
                            }
                            tile_regs_acquire();
                            uint32_t index = j + index_h_offset;
                            uint32_t index_gamma = j + index_g_offset;
                            if (apply_gamma_beta[j]) {
                                mul_tiles_bcast_rows(dfb_reread_write_out_id, dfb_gamma_id, index, index_gamma, dst0);
                            } else {
                                copy_tile(dfb_reread_write_out_id, index, dst0);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(dst0, dfb_outgamma_id);
                            tile_regs_release();
                        }
                        index_h_offset += block_w_curr;
                    }
                    dfb_outgamma.push_back(out_block_hw_normal);
                    dfb_reread_write_out.pop_front(out_block_hw_normal);
                    dfb_outgamma.wait_front(out_block_hw_normal);
                }
                // End Optional Gamma
                //
                // Start Optional Beta
                if constexpr (do_beta) {
                    index_h_offset = 0;
                    dfb_outbeta.reserve_back(out_block_hw_normal);
                    dfb_beta.wait_front(per_core_N);
                    for (uint32_t i = 0; i < out_block_h_actual; ++i) {
                        for (uint32_t j = 0; j < block_w_curr; ++j) {
                            if (apply_gamma_beta[j]) {
                                // fp32: reset both srcs so bf16 beta isn't read through the fp32 dfb_inbeta format.
                                if constexpr (enable_fp32_reconfig) {
                                    reconfig_data_format_srca(dfb_inbeta_id);
                                    reconfig_data_format_srcb(dfb_beta_id);
                                }
                                add_bcast_rows_init(dfb_inbeta_id, dfb_beta_id);
                            } else {
                                copy_tile_init(dfb_inbeta_id);
                            }
                            tile_regs_acquire();
                            uint32_t index = j + index_h_offset;
                            uint32_t index_beta = j + index_g_offset;
                            if (apply_gamma_beta[j]) {
                                add_tiles_bcast_rows(dfb_inbeta_id, dfb_beta_id, index, index_beta, dst0);
                            } else {
                                copy_tile(dfb_inbeta_id, index, dst0);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(dst0, dfb_outbeta_id);
                            tile_regs_release();
                        }
                        index_h_offset += block_w_curr;
                    }
                    dfb_outbeta.push_back(out_block_hw_normal);
                    dfb_inbeta.pop_front(out_block_hw_normal);
                    dfb_outbeta.wait_front(out_block_hw_normal);
                }
                // End Optional Beta

#ifdef UNTILIZE_OUT
                // untilize - DEST capacity auto-detected.
                compute_kernel_lib::untilize<
                    block_w,
                    dfb_untilize_in_id,
                    dfb_untilize_out_id,
                    compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                    compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                    compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
                    out_block_h_normal);
#endif
            }
            // End Final Val Calc
#ifdef TILIZE_IN
            // All passes done with the group, resident group popped.
            dfb_in_resident.pop_front(num_out_blocks_padded * out_block_hw_normal);
#endif
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
            dfb_input_mask.pop_front(mask_tiles_per_group);
        }
        // End Group Loop
        index_b_offset += num_tiles_per_batch;
    }
    // End Batch Loop
}
