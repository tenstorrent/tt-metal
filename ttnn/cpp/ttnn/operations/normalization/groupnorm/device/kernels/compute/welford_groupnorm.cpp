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
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    /*
     * Definitions
     * block_h: This the length of the row we wish to processes in terms of tiles
     *
     * out_block_...: This is the length of our Circular Buffer, sometimes the length of out
     *                tensors(block_h) are larger than L1 space, so we have to process chunks of
     *                this data at a time. This chunk is called an out_block.
     *
     * num_out_blocks: This is the number of chunks specified by the use, such that a DFBs
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
     * Welford's Online Algorithm for Group Normalization:
     * This kernel implements Welford's online algorithm for numerically stable computation of
     * mean and variance across groups in a single pass through the data.
     *
     * Welford's algorithm maintains running statistics:
     * - Count: number of elements processed
     * - Mean: running average μ = Σ(x_i) / count
     * - M2: sum of squared differences from current mean = Σ((x_i - μ)^2)
     * - Variance: M2 / count (for population)
     *
     * Batch Loop:
     *   Input tiles:
     *       For each group, we accumulate Welford statistics across all channels in that group
     *       Welford Partial Tile Updates:
     *           Process input tiles incrementally, updating running mean and M2 for each group
     *           Uses welford_update_rows()
     *           Intermediate results stored in dst registers between tiles
     *   Statistics Aggregation:
     *       Local aggregation per core:
     *           Convert accumulated M2 to variance using welford_finalize_to_face()
     *           Store per-group statistics in dfb_ex_partial_id for inter-core communication
     *       Global reduction across cores:
     *           Reader kernels aggregate local statistics from all cores into dfb_ex_global_id
     *           Designated sender core produces final global mean and variance per group
     *   Normalization Factor Calculation:
     *       Add epsilon to variance: Var + eps
     *       Compute reciprocal square root: 1/sqrt(Var + eps)
     *       Store normalization factor in dfb_ex2pe_id for use in final calculation
     *   Final Normalization:
     *       Center inputs: x - μ (mean subtraction)
     *       Scale by normalization factor: (x - μ) * (1/sqrt(Var + eps))
     *       Apply input mask to handle padding/ignored elements
     *       Optional affine transform:
     *         Gamma scaling: result * γ
     *         Beta shift: result + β
     *
     * Key Welford operations:
     * - welford_init(): Initialize algorithm state
     * - welford_update_rows(): Update running statistics for partial tile data
     * - welford_save_state(): Save intermediate statistics to dst registers
     * - welford_restore_state(): Restore statistics for continued processing
     * - welford_finalize_to_face(): Convert M2 to final variance
     *
     * To find code sections, search for "Start LABEL" or "End LABEL" comments
     * Examples: "Start Welford Partial Tile" or "End Statistics Aggregation"
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
    constexpr uint32_t reciprocal_size = get_named_compile_time_arg_val("reciprocal_size");
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
#ifdef TILIZE_IN
    constexpr uint32_t dfb_welford_in_id = dfb_in_id;
#else
    constexpr uint32_t dfb_welford_in_id = dfb_in0_welford_id;
#endif
    // Boolean indicating whether the welford kernel uses the alias DFB.
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;
    // True when the welford intake DFB is configured with UnpackToDestFp32, i.e. the FP32
    // path. Covers both the TILIZE_IN branch (intake DFB is c_29) and the non-TILIZE_IN
    // alias branch (intake DFB is dfb_in0_welford, see welford_fp32_alias). On this path,
    // transpose_tile routes through llk_math_transpose_dest, whose math-side init
    // records slots [16, 32) of the math-thread replay buffer, clobbering welford's
    // LREG2 / LREG3 portions, so the welford SFPU state must be re-initialized after each
    // transpose. For bf16 input, transpose routes through SrcA without touching the
    // math-thread replay buffer, so no re-init is needed.
    constexpr bool welford_unpack_fp32_active = get_named_compile_time_arg_val("welford_unpack_fp32_active") != 0;
    // True when a reconfig-relevant operand is fp32: the per-tile reconfig_data_format calls below
    // are then required. All-bf16 compiles them out (no-ops). See program factory.
    constexpr bool enable_fp32_reconfig = get_named_compile_time_arg_val("enable_fp32_reconfig") != 0;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t dfb_input_mask_id = tt::CBIndex::c_28;
    constexpr uint32_t dfb_reciprocals_id = tt::CBIndex::c_18;

    // interm cbs
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_31;
    constexpr uint32_t dfb_x_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_xmm_id = tt::CBIndex::c_25;
    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_27;

    // interm cbs reuse
    constexpr uint32_t dfb_reread_write_out_id = tt::CBIndex::c_22;

    // output dfb_id
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

    constexpr auto ex_global_scalar_offset_input = ckl::input(
        dfb_ex_global_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Disabled,
        ckl::TileAddressing::Offset);
    constexpr auto in0_scalar_offset_input = ckl::input(
        dfb_in0_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Disabled,
        ckl::TileAddressing::Offset);
    constexpr auto ex2pe_scalar_offset_input = ckl::input(
        dfb_ex2pe_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Disabled,
        ckl::TileAddressing::Offset);
    constexpr auto input_mask_scalar_offset_input = ckl::input(
        dfb_input_mask_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Disabled,
        ckl::TileAddressing::Offset);
    constexpr auto gamma_scalar_offset_input = ckl::input(
        dfb_gamma_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Disabled,
        ckl::TileAddressing::Offset);
    constexpr auto beta_scalar_offset_input = ckl::input(
        dfb_beta_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Disabled,
        ckl::TileAddressing::Offset);
    constexpr auto xmm_per_tile_input =
        ckl::input(dfb_xmm_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    constexpr auto x_per_tile_input =
        ckl::input(dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    constexpr auto xmm_per_tile_output = ckl::output(
        dfb_xmm_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    constexpr auto x_per_tile_output =
        ckl::output(dfb_x_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    constexpr auto ex2pe_per_tile_output = ckl::output(
        dfb_ex2pe_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    constexpr auto out_per_tile_output = ckl::output(
        dfb_out_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled);

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
    compute_kernel_hw_startup(dfb_in0_id, dfb_in0_id, dfb_in0_id);
#endif

    if constexpr (welford_unpack_fp32_active) {
        // Reconfigure the transpose op for the welford intake DFB. The factory marks this DFB
        // with UnpackToDestFp32: c_29 in the TILIZE_IN branch, c_19 in the non-TILIZE_IN alias branch.
        transpose_init(dfb_welford_in_id);
    }

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    constexpr bool extra_out_block = block_h % num_out_blocks != 0;
    constexpr uint32_t num_out_blocks_padded = num_out_blocks + (extra_out_block ? 1 : 0);
    constexpr uint32_t out_block_h_last = extra_out_block ? block_h % num_out_blocks : out_block_h_normal;

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, reciprocal_size>;
    auto p_reciprocal =
        norm::kernel_util::compute::memory::get_pointer_to_cb_data<recip_lut_t>(dfb_reciprocals_id, /*tile_idx=*/0);

    dfb_eps.wait_front(1);
    dfb_input_mask.wait_front(num_tiles_input_mask);

    if constexpr (do_gamma) {
        dfb_gamma.wait_front(per_core_N);
    }
    if constexpr (do_beta) {
        dfb_beta.wait_front(per_core_N);
    }

    for (uint32_t b = 0; b < num_batches; ++b) {
        dfb_ex_partial.reserve_back(2);
        tile_regs_acquire();
        welford_init();

        uint32_t block_xy_coord = 0;

        for (uint32_t g = 0; g < num_groups; ++g) {
            welford_save_state(mean_dst, g);
        }

        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual = out_block_h_normal;
            if constexpr (extra_out_block) {
                if (out_block_index == (num_out_blocks_padded - 1)) {
                    out_block_h_actual = out_block_h_last;
                }
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

                // This tracks the global index of the first element in a given group in a tile.
                // It is used by the Welford's algorithm to scale the running mean and m2.
                // This moves reverse of channels_left, except that it is the global index.
                uint32_t curr_xy_coord = block_xy_coord;

                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    dfb_in0.wait_front(1);
                    if constexpr (welford_fp32_alias) {
                        // The reader pushes dfb_in0 and dfb_in0_welford in separate push_back
                        // calls (dfb_in0 first, alias second); dfb_in0.wait_front above only
                        // synchronizes on the first. Wait on the alias to synchronize on the
                        // second before transpose_tile reads via the alias below.
                        dfb_in0_welford.wait_front(1);
                    }
                    transpose_init(dfb_welford_in_id);
                    transpose_tile(dfb_welford_in_id, 0, input_dst);

                    // Re-establish the welford SFPU replay buffer state. When transpose_tile
                    // takes the unpack-to-DEST fp32 path, transpose_tile calls
                    // llk_math_transpose_dest, whose math-side init records slots [16, 32) of
                    // the math-thread replay buffer, clobbering welford's LREG2 / LREG3 portions.
                    // Without welford_init<WelfordInitMode::PreserveStats>(), welford_update_rows would replay stale
                    // transpose-dest ops.
                    // When the unpack-to-DEST fp32 path is inactive, transpose_tile routes
                    // through SrcA without touching the math-thread replay buffer, so re-init is
                    // not needed.
                    if constexpr (welford_unpack_fp32_active) {
                        welford_init<WelfordInitMode::PreserveStats>();
                    }

                    uint32_t group_offset = 0;
                    for (uint32_t g = min_group; g < num_groups; ++g) {
                        // Start Welford Partial Tile Updates
                        uint32_t cols_available = tile_width - group_offset;
                        uint32_t cols_consumed = std::min(cols_available, channels_left);

                        welford_restore_state(mean_dst, g);
                        welford_update_rows<reciprocal_size>(
                            input_dst, curr_xy_coord, group_offset, cols_consumed, *p_reciprocal);
                        welford_save_state(mean_dst, g);
                        // End Welford Partial Tile Updates

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
                    dfb_in0.pop_front(1);
                    if constexpr (welford_fp32_alias) {
                        dfb_in0_welford.pop_front(1);
                    }
                }
                block_xy_coord += num_channels_per_group;
            }
        }

        // Start Statistics Aggregation
        for (uint32_t g = 0; g < num_groups; ++g) {
            // Convert M2 to variance
            welford_restore_state(mean_dst, g);
            welford_finalize_to_face<reciprocal_size>(mean_dst, g, block_xy_coord - 1, *p_reciprocal);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_block(mean_dst, dfb_ex_partial_id, 2);
        tile_regs_release();
        dfb_ex_partial.push_back(2);
        // End Statistics Aggregation

        // Start Normalization Factor Calculation
        // Wait for final welford values in dfb_ex_global_id
        dfb_ex_global.wait_front(2 * num_groups);
        // fp32: dfb_ex_global is fp32 (var), dfb_eps is bf16; the welford intake left SrcA on the fp32 input alias.
        // Reset both srcs so they match the operands read below. no-op for bf16.
        if constexpr (enable_fp32_reconfig) {
            reconfig_data_format_srca(dfb_ex_global_id);
        }
        reconfig_data_format_srcb(dfb_eps_id);
        for (uint32_t g = 0; g < num_groups; ++g) {
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ex_global_scalar_offset_input,
                    ckl::input(
                        dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled)>{
                    1 + (g << 1), 0u},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
                ckl::PackTile<ex2pe_per_tile_output>{});
        }
        // End Normalization Factor Calculation

        dfb_ex2pe.wait_front(num_groups);

        // Start Final Normalization
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual = out_block_h_normal;
            if constexpr (extra_out_block) {
                if (out_block_index == (num_out_blocks_padded - 1)) {
                    out_block_h_actual = out_block_h_last;
                }
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
                        // // Now let us do the actual computation for the current group here
                        // // a. x-u
                        // fp32: reset both sources after the previous group's multiply; the old two-argument SrcB
                        // transition did not reset SrcA.
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_in0_id);
                            reconfig_data_format_srcb(dfb_ex_global_id);
                        } else {
                            reconfig_data_format_srcb(dfb_eps_id, dfb_ex_global_id);
                        }
                        ckl::eltwise_chain(
                            ckl::IterationShape::one_tile(),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Sub,
                                in0_scalar_offset_input,
                                ckl::input(ex_global_scalar_offset_input, ckl::BroadcastDim::Scalar)>{0u, g << 1},
                            ckl::PackTile<xmm_per_tile_output>{});

                        // Normalize the centered input: (x - mean) * rsqrt(variance + epsilon).
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_in0_id, dfb_xmm_id);
                        }
                        reconfig_data_format_srcb(dfb_ex_global_id, dfb_ex2pe_id);
                        ckl::eltwise_chain(
                            ckl::IterationShape::one_tile(),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                xmm_per_tile_input,
                                ckl::input(ex2pe_scalar_offset_input, ckl::BroadcastDim::Scalar)>{0u, g},
                            ckl::PackTile<xmm_per_tile_output>{});

                        // Apply the current group's row selector and accumulate into dfb_x. The mask
                        // product stays in DEST and, when a tile spans multiple groups, the running
                        // dfb_x is added via DEST_TO_SRCB reuse before the single pack — mirroring the
                        // raw kernel's fused c+d structure (no extra DEST -> L1 -> DEST round trip).
                        const uint32_t mask_offset = g * block_w;
                        const uint32_t mask_index = mask_offset + block_w_index;
                        reconfig_data_format_srcb(dfb_ex2pe_id, dfb_input_mask_id);
                        if (group_offset == 0) {
                            ckl::eltwise_chain(
                                ckl::IterationShape::one_tile(),
                                ckl::BinaryFpu<
                                    ckl::BinaryFpuOp::Mul,
                                    xmm_per_tile_input,
                                    ckl::input(input_mask_scalar_offset_input, ckl::BroadcastDim::Row)>{0u, mask_index},
                                ckl::PackTile<x_per_tile_output>{});
                        } else {
                            // Not the first group for this tile: add what is already in dfb_x.
                            reconfig_data_format_srca(dfb_x_id);
                            ckl::eltwise_chain(
                                ckl::IterationShape::one_tile(),
                                ckl::BinaryFpu<
                                    ckl::BinaryFpuOp::Mul,
                                    xmm_per_tile_input,
                                    ckl::input(input_mask_scalar_offset_input, ckl::BroadcastDim::Row)>{0u, mask_index},
                                ckl::DestReuseBinary<
                                    ckl::BinaryFpuOp::Add,
                                    x_per_tile_input,
                                    ckl::DestReuseType::DEST_TO_SRCB>{},
                                ckl::PackTile<x_per_tile_output>{});
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
                    dfb_in0.pop_front(1);
                    if constexpr (welford_fp32_alias) {
                        dfb_in0_welford.pop_front(1);
                    }

                    if constexpr (do_gamma) {
                        // fp32: reset SrcA to dfb_x; the prior mask/accumulate step may leave it on bf16.
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_x_id);
                        }
                        reconfig_data_format_srcb(dfb_xmm_id, dfb_gamma_id);
                        ckl::eltwise_chain(
                            ckl::IterationShape::one_tile(),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                x_per_tile_input,
                                ckl::input(gamma_scalar_offset_input, ckl::BroadcastDim::Row)>{0u, nt},
                            ckl::PackTile<x_per_tile_output>{});
                    }

                    if constexpr (do_beta) {
                        // fp32: reset SrcA to dfb_x, as in the gamma step above.
                        if constexpr (enable_fp32_reconfig) {
                            reconfig_data_format_srca(dfb_x_id);
                        }
                        reconfig_data_format_srcb(do_gamma ? dfb_gamma_id : dfb_xmm_id, dfb_beta_id);
                        ckl::eltwise_chain(
                            ckl::IterationShape::one_tile(),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Add,
                                x_per_tile_input,
                                ckl::input(beta_scalar_offset_input, ckl::BroadcastDim::Row)>{0u, nt},
                            ckl::PackTile<x_per_tile_output>{});
                    }

                    // Write out the final output
                    if constexpr (enable_fp32_reconfig) {
                        reconfig_data_format_srca(dfb_x_id);
                    }
                    reconfig_data_format_srcb(do_beta ? dfb_beta_id : dfb_xmm_id, dfb_x_id);
#ifndef UNTILIZE_OUT
                    // The streaming output disables automatic reconfiguration, so select the fp32 output format.
                    // Packer was last set for bf16 dfb_x; reconfigure to dfb_out_id (may be fp32) before pack, restore
                    // after. Only needed when out differs from dfb_x (fp32 path); no-op gated out for bf16.
                    if constexpr (enable_fp32_reconfig) {
                        pack_reconfig_data_format(dfb_out_id);
                    }
#endif
                    ckl::copy<x_per_tile_input, out_per_tile_output>(ckl::IterationShape::one_tile());
#ifndef UNTILIZE_OUT
                    if constexpr (enable_fp32_reconfig) {
                        pack_reconfig_data_format(dfb_x_id);
                    }
#endif
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
        // End Final Normalization

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
}
