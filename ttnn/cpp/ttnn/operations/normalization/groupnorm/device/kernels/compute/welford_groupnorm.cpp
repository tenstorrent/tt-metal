// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "tt-metalium/constants.hpp"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/welford.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

namespace NAMESPACE {
void MAIN {
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
     * This is a high level desciption of the stages of this kernel, tags will be added to show
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
     *           Store per-group statistics in cb_ex_partial for inter-core communication
     *       Global reduction across cores:
     *           Reader kernels aggregate local statistics from all cores into cb_ex_global
     *           Designated sender core produces final global mean and variance per group
     *   Normalization Factor Calculation:
     *       Add epsilon to variance: Var + eps
     *       Compute reciprocal square root: 1/sqrt(Var + eps)
     *       Store normalization factor in cb_ex2pe for use in final calculation
     *   Final Normalization:
     *       Center inputs: x - μ (mean subtraction)
     *       Apply input mask to handle padding/ignored elements
     *       Scale by normalization factor: (x - μ) * (1/sqrt(Var + eps))
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

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // input cbs
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in = tt::CBIndex::c_29;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask = tt::CBIndex::c_28;
    constexpr uint32_t cb_reciprocals = tt::CBIndex::c_18;

    // interm cbs
    constexpr uint32_t cb_repack = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_31;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;
    constexpr uint32_t cb_xmm = tt::CBIndex::c_25;
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_27;

    // interm cbs reuse
    constexpr uint32_t cb_reread_write_out = tt::CBIndex::c_22;

    // output cb
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t cb_out = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out = (do_gamma or do_beta)
                                    ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? cb_in : cb_out0)
                                    : cb_out0;
#endif

#ifdef UNTILIZE_OUT
    constexpr int cb_outgamma = cb_in;
    constexpr int cb_inbeta = do_gamma ? cb_outgamma : cb_out;
    constexpr int cb_outbeta = do_gamma ? cb_out : cb_in;
    constexpr int cb_untilize_in = (do_gamma and not do_beta) ? cb_outgamma : do_beta ? cb_outbeta : cb_out;
    constexpr int cb_untilize_out =
#ifdef READER_REPACK
        cb_repack_out;
#else
        cb_out0;
#endif
#else
    constexpr int cb_outgamma = do_beta ? cb_in : cb_out0;
    constexpr int cb_inbeta = do_gamma ? cb_outgamma : cb_reread_write_out;
    constexpr int cb_outbeta = cb_out0;
#endif

// tilize input from RM to tile layout
#ifdef TILIZE_IN
    binary_op_init_common(cb_in0, cb_in0, cb_in);
// tilize in0 -> in
#ifdef READER_REPACK
    constexpr uint32_t cb_in_rm = cb_repack;
#else
    constexpr uint32_t cb_in_rm = cb_in0;
#endif
    tilize_init(cb_in_rm, per_core_N, cb_in);
    for (uint32_t m = 0; m < per_core_M; ++m) {
#ifdef READER_REPACK
        cb_wait_front(cb_in_rm, per_core_N);
#endif
        cb_reserve_back(cb_in, per_core_N);
        tilize_block(cb_in_rm, per_core_N, cb_in);
        cb_push_back(cb_in, per_core_N);
        cb_pop_front(cb_in_rm, per_core_N);
    }
    tilize_uninit(cb_in_rm, cb_in);
    cb_wait_front(cb_in, per_core_MN);
#else
    binary_op_init_common(cb_in0, cb_in0, cb_in0);
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
    uint32_t cb_ex_external_tiles_required =
        num_out_blocks_padded * num_cores_per_mcast_group * 16 / single_tile_size_bytes;
    if ((num_out_blocks_padded * num_cores_per_mcast_group * 16) % single_tile_size_bytes) {
        cb_ex_external_tiles_required++;
    }

    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, reciprocal_size>;
    auto p_reciprocal =
        norm::kernel_util::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, /*tile_idx=*/0);

    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_input_mask, num_tiles_input_mask);

    if constexpr (do_gamma) {
        cb_wait_front(cb_gamma, per_core_N);
    }
    if constexpr (do_beta) {
        cb_wait_front(cb_beta, per_core_N);
    }

    for (uint32_t b = 0; b < num_batches; ++b) {
        cb_reserve_back(cb_ex_partial, 2);
        tile_regs_acquire();
        welford_init();

        uint32_t block_xy_coord = 0;

        for (uint32_t g = 0; g < num_groups; ++g) {
            welford_save_state(mean_dst, g);
        }

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

                // This tracks the global index of the first element in a given group in a tile.
                // It is used by the Welford's algorithm to scale the running mean and m2.
                // This moves reverse of channels_left, except that it is the global index.
                uint32_t curr_xy_coord = block_xy_coord;

                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_wait_front(cb_in0, 1);
#ifdef TILIZE_IN
                    transpose_wh_init_short(cb_in);
                    transpose_wh_tile(cb_in, 0, input_dst);
#else
                    transpose_wh_init_short(cb_in0);
                    transpose_wh_tile(cb_in0, 0, input_dst);
#endif

                    uint32_t group_offset = 0;
                    for (uint32_t g = min_group; g < num_groups; ++g) {
                        // Start Welford Partial Tile Updates
                        uint32_t cols_available = tt::constants::TILE_WIDTH - group_offset;
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
                        if (group_offset == tt::constants::TILE_WIDTH) {
                            break;
                        }
                    }
                    cb_pop_front(cb_in0, 1);
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
        pack_tile_block(mean_dst, cb_ex_partial, 2);
        tile_regs_release();
        cb_push_back(cb_ex_partial, 2);
        // End Statistics Aggregation

        // Start Normalization Factor Calculation
        // Wait for final welford values in cb_ex_global
        cb_wait_front(cb_ex_global, 2 * num_groups);
        cb_reserve_back(cb_ex2pe, num_groups);
        // (Var + eps)
        add_tiles_init(cb_ex_global, cb_eps);
        reconfig_data_format_srcb(cb_eps);
        for (uint32_t g = 0; g < num_groups; ++g) {
            tile_regs_acquire();
            add_tiles(cb_ex_global, cb_eps, 1 + (g << 1), 0, dst0);

            // 1/[sqrt(Var + eps)]
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            tile_regs_release();
        }
        cb_push_back(cb_ex2pe, num_groups);
        // End Normalization Factor Calculation

        cb_wait_front(cb_ex2pe, num_groups);

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
                    cb_wait_front(cb_in0, 1);

                    uint32_t group_offset = 0;
                    for (uint32_t g = min_group; g < num_groups; ++g) {
                        cb_reserve_back(cb_xmm, 2);

                        // // Now let us do the actual computation for the current group here
                        // // a. x-u
                        sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
                        reconfig_data_format_srcb(cb_eps, cb_ex_global);

                        tile_regs_acquire();
                        sub_tiles_bcast_scalar(cb_in0, cb_ex_global, 0, 0 + (g << 1), dst0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(dst0, cb_xmm);
                        tile_regs_release();

                        // // b. 1/[sqrt(Var + eps)] * mask
                        const uint32_t mask_offset = g * block_w;
                        const uint32_t mask_index = mask_offset + block_w_index;

                        mul_tiles_bcast_scalar_init_short(cb_input_mask, cb_ex2pe);
                        reconfig_data_format_srcb(cb_ex_global, cb_ex2pe);
                        tile_regs_acquire();
                        mul_tiles_bcast_scalar(cb_input_mask, cb_ex2pe, mask_index, g, dst0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(dst0, cb_xmm);
                        tile_regs_release();
                        cb_push_back(cb_xmm, 2);

                        // // c. a * b
                        cb_wait_front(cb_xmm, 2);
                        mul_tiles_init(cb_xmm, cb_xmm);
                        reconfig_data_format_srcb(cb_ex2pe, cb_xmm);
                        tile_regs_acquire();
                        mul_tiles(cb_xmm, cb_xmm, 0, 1, dst0);
                        tile_regs_commit();
                        cb_pop_front(cb_xmm, 2);
                        cb_reserve_back(cb_xmm, 1);
                        tile_regs_wait();
                        pack_tile(dst0, cb_xmm);
                        tile_regs_release();
                        cb_push_back(cb_xmm, 1);

                        // // d. Add to cb_xmm (accumulate results)
                        // // First we get the result in dst0
                        if (group_offset == 0) {
                            // When group_offset is 0, this is the first group for this tile,
                            // so we can copy the results to cb_x without needing to add them
                            copy_tile_init(cb_xmm);

                            cb_wait_front(cb_xmm, 1);
                            tile_regs_acquire();
                            copy_tile(cb_xmm, 0, dst0);
                            tile_regs_commit();
                            cb_pop_front(cb_xmm, 1);
                        } else {
                            // This is not the first group for this tile, so we need to add
                            // the results over what is already in cb_x
                            add_tiles_init(cb_x, cb_xmm);

                            cb_wait_front(cb_xmm, 1);
                            cb_wait_front(cb_x, 1);
                            tile_regs_acquire();
                            add_tiles(cb_x, cb_xmm, 0, 0, dst0);
                            tile_regs_commit();
                            cb_pop_front(cb_xmm, 1);
                            cb_pop_front(cb_x, 1);
                        }

                        // Then we pack the result into cb_x
                        cb_reserve_back(cb_x, 1);
                        tile_regs_wait();
                        pack_tile(dst0, cb_x);
                        tile_regs_release();
                        cb_push_back(cb_x, 1);

                        uint32_t cols_available = tt::constants::TILE_WIDTH - group_offset;
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
                        if (group_offset == tt::constants::TILE_WIDTH) {
                            break;
                        }
                    }
                    cb_pop_front(cb_in0, 1);

                    if constexpr (do_gamma) {
                        mul_bcast_rows_init_short(cb_x, cb_gamma);
                        reconfig_data_format_srcb(cb_xmm, cb_gamma);

                        cb_wait_front(cb_x, 1);
                        tile_regs_acquire();
                        mul_tiles_bcast_rows(cb_x, cb_gamma, 0, nt, dst0);
                        tile_regs_commit();
                        cb_pop_front(cb_x, 1);
                        cb_reserve_back(cb_x, 1);
                        tile_regs_wait();
                        pack_tile(dst0, cb_x);
                        tile_regs_release();
                        cb_push_back(cb_x, 1);
                    }

                    if constexpr (do_beta) {
                        add_bcast_rows_init_short(cb_x, cb_beta);
                        reconfig_data_format_srcb(do_gamma ? cb_gamma : cb_xmm, cb_beta);

                        cb_wait_front(cb_x, 1);
                        tile_regs_acquire();
                        add_tiles_bcast_rows(cb_x, cb_beta, 0, nt, dst0);
                        tile_regs_commit();
                        cb_pop_front(cb_x, 1);
                        cb_reserve_back(cb_x, 1);
                        tile_regs_wait();
                        pack_tile(dst0, cb_x);
                        tile_regs_release();
                        cb_push_back(cb_x, 1);
                    }

                    // Write out the final output
                    copy_tile_init(cb_x);
                    reconfig_data_format_srcb(do_beta ? cb_beta : cb_xmm, cb_x);

                    cb_wait_front(cb_x, 1);
                    tile_regs_acquire();
                    copy_tile(cb_x, 0, dst0);
                    tile_regs_commit();
                    cb_pop_front(cb_x, 1);
                    cb_reserve_back(cb_out0, 1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_out0);
                    tile_regs_release();
                    cb_push_back(cb_out0, 1);
                }
            }

#ifdef UNTILIZE_OUT
            // untilize
            untilize_init(cb_untilize_in);
            cb_wait_front(cb_untilize_in, per_core_MN);
            for (uint32_t m = 0; m < per_core_M; ++m) {
                cb_reserve_back(cb_untilize_out, per_core_N);
                untilize_block(cb_untilize_in, per_core_N, cb_untilize_out);
                cb_push_back(cb_untilize_out, per_core_N);
                cb_pop_front(cb_untilize_in, per_core_N);
            }
            untilize_uninit(cb_untilize_in);
#endif
        }
        // End Final Normalization

        cb_pop_front(cb_ex_global, 2 * num_groups);
        cb_pop_front(cb_ex2pe, num_groups);
    }

    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_input_mask, num_tiles_input_mask);

    // Pop all the cb_beta and cb_gamma if used
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, per_core_N);
    }
    if constexpr (do_gamma) {
        cb_pop_front(cb_gamma, per_core_N);
    }
}
}  // namespace NAMESPACE
