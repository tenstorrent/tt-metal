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

namespace NAMESPACE {
void MAIN {
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

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // input cbs
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask = tt::CBIndex::c_7;

    // interm cbs
    constexpr uint32_t cb_repack = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_12;
    constexpr uint32_t cb_x = tt::CBIndex::c_13;
    constexpr uint32_t cb_xmm = tt::CBIndex::c_2;
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_17;

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

    // Sharded v2 does not use reciprocal lookup table, so we pass an empty array
    constexpr std::array<uint32_t, 0> empty_reciprocal_lut{};

    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_input_mask, num_tiles_input_mask);

    if constexpr (do_gamma) {
        cb_wait_front(cb_gamma, per_core_N);
    }
    if constexpr (do_beta) {
        cb_wait_front(cb_beta, per_core_N);
    }

    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t tile_id = b * block_hw;
        cb_reserve_back(cb_ex_partial, 2);
        transpose_wh_init(cb_in0, cb_ex_partial);
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
#ifdef TILIZE_IN
                transpose_wh_init_short(cb_in);
                transpose_wh_tile(cb_in, tile_id, input_dst);
#else
                transpose_wh_init_short(cb_in0);
                transpose_wh_tile(cb_in0, tile_id, input_dst);
#endif

                uint32_t group_offset = 0;
                for (uint32_t g = min_group; g < num_groups; ++g) {
                    // Start Welford's Calculation
                    uint32_t cols_available = tt::constants::TILE_WIDTH - group_offset;
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
                    if (group_offset == tt::constants::TILE_WIDTH) {
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
        pack_tile_block(mean_dst, cb_ex_partial, 2);
        tile_regs_release();
        cb_push_back(cb_ex_partial, 2);

        // Start Variance Calc
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
        // End Variance Calc

        cb_wait_front(cb_ex2pe, num_groups);

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
                    cb_reserve_back(cb_xmm, 2);

                    // // Now let us do the actual computation for the current group here
                    // // a. x-u
                    sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
                    reconfig_data_format(cb_in0, cb_ex_global);

                    tile_regs_acquire();
#ifdef TILIZE_IN
                    sub_tiles_bcast_scalar(cb_in, cb_ex_global, tile_id, 0 + (g << 1), dst0);
#else
                    sub_tiles_bcast_scalar(cb_in0, cb_ex_global, tile_id, 0 + (g << 1), dst0);
#endif
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, cb_xmm);
                    tile_regs_release();

                    // // b. 1/[sqrt(Var + eps)] * mask
                    const uint32_t mask_offset = g * block_w;
                    const uint32_t mask_index = mask_offset + block_w_index;

                    mul_tiles_bcast_scalar_init_short(cb_input_mask, cb_ex2pe);
                    reconfig_data_format(cb_in0, cb_input_mask, cb_ex_global, cb_ex2pe);
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
                    reconfig_data_format(cb_input_mask, cb_xmm, cb_ex2pe, cb_xmm);
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
                        reconfig_data_format_srca(cb_xmm, cb_x);
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
                ++tile_id;

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
#ifdef UNTILIZE_OUT
                auto write_cb = cb_untilize_in;
#else
                auto write_cb = cb_out0;
#endif
                cb_reserve_back(write_cb, 1);
                tile_regs_wait();
                pack_tile(dst0, write_cb);
                tile_regs_release();
                cb_push_back(write_cb, 1);
            }
        }

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
}  // namespace NAMESPACE
