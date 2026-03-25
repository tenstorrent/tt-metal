// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/welford.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/circular_buffer.h"

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

    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // input cbs
    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_in_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask_id = tt::CBIndex::c_7;

    // interm cbs
    constexpr uint32_t cb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t cb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t cb_x_id = tt::CBIndex::c_13;
    constexpr uint32_t cb_xmm_id = tt::CBIndex::c_2;
    constexpr uint32_t cb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global_id = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2pe_id = tt::CBIndex::c_17;

    // output cb
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t cb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out_id =
        (do_gamma or do_beta) ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? cb_in_id : cb_out0_id)
                              : cb_out0_id;
#endif

#ifdef UNTILIZE_OUT
    constexpr int cb_outgamma_id = cb_in_id;
    constexpr int cb_outbeta_id = do_gamma ? cb_out_id : cb_in_id;
    constexpr int cb_untilize_in_id = (do_gamma and not do_beta) ? cb_outgamma_id : do_beta ? cb_outbeta_id : cb_out_id;
    constexpr int cb_untilize_out_id =
#ifdef READER_REPACK
        cb_repack_out_id;
#else
        cb_out0_id;
#endif
#else
    constexpr int cb_outgamma_id = do_beta ? cb_in_id : cb_out0_id;
    constexpr int cb_outbeta_id = cb_out0_id;
#endif

    experimental::CircularBuffer cb_beta(cb_beta_id);
    experimental::CircularBuffer cb_eps(cb_eps_id);
    experimental::CircularBuffer cb_ex2pe(cb_ex2pe_id);
    experimental::CircularBuffer cb_ex_global(cb_ex_global_id);
    experimental::CircularBuffer cb_ex_partial(cb_ex_partial_id);
    experimental::CircularBuffer cb_gamma(cb_gamma_id);
    experimental::CircularBuffer cb_in(cb_in_id);
    experimental::CircularBuffer cb_input_mask(cb_input_mask_id);
    experimental::CircularBuffer cb_x(cb_x_id);
    experimental::CircularBuffer cb_xmm(cb_xmm_id);

// tilize input from RM to tile layout
#ifdef TILIZE_IN
    binary_op_init_common(cb_in0_id, cb_in0_id, cb_in_id);
// Tilize in0 -> in (row-major to tiled)
#ifdef READER_REPACK
    constexpr uint32_t cb_in_rm_id = cb_repack_id;
    compute_kernel_lib::tilize<
        per_core_N,
        cb_in_rm_id,
        cb_in_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#else
    constexpr uint32_t cb_in_rm_id = cb_in0_id;
    compute_kernel_lib::tilize<
        per_core_N,
        cb_in_rm_id,
        cb_in_id,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::NoWait,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
    cb_in.wait_front(per_core_MN);
#else
    binary_op_init_common(cb_in0_id, cb_in0_id, cb_in0_id);
#endif

    // Sharded v2 does not use reciprocal lookup table, so we pass an empty array
    constexpr std::array<uint32_t, 0> empty_reciprocal_lut{};

    cb_eps.wait_front(1);
    cb_input_mask.wait_front(num_tiles_input_mask);

    if constexpr (do_gamma) {
        cb_gamma.wait_front(per_core_N);
    }
    if constexpr (do_beta) {
        cb_beta.wait_front(per_core_N);
    }

    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t tile_id = b * block_hw;
        cb_ex_partial.reserve_back(2);
        transpose_wh_init(cb_in0_id, cb_ex_partial_id);
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
                transpose_wh_init_short(cb_in_id);
                transpose_wh_tile(cb_in_id, tile_id, input_dst);
#else
                transpose_wh_init_short(cb_in0_id);
                transpose_wh_tile(cb_in0_id, tile_id, input_dst);
#endif

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
        pack_tile_block(mean_dst, cb_ex_partial_id, 2);
        tile_regs_release();
        cb_ex_partial.push_back(2);

        // Start Variance Calc
        // Wait for final welford values in cb_ex_global_id
        cb_ex_global.wait_front(2 * num_groups);
        cb_ex2pe.reserve_back(num_groups);
        // (Var + eps)
        add_tiles_init(cb_ex_global_id, cb_eps_id);
        reconfig_data_format_srcb(cb_eps_id);
        for (uint32_t g = 0; g < num_groups; ++g) {
            tile_regs_acquire();
            add_tiles(cb_ex_global_id, cb_eps_id, 1 + (g << 1), 0, dst0);

            // 1/[sqrt(Var + eps)]
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe_id);
            tile_regs_release();
        }
        cb_ex2pe.push_back(num_groups);
        // End Variance Calc

        cb_ex2pe.wait_front(num_groups);

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
                    cb_xmm.reserve_back(2);

                    // // Now let us do the actual computation for the current group here
                    // // a. x-u
                    sub_tiles_bcast_scalar_init_short(cb_in0_id, cb_ex_global_id);
                    reconfig_data_format(cb_in0_id, cb_ex_global_id);

                    tile_regs_acquire();
#ifdef TILIZE_IN
                    sub_tiles_bcast_scalar(cb_in_id, cb_ex_global_id, tile_id, 0 + (g << 1), dst0);
#else
                    sub_tiles_bcast_scalar(cb_in0_id, cb_ex_global_id, tile_id, 0 + (g << 1), dst0);
#endif
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, cb_xmm_id);
                    tile_regs_release();

                    // // b. 1/[sqrt(Var + eps)] * mask
                    const uint32_t mask_offset = g * block_w;
                    const uint32_t mask_index = mask_offset + block_w_index;

                    mul_tiles_bcast_scalar_init_short(cb_input_mask_id, cb_ex2pe_id);
                    reconfig_data_format(cb_in0_id, cb_input_mask_id, cb_ex_global_id, cb_ex2pe_id);
                    tile_regs_acquire();
                    mul_tiles_bcast_scalar(cb_input_mask_id, cb_ex2pe_id, mask_index, g, dst0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, cb_xmm_id);
                    tile_regs_release();
                    cb_xmm.push_back(2);

                    // // c. a * b
                    cb_xmm.wait_front(2);
                    mul_tiles_init(cb_xmm_id, cb_xmm_id);
                    reconfig_data_format(cb_input_mask_id, cb_xmm_id, cb_ex2pe_id, cb_xmm_id);
                    tile_regs_acquire();
                    mul_tiles(cb_xmm_id, cb_xmm_id, 0, 1, dst0);
                    tile_regs_commit();
                    cb_xmm.pop_front(2);
                    cb_xmm.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_xmm_id);
                    tile_regs_release();
                    cb_xmm.push_back(1);

                    // // d. Add to cb_xmm_id (accumulate results)
                    // // First we get the result in dst0
                    if (group_offset == 0) {
                        // When group_offset is 0, this is the first group for this tile,
                        // so we can copy the results to cb_x_id without needing to add them
                        copy_tile_init(cb_xmm_id);

                        cb_xmm.wait_front(1);
                        tile_regs_acquire();
                        copy_tile(cb_xmm_id, 0, dst0);
                        tile_regs_commit();
                        cb_xmm.pop_front(1);
                    } else {
                        // This is not the first group for this tile, so we need to add
                        // the results over what is already in cb_x_id
                        reconfig_data_format_srca(cb_xmm_id, cb_x_id);
                        add_tiles_init(cb_x_id, cb_xmm_id);

                        cb_xmm.wait_front(1);
                        cb_x.wait_front(1);
                        tile_regs_acquire();
                        add_tiles(cb_x_id, cb_xmm_id, 0, 0, dst0);
                        tile_regs_commit();
                        cb_xmm.pop_front(1);
                        cb_x.pop_front(1);
                    }

                    // Then we pack the result into cb_x_id
                    cb_x.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_x_id);
                    tile_regs_release();
                    cb_x.push_back(1);

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
                    mul_bcast_rows_init_short(cb_x_id, cb_gamma_id);
                    reconfig_data_format_srcb(cb_xmm_id, cb_gamma_id);

                    cb_x.wait_front(1);
                    tile_regs_acquire();
                    mul_tiles_bcast_rows(cb_x_id, cb_gamma_id, 0, nt, dst0);
                    tile_regs_commit();
                    cb_x.pop_front(1);
                    cb_x.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_x_id);
                    tile_regs_release();
                    cb_x.push_back(1);
                }

                if constexpr (do_beta) {
                    add_bcast_rows_init_short(cb_x_id, cb_beta_id);
                    reconfig_data_format_srcb(do_gamma ? cb_gamma_id : cb_xmm_id, cb_beta_id);

                    cb_x.wait_front(1);
                    tile_regs_acquire();
                    add_tiles_bcast_rows(cb_x_id, cb_beta_id, 0, nt, dst0);
                    tile_regs_commit();
                    cb_x.pop_front(1);
                    cb_x.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_x_id);
                    tile_regs_release();
                    cb_x.push_back(1);
                }

                // Write out the final output
                copy_tile_init(cb_x_id);
                reconfig_data_format_srcb(do_beta ? cb_beta_id : cb_xmm_id, cb_x_id);

                cb_x.wait_front(1);
                tile_regs_acquire();
                copy_tile(cb_x_id, 0, dst0);
                tile_regs_commit();
                cb_x.pop_front(1);
#ifdef UNTILIZE_OUT
                auto write_cb_id = cb_untilize_in_id;
#else
                auto write_cb_id = cb_out0_id;
#endif
                experimental::CircularBuffer write_cb(write_cb_id);
                write_cb.reserve_back(1);
                tile_regs_wait();
                pack_tile(dst0, write_cb_id);
                tile_regs_release();
                write_cb.push_back(1);
            }
        }

        cb_ex_global.pop_front(2 * num_groups);
        cb_ex2pe.pop_front(num_groups);
    }

    cb_eps.pop_front(1);
    cb_input_mask.pop_front(num_tiles_input_mask);

    // Pop all the cb_beta_id and cb_gamma_id if used
    if constexpr (do_beta) {
        cb_beta.pop_front(per_core_N);
    }
    if constexpr (do_gamma) {
        cb_gamma.pop_front(per_core_N);
    }

#ifdef UNTILIZE_OUT
    // untilize - DEST capacity auto-detected
    compute_kernel_lib::untilize<
        per_core_N,
        cb_untilize_in_id,
        cb_untilize_out_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_M);
#endif
}
