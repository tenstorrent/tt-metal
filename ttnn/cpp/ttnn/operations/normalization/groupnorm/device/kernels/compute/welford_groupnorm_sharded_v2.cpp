// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

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
    constexpr uint32_t channels_per_group = get_compile_time_arg_val(24);

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = TILE_WIDTH - num_cols_per_group;

    // dst regs
    constexpr uint32_t dst0 = 0;

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
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_17;

    // output cb
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    // not used in cases of negative mask
    constexpr uint32_t cb_out = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out = (do_gamma or do_beta)
                                    ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? cb_in : cb_out0)
                                    : cb_out0;
#endif

    // tile offset
    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
    uint32_t index_b_offset = 0;
    uint32_t index_g_offset = 0;
    uint32_t tile_offset = 0;
    // inplace out cbs
    bool copy_or_add = true;
    uint32_t group_reset_index = 0;
    uint32_t index_block_w = 0;
    uint32_t row_offset = num_cols_per_group;

#ifdef UNTILIZE_OUT
#ifndef FUSE_NEGATIVE_MASK
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
    constexpr int cb_outgamma = cb_in;
    constexpr int cb_inbeta = cb_in;
    constexpr int cb_outbeta = cb_in;
    constexpr int cb_untilize_in = cb_in;
    constexpr int cb_untilize_out =
#ifdef READER_REPACK
        cb_repack_out;
#else
        cb_out0;
#endif
#endif
#else
    constexpr int cb_outgamma = do_beta ? cb_in : cb_out0;
    constexpr int cb_inbeta = do_gamma ? cb_outgamma : cb_out;
    constexpr int cb_outbeta = cb_out0;
#endif

    // Used in cases of negative mask provided
    constexpr uint32_t cb_in_negative_mask = tt::CBIndex::c_14;

    // Sharded v2 does not use reciprocal lookup table, so we pass an empty array
    constexpr std::array<uint32_t, 0> empty_reciprocal_lut{};

#ifdef FUSE_NEGATIVE_MASK
    constexpr bool use_negative_mask = true;
#else
    constexpr bool use_negative_mask = false;
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
#ifdef TILIZE_IN
    binary_op_init_common(cb_in, cb_ex_global, cb_x);
#else
    binary_op_init_common(cb_in0, cb_ex_global, cb_x);
#endif
#endif

    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        tile_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {
            uint32_t curr_xy_coord = 0;
            uint32_t curr_xy_limit = 0;

            // Compute Welford values and write to cb_ex_partial
            index_h_offset = index_b_offset + index_g_offset;
            cb_reserve_back(cb_ex_partial, 2);
            welford_init();
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_h; ++i) {
                curr_xy_limit += channels_per_group;
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;

                        // Check if this is the first tile in the row and set tile_offset accordingly
                        auto this_tile_offset = (j + w) ? 0 : tile_offset;
#ifdef TILIZE_IN
                        transpose_wh_init_short(cb_in);
                        transpose_wh_tile(cb_in, index, 0);
#else
                        transpose_wh_init_short(cb_in0);
                        transpose_wh_tile(cb_in0, index, 0);
#endif
                        welford_tile<dst0, 1, 2, false, 0>(
                            curr_xy_coord, curr_xy_limit, this_tile_offset, empty_reciprocal_lut);
                        curr_xy_coord += std::min(32 - this_tile_offset, curr_xy_limit - curr_xy_coord);
                    }
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }
            welford_M2_to_var<1, 2, 0>(curr_xy_limit, empty_reciprocal_lut);  // Convert M2 to variance

            // Update for next group
            tile_offset = (tile_offset + channels_per_group) % TILE_WIDTH;

            tile_regs_commit();
            tile_regs_wait();
            pack_tile_block(1, cb_ex_partial, 2);
            tile_regs_release();
            cb_push_back(cb_ex_partial, 2);

            // x - E[x]
            reconfig_data_format_srcb(cb_x, cb_ex_global);
#ifdef TILIZE_IN
            sub_tiles_bcast_scalar_init_short(cb_in, cb_ex_global);
#else
            sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
#endif
            // Wait for final welford values in cb_ex_global
            cb_wait_front(cb_ex_global, 2);
            index_h_offset = index_b_offset + index_g_offset;
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
#ifdef TILIZE_IN
                        sub_tiles_bcast_scalar(cb_in, cb_ex_global, index, 0, w);
#else
                        sub_tiles_bcast_scalar(cb_in0, cb_ex_global, index, 0, w);
#endif
                    }
                    tile_regs_commit();
                    cb_reserve_back(cb_x, subblock_w);
                    tile_regs_wait();
                    for (uint32_t k = 0; k < subblock_w; k++) {
                        pack_tile(k, cb_x);
                    }
                    cb_push_back(cb_x, subblock_w);
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }

            // Mask out the garbage values
            reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
            mul_tiles_init(cb_x, cb_input_mask);
            cb_wait_front(cb_input_mask, block_w);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    cb_wait_front(cb_x, subblock_w);
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index_mask = w + index_subblock_w_offset;
                        mul_tiles(cb_x, cb_input_mask, w, index_mask, w);
                    }
                    tile_regs_commit();

                    cb_pop_front(cb_x, subblock_w);
                    cb_reserve_back(cb_x, subblock_w);

                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, cb_x);
                    }
                    cb_push_back(cb_x, subblock_w);
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
            }
            cb_pop_front(cb_input_mask, block_w);
            reconfig_data_format_srcb(cb_input_mask, cb_eps);

            // (Var + eps)
            cb_wait_front(cb_eps, 1);
            cb_reserve_back(cb_ex2pe, 1);
            tile_regs_acquire();
            add_tiles_init(cb_ex_global, cb_eps);
            add_tiles(cb_ex_global, cb_eps, 1, 0, dst0);
            // 1/[sqrt(Var + eps)]
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            tile_regs_release();
            cb_push_back(cb_ex2pe, 1);
            cb_pop_front(cb_ex_global, 2);

            //  (x - Ex) * 1/[sqrt(Var + eps)]
            mul_tiles_bcast_scalar_init_short(cb_x, cb_ex2pe);

            cb_wait_front(cb_ex2pe, 1);
            cb_wait_front(cb_x, block_hw);

            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset;
                        mul_tiles_bcast_scalar(cb_x, cb_ex2pe, index, 0, w);
                    }
                    tile_regs_commit();
                    cb_pop_front(cb_x, subblock_w);
                    cb_reserve_back(cb_x, subblock_w);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_x);
                    }
                    cb_push_back(cb_x, subblock_w);
                    tile_regs_release();
                }
            }
            cb_pop_front(cb_ex2pe, 1);
            cb_wait_front(cb_x, block_hw);
            //  add or copy with previous output results
            uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

            // if we are using negative mask, we are overlapping tilized in and out, otherwise they are 2 separate
            // buffers.
            if constexpr (use_negative_mask == false) {
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    if (copy_or_add == true) {
                        copy_tile_init(cb_x);
                    } else {
                        add_tiles_init(cb_out, cb_x);
                    }

                    for (uint32_t i = 0; i < block_h; ++i) {
                        tile_regs_acquire();
                        uint32_t index_x = w + index_h1_offset;
                        uint32_t index = w + index_h_offset;

                        if (copy_or_add == true) {
                            copy_tile(cb_x, index_x, dst0);
                        } else {
                            add_tiles(cb_out, cb_x, index, index_x, dst0);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, cb_out, index);
                        tile_regs_release();

                        index_h_offset += per_core_N;
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
                }
            } else {
                // zero out values in cb_tilized_in input by multiplying with negative mask for the current group
                cb_wait_front(cb_in_negative_mask, block_w);
                reconfig_data_format_srcb(cb_x, cb_in_negative_mask);
                mul_tiles_init(cb_in, cb_in_negative_mask);

                for (uint32_t w = 0; w < block_w_curr; w++) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    for (uint32_t i = 0; i < block_h; i++) {
                        tile_regs_acquire();
                        uint32_t index_in = w + index_h_offset;
                        uint32_t index_mask = w;

                        mul_tiles(cb_in, cb_in_negative_mask, index_in, index_mask, dst0);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile<true>(dst0, cb_in, index_in);
                        tile_regs_release();

                        index_h_offset += per_core_N;
                    }
                }

                reconfig_data_format_srcb(cb_in_negative_mask, cb_x);
                add_tiles_init(cb_in, cb_x);
                // data in cb_x has valid data only for current group
                // cb_in has cleared data for that group
                // just add them together
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    for (uint32_t i = 0; i < block_h; ++i) {
                        tile_regs_acquire();
                        uint32_t index_x = w + index_h1_offset;
                        uint32_t index = w + index_h_offset;

                        add_tiles(cb_in, cb_x, index, index_x, dst0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, cb_in, index);
                        tile_regs_release();

                        index_h_offset += per_core_N;
                        index_h1_offset += block_w;
                    }
                }
                cb_pop_front(cb_in_negative_mask, block_w);
            }

            cb_pop_front(cb_x, block_hw);

            if constexpr (GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;
                }
            } else if constexpr (GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            } else {
                if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset - tile_w_minux_group_size;
                } else {
                    row_offset += num_cols_per_group;
                    index_g_offset += block_w_minus_two;
                }
            }
        }
        index_b_offset += num_tiles_per_batch;
    }

    if constexpr (use_negative_mask == false) {
        cb_push_back(cb_out, per_core_MN);
        cb_pop_front(cb_in, per_core_MN);

    } else {
        // nothing, for the negative mask implementation, cb_in is the only cb in use, and it already has the data
        // required for the rest of kernel.
    }

    if constexpr (do_gamma) {
        index_h_offset = 0;
        if constexpr (use_negative_mask == false) {
            mul_bcast_rows_init_short(cb_out, cb_gamma);
            cb_reserve_back(cb_outgamma, per_core_MN);
            cb_wait_front(cb_gamma, per_core_N);
            for (uint32_t i = 0; i < per_core_M; ++i) {
                for (uint32_t j = 0; j < per_core_N; ++j) {
                    tile_regs_acquire();
                    uint32_t index = j + index_h_offset;
                    mul_tiles_bcast_rows(cb_out, cb_gamma, index, j, dst0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, cb_outgamma);
                    tile_regs_release();
                }
                index_h_offset += per_core_N;
            }

            cb_push_back(cb_outgamma, per_core_MN);
            cb_pop_front(cb_out, per_core_MN);
            cb_wait_front(cb_outgamma, per_core_MN);
        } else {
            // cb in has data required for gamma, so we do it inplace
            mul_bcast_rows_init_short(cb_in, cb_gamma);
            cb_wait_front(cb_gamma, per_core_N);
            cb_wait_front(cb_in, per_core_MN);
            for (uint32_t i = 0; i < per_core_M; i++) {
                for (uint32_t j = 0; j < per_core_N; j++) {
                    tile_regs_acquire();
                    mul_tiles_bcast_rows(cb_in, cb_gamma, 0, j, dst0);
                    tile_regs_commit();
                    cb_pop_front(cb_in, 1);
                    cb_reserve_back(cb_in, 1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_in);
                    cb_push_back(cb_in, 1);
                    tile_regs_release();
                }
            }
        }
    }

    if constexpr (do_beta) {
        if constexpr (use_negative_mask == false) {
            index_h_offset = 0;
            add_bcast_rows_init_short(cb_inbeta, cb_beta);
            cb_reserve_back(cb_outbeta, per_core_MN);
            cb_wait_front(cb_beta, per_core_N);
            for (uint32_t i = 0; i < per_core_M; ++i) {
                for (uint32_t j = 0; j < per_core_N; ++j) {
                    tile_regs_acquire();
                    uint32_t index = j + index_h_offset;
                    add_tiles_bcast_rows(cb_inbeta, cb_beta, index, j, dst0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, cb_outbeta);
                    tile_regs_release();
                }
                index_h_offset += per_core_N;
            }
            cb_push_back(cb_outbeta, per_core_MN);
            cb_pop_front(cb_inbeta, per_core_MN);
            cb_wait_front(cb_outbeta, per_core_MN);
        } else {
            // cb_in has data required for beta, so we do it inplace
            add_bcast_rows_init_short(cb_in, cb_beta);
            cb_wait_front(cb_beta, per_core_N);
            cb_wait_front(cb_in, per_core_MN);
            for (uint32_t i = 0; i < per_core_M; i++) {
                for (uint32_t j = 0; j < per_core_N; j++) {
                    tile_regs_acquire();
                    add_tiles_bcast_rows(cb_in, cb_beta, 0, j, dst0);
                    tile_regs_commit();
                    cb_pop_front(cb_in, 1);
                    cb_reserve_back(cb_in, 1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_in);
                    cb_push_back(cb_in, 1);
                    tile_regs_release();
                }
            }
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
}  // namespace NAMESPACE
