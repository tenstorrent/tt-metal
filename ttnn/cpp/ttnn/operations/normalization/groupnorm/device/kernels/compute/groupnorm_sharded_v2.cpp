// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

// #include "debug/dprint.h"
// #include "debug/waypoint.h"


// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {
    constexpr uint32_t is_mcast_sender                = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma                       = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta                        = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_per_mcast_group      = get_compile_time_arg_val(3);

    constexpr uint32_t batch                          = get_compile_time_arg_val(4);
    constexpr uint32_t group                          = get_compile_time_arg_val(5);

    constexpr uint32_t num_cols_per_group            = get_compile_time_arg_val(6);

    volatile uint32_t block_h                        = get_compile_time_arg_val(7);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(8);
    constexpr uint32_t block_hw                       = get_compile_time_arg_val(9);

    constexpr uint32_t subblock_w                     = get_compile_time_arg_val(10);
    constexpr uint32_t num_subblocks_w                = get_compile_time_arg_val(11);

    constexpr uint32_t per_core_M                       = get_compile_time_arg_val(12);
    constexpr uint32_t per_core_N                       = get_compile_time_arg_val(13);
    constexpr uint32_t per_core_MN                       = get_compile_time_arg_val(14);

    constexpr uint32_t per_core_N_tile_bytes               = get_compile_time_arg_val(15);
    constexpr uint32_t num_groups_per_reset                = get_compile_time_arg_val(16);

    constexpr uint32_t single_tile_size_bytes                = get_compile_time_arg_val(17);
    constexpr uint32_t num_tiles_per_batch                  = get_compile_time_arg_val(18);

    constexpr uint32_t num_tiles_input_mask                  = get_compile_time_arg_val(19);
    constexpr uint32_t block_w_last                  = get_compile_time_arg_val(20);
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2         = get_compile_time_arg_val(21);
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W         = get_compile_time_arg_val(22);
    constexpr uint32_t group_row_offset         = get_compile_time_arg_val(23);

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = TILE_WIDTH - num_cols_per_group;


    // dst regs
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // input cbs
    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in = tt::CB::c_intermed5;
    constexpr uint32_t cb_scaler = tt::CB::c_in2;
    constexpr uint32_t cb_scaler_global = tt::CB::c_in4;
    constexpr uint32_t cb_eps = tt::CB::c_in3;
    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;
    constexpr uint32_t cb_input_mask = tt::CB::c_intermed4;

    // interm cbs
    constexpr uint32_t cb_repack = tt::CB::c_intermed2;
    constexpr uint32_t cb_repack_out = tt::CB::c_intermed7;
    constexpr uint32_t cb_x = tt::CB::c_intermed0;
    constexpr uint32_t cb_xmm = tt::CB::c_intermed1;
    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0;
    constexpr uint32_t cb_ex = tt::CB::dataflow1;
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_global = num_cores_per_mcast_group == 1 ? cb_ex_partial : tt::CB::dataflow7;
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3;

    // interm cbs reuse
    constexpr uint32_t cb_fusion = cb_xmm;
    constexpr uint32_t cb_xmm2 = cb_x;

    // output cb
    constexpr uint32_t cb_out0 = tt::CB::c_out0;
    #ifdef UNTILIZE_OUT
    constexpr uint32_t cb_out = tt::CB::c_intermed6;
    #else
    constexpr uint32_t cb_out = (do_gamma or do_beta) ? (((do_gamma and not do_beta) or (not do_gamma and do_beta)) ? cb_in : cb_out0) : cb_out0;
    #endif


    // tile offset
    uint32_t index_subblock_w_offset = 0;
    uint32_t index_h_offset = 0;
    uint32_t index_w_offset = 0;
    uint32_t index_b_offset = 0;
    uint32_t index_g_offset = 0;
    uint32_t index_mask_offset = 0;
    // data offset
    uint32_t num_datum_per_row_offeset = 0;
    // inplace out cbs
    bool copy_or_add = true;
    bool reset_index = false;
    uint32_t group_reset_index = 0;
    uint32_t index_block_w = 0;
    uint32_t row_offset = num_cols_per_group;
    uint32_t output_tile_index = 0;

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
        constexpr int cb_inbeta = do_gamma ? cb_outgamma : cb_out;
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
        tilize_init_short(cb_in_rm, per_core_N);
        for (uint32_t m = 0; m < per_core_M; ++m) {
            #ifdef READER_REPACK
            cb_wait_front(cb_in_rm, per_core_N);
            #endif
            cb_reserve_back(cb_in, per_core_N);
            tilize_block(cb_in_rm, per_core_N, cb_in);
            cb_push_back(cb_in, per_core_N);
            cb_pop_front(cb_in_rm, per_core_N);
        }
        tilize_uninit(cb_in_rm);
        cb_wait_front(cb_in, per_core_MN);
    #else
        binary_op_init_common(cb_in0, cb_input_mask, cb_x);
    #endif

    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        index_mask_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {

            // mask input
            index_h_offset = index_b_offset + index_g_offset;
            reconfig_data_format_srcb(cb_in0, cb_input_mask);
            mul_tiles_init();
            cb_reserve_back(cb_x, block_hw);
            cb_wait_front(cb_input_mask, block_w);
            for (uint32_t i = 0; i < block_h; ++i) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        uint32_t index_mask = w + index_subblock_w_offset;
                        #ifdef TILIZE_IN
                        mul_tiles(cb_in, cb_input_mask, index, index_mask, w);
                        #else
                        mul_tiles(cb_in0, cb_input_mask, index, index_mask, w);
                        #endif
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, cb_x);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += per_core_N;
            }
            cb_push_back(cb_x, block_hw);
            reconfig_data_format_srcb(cb_input_mask, cb_scaler);

            // Partial-E[x]
            index_h_offset = 0;
            reduce_init_delta<false>();
            cb_reserve_back(cb_ex_partial, 1);
            tile_regs_acquire();
            cb_wait_front(cb_scaler, 1);
            cb_wait_front(cb_x, block_hw);
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_h_offset + w;
                    reduce_tile(cb_x, cb_scaler, index, scaler0, dst0);
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial);
            tile_regs_release();
            cb_push_back(cb_ex_partial, 1);
            reduce_revert_delta();

            if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
                reduce_init_delta<false>();
                cb_reserve_back(cb_ex_global, 1);
                cb_reserve_back(cb_ex, 1);
                tile_regs_acquire();
                cb_wait_front(cb_scaler_global, 1);
                cb_wait_front(cb_ex_external, 1);
                reduce_tile(cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex_global);
                tile_regs_release();
                reduce_revert_delta();
                cb_push_back(cb_ex_global, 1);
                cb_push_back(cb_ex, 1);
            }

            // x - E[x]
            sub_tiles_bcast_scalar_init_short();
            cb_reserve_back(cb_xmm, block_hw);
            cb_wait_front(cb_ex_global, 1);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset;
                        sub_tiles_bcast_scalar(cb_x, cb_ex_global, index, 0, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_xmm);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                cb_pop_front(cb_x, block_w);
            }
            cb_pop_front(cb_ex_global, 1);
            cb_push_back(cb_xmm, block_hw);

            // zero out the garbage values by mult mask again
            reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
            mul_tiles_init();
            cb_reserve_back(cb_x, block_hw);
            cb_wait_front(cb_xmm, block_hw);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset;
                        uint32_t index_mask = index;
                        mul_tiles(cb_xmm, cb_input_mask, index, index_mask, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, cb_x);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                cb_pop_front(cb_xmm, block_w);
            }
            cb_pop_front(cb_input_mask, block_w);
            cb_push_back(cb_x, block_hw);
            reconfig_data_format_srcb(cb_input_mask, cb_x);


            // (x - E[x])^2
            index_h_offset = 0;
            mul_tiles_init();
            cb_reserve_back(cb_xmm, block_hw);
            cb_wait_front(cb_x, block_hw);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        mul_tiles(cb_x, cb_x, index, index, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_xmm);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += block_w;
            }
            cb_push_back(cb_xmm, block_hw);

            // Partial-Var(x)
            index_h_offset = 0;
            reduce_init_delta<false>();
            cb_reserve_back(cb_ex_partial, 1);
            tile_regs_acquire();
            cb_wait_front(cb_xmm, block_hw);
            cb_wait_front(cb_scaler, 1);
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_h_offset + w;
                    reduce_tile(cb_xmm, cb_scaler, index, scaler0, dst0);
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial);
            tile_regs_release();
            cb_push_back(cb_ex_partial, 1);
            cb_pop_front(cb_xmm, block_hw);
            reduce_revert_delta();

            if constexpr(is_mcast_sender and num_cores_per_mcast_group > 1) {
                reduce_init_delta<false>();
                cb_reserve_back(cb_ex_global, 1);
                cb_reserve_back(cb_ex, 1);
                tile_regs_acquire();
                cb_wait_front(cb_scaler_global, 1);
                cb_wait_front(cb_ex_external, 1);
                reduce_tile(cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex_global);
                tile_regs_release();
                reduce_revert_delta();
                cb_push_back(cb_ex_global, 1);
                cb_push_back(cb_ex, 1);
            }

            // global reduce results
            cb_wait_front(cb_eps, 1);
            cb_wait_front(cb_ex_global, 1);
            cb_reserve_back(cb_ex2pe, 1);
            // (Var + eps)
            tile_regs_acquire();
            add_tiles_init();
            add_tiles(cb_ex_global, cb_eps, 0, 0, dst0);
            tile_regs_wait();
            // sqrt(Var + eps)
            sqrt_tile_init();
            sqrt_tile(dst0);
            tile_regs_wait();
            // 1/[sqrt(Var + eps)]
            recip_tile_init();
            recip_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            tile_regs_release();
            cb_push_back(cb_ex2pe, 1);
            cb_pop_front(cb_ex_global, 1);


            // (x - Ex) * 1/[sqrt(Var + eps)]
            index_h_offset = 0;
            mul_tiles_bcast_scalar_init_short();
            cb_reserve_back(cb_xmm, block_hw);
            cb_wait_front(cb_ex2pe, 1);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        mul_tiles_bcast_scalar(cb_x, cb_ex2pe, index, 0, w);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_xmm);
                    }
                    tile_regs_release();
                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += block_w;
            }
            cb_push_back(cb_xmm, block_hw);
            cb_pop_front(cb_ex2pe, 1);
            cb_pop_front(cb_x, block_hw);
            cb_wait_front(cb_xmm, block_hw);

            // add or copy with previous output results
            uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

            for (uint32_t w = 0; w < block_w_curr; ++w) {

                index_h_offset = index_b_offset + index_g_offset;
                uint32_t index_h1_offset = 0;

                if (copy_or_add == true) {
                    copy_tile_init();
                } else {
                    add_tiles_init();
                }

                for (uint32_t i = 0; i < block_h; ++i) {

                    tile_regs_acquire();
                    uint32_t index_xmm = w + index_h1_offset;
                    uint32_t index = w + index_h_offset;

                    if (copy_or_add == true) {
                        copy_tile(cb_xmm, index_xmm, dst0);
                    } else {
                        add_tiles(cb_out, cb_xmm, index, index_xmm, dst0);
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
            cb_pop_front(cb_xmm, block_hw);

            if constexpr(GROUP_SIZE_IS_POWER_OF_2) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w;
                    row_offset = num_cols_per_group;

                } else {
                    index_g_offset += block_w_minus_one;
                    row_offset += num_cols_per_group;

                }
            }
            else if constexpr(GROUP_SIZE_SMALLER_THAN_TILE_W) {
                if (row_offset == TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = num_cols_per_group;

                } else if (row_offset > TILE_WIDTH) {
                    index_g_offset += block_w_minus_one;
                    row_offset = row_offset + group_row_offset;

                } else {
                    row_offset += num_cols_per_group;
                }
            }
            else {
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

    cb_push_back(cb_out, per_core_MN);
    cb_pop_front(cb_in, per_core_MN);

    if constexpr(do_gamma) {
        index_h_offset = 0;
        mul_bcast_rows_init_short();
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
    }

    if constexpr(do_beta) {
        index_h_offset = 0;
        add_bcast_rows_init_short();
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
    }

    #ifdef UNTILIZE_OUT
    // untilize
    untilize_init_short(cb_untilize_in);
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
}
