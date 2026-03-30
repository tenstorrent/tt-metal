// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/circular_buffer.h"

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

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

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

    experimental::CircularBuffer cb_beta(cb_beta_id);
    experimental::CircularBuffer cb_eps(cb_eps_id);
    experimental::CircularBuffer cb_ex(cb_ex_id);
    experimental::CircularBuffer cb_ex2pe(cb_ex2pe_id);
    experimental::CircularBuffer cb_ex_external(cb_ex_external_id);
    experimental::CircularBuffer cb_ex_global(cb_ex_global_id);
    experimental::CircularBuffer cb_ex_partial(cb_ex_partial_id);
    experimental::CircularBuffer cb_gamma(cb_gamma_id);
    experimental::CircularBuffer cb_in(cb_in_id);
    experimental::CircularBuffer cb_in_negative_mask(cb_in_negative_mask_id);
    experimental::CircularBuffer cb_inbeta(cb_inbeta_id);
    experimental::CircularBuffer cb_input_mask(cb_input_mask_id);
    experimental::CircularBuffer cb_ones(cb_ones_id);
    experimental::CircularBuffer cb_out(cb_out_id);
    experimental::CircularBuffer cb_outbeta(cb_outbeta_id);
    experimental::CircularBuffer cb_outgamma(cb_outgamma_id);
    experimental::CircularBuffer cb_scaler(cb_scaler_id);
    experimental::CircularBuffer cb_scaler_global(cb_scaler_global_id);
    experimental::CircularBuffer cb_x(cb_x_id);

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
    binary_op_init_common(cb_in0_id, cb_input_mask_id, cb_x_id);
#endif

    index_b_offset = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        index_g_offset = 0;
        index_mask_offset = 0;
        for (uint32_t g = 0; g < group; ++g) {
            // mask input
            index_h_offset = index_b_offset + index_g_offset;
            reconfig_data_format_srcb(cb_in0_id, cb_input_mask_id);
            mul_tiles_init(cb_in0_id, cb_input_mask_id);
            cb_x.reserve_back(block_hw);
            cb_input_mask.wait_front(block_w);
            for (uint32_t i = 0; i < block_h; ++i) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
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
            reconfig_data_format_srcb(cb_input_mask_id, cb_ones_id);

            // Partial-E[x]
            index_h_offset = 0;
            mul_tiles_init(cb_x_id, cb_ones_id);
            cb_ex2pe.reserve_back(1);
            tile_regs_acquire();
            cb_x.wait_front(block_hw);
            cb_ones.wait_front(1);

            index_h_offset = 0;
            // Accomulate into dest directly by using mul_tiles (tile * 1 is accomulated into dest at the moment)
            // Alternative is to use reduce_tile multiple times, but this showed to be more precise and faster.
            for (uint32_t h = 0; h < block_h; ++h) {
                for (uint32_t w = 0; w < block_w; ++w) {
                    uint32_t index = index_h_offset + w;
                    mul_tiles(cb_x_id, cb_ones_id, index, 0, dst0);
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe_id);
            tile_regs_release();
            cb_ex2pe.push_back(1);
            tile_regs_acquire();
            reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2pe_id, cb_scaler_id, cb_ex_partial_id);
            cb_ex_partial.reserve_back(1);
            cb_scaler.wait_front(1);
            cb_ex2pe.wait_front(1);
            // reduce only one final tile
            reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2pe_id, cb_scaler_id, 0, scaler0, dst0);
            cb_ex2pe.pop_front(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial_id);
            tile_regs_release();
            cb_ex_partial.push_back(1);
            reduce_uninit<FP32_DEST_ACC>();

            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(
                    cb_ex_external_id, cb_scaler_global_id, cb_ex_global_id);
                cb_ex_global.reserve_back(1);
                cb_ex.reserve_back(1);
                tile_regs_acquire();
                cb_scaler_global.wait_front(1);
                cb_ex_external.wait_front(1);
                reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(
                    cb_ex_external_id, cb_scaler_global_id, 0, scaler0, dst0);
                cb_ex_external.pop_front(1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex_global_id);
                tile_regs_release();
                reduce_uninit<FP32_DEST_ACC>();
                cb_ex_global.push_back(1);
                cb_ex.push_back(1);
            }
            // x - E[x]
            sub_tiles_bcast_scalar_init_short(cb_x_id, cb_ex_global_id);

            cb_ex_global.wait_front(1);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset;
                        sub_tiles_bcast_scalar(cb_x_id, cb_ex_global_id, index, 0, w);
                    }
                    tile_regs_commit();
                    cb_x.pop_front(subblock_w);
                    cb_x.reserve_back(subblock_w);
                    tile_regs_wait();
                    for (uint32_t k = 0; k < subblock_w; k++) {
                        pack_tile(k, cb_x_id);
                    }
                    cb_x.push_back(subblock_w);
                    tile_regs_release();
                    cb_x.wait_front(block_hw);
                }
            }
            cb_ex_global.pop_front(1);

            reconfig_data_format_srcb(cb_ex_global_id, cb_input_mask_id);
            mul_tiles_init(cb_x_id, cb_input_mask_id);
            cb_x.wait_front(block_hw);

            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; ++j) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; ++w) {
                        uint32_t index = w + index_subblock_w_offset;
                        uint32_t index_mask = index;
                        mul_tiles(cb_x_id, cb_input_mask_id, index, index_mask, w);
                    }
                    tile_regs_commit();

                    cb_x.pop_front(subblock_w);
                    cb_x.reserve_back(subblock_w);

                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; ++i) {
                        pack_tile(i, cb_x_id);
                    }
                    cb_x.push_back(subblock_w);
                    tile_regs_release();
                }
            }
            cb_input_mask.pop_front(block_w);
            reconfig_data_format_srcb(cb_input_mask_id, cb_x_id);

            // (x - E[x])^2
            index_h_offset = 0;
            mul_tiles_init(cb_x_id, cb_x_id);
            cb_x.wait_front(block_hw);

            tile_regs_acquire();
            cb_ex2pe.reserve_back(1);

            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset + index_h_offset;
                        mul_tiles(cb_x_id, cb_x_id, index, index, dst0);
                    }

                    index_subblock_w_offset += subblock_w;
                }
                index_h_offset += block_w;
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe_id);
            tile_regs_release();
            cb_ex2pe.push_back(1);

            cb_ex_partial.reserve_back(1);
            cb_scaler.wait_front(1);
            cb_ex2pe.wait_front(1);

            reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2pe_id, cb_scaler_id, cb_ex_partial_id);

            tile_regs_acquire();
            reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2pe_id, cb_scaler_id, 0, scaler0, dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_partial_id);
            tile_regs_release();
            cb_ex_partial.push_back(1);

            reduce_uninit<FP32_DEST_ACC>();

            cb_ex2pe.pop_front(1);
            cb_ex_partial.wait_front(1);
            if constexpr (is_mcast_sender and num_cores_per_mcast_group > 1) {
                reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(
                    cb_ex_external_id, cb_scaler_global_id, cb_ex_global_id);
                cb_ex_global.reserve_back(1);
                cb_ex.reserve_back(1);
                tile_regs_acquire();
                cb_scaler_global.wait_front(1);
                cb_ex_external.wait_front(1);
                reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(
                    cb_ex_external_id, cb_scaler_global_id, 0, scaler0, dst0);
                cb_ex_external.pop_front(1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex_global_id);
                tile_regs_release();
                reduce_uninit<FP32_DEST_ACC>();
                cb_ex_global.push_back(1);
                cb_ex.push_back(1);
            }

            // global reduce results

            cb_eps.wait_front(1);
            cb_ex_global.wait_front(1);
            cb_ex2pe.reserve_back(1);

            // (Var + eps)
            tile_regs_acquire();
            add_tiles_init(cb_ex_global_id, cb_eps_id);
            add_tiles(cb_ex_global_id, cb_eps_id, 0, 0, dst0);
            // 1/[sqrt(Var + eps)]
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe_id);
            tile_regs_release();
            cb_ex2pe.push_back(1);
            cb_ex_global.pop_front(1);
            //  (x - Ex) * 1/[sqrt(Var + eps)]
            index_h_offset = 0;
            mul_tiles_bcast_scalar_init_short(cb_x_id, cb_ex2pe_id);

            cb_ex2pe.wait_front(1);
            for (uint32_t i = 0; i < block_h; i++) {
                index_subblock_w_offset = 0;
                for (uint32_t j = 0; j < num_subblocks_w; j++) {
                    tile_regs_acquire();
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t index = w + index_subblock_w_offset;
                        mul_tiles_bcast_scalar(cb_x_id, cb_ex2pe_id, index, 0, w);
                    }
                    tile_regs_commit();
                    cb_x.pop_front(subblock_w);
                    cb_x.reserve_back(subblock_w);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < subblock_w; i++) {
                        pack_tile(i, cb_x_id);
                    }
                    cb_x.push_back(subblock_w);
                    tile_regs_release();
                }
            }
            cb_ex2pe.pop_front(1);
            cb_x.wait_front(block_hw);
            //  add or copy with previous output results
            uint32_t block_w_curr = index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w;

            // if we are using negative mask, we are overlapping tilized in and out, otherwise they are 2 separate
            // buffers.
            if constexpr (use_negative_mask == false) {
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    if (copy_or_add == true) {
                        copy_tile_init(cb_x_id);
                    } else {
                        add_tiles_init(cb_out_id, cb_x_id);
                    }

                    for (uint32_t i = 0; i < block_h; ++i) {
                        tile_regs_acquire();
                        uint32_t index_x = w + index_h1_offset;
                        uint32_t index = w + index_h_offset;

                        if (copy_or_add == true) {
                            copy_tile(cb_x_id, index_x, dst0);
                        } else {
                            add_tiles(cb_out_id, cb_x_id, index, index_x, dst0);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, cb_out_id, index);
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
                cb_in_negative_mask.wait_front(block_w);
                reconfig_data_format_srcb(cb_x_id, cb_in_negative_mask_id);
                mul_tiles_init(cb_in_id, cb_in_negative_mask_id);

                for (uint32_t w = 0; w < block_w_curr; w++) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    for (uint32_t i = 0; i < block_h; i++) {
                        tile_regs_acquire();
                        uint32_t index_in = w + index_h_offset;
                        uint32_t index_mask = w;

                        mul_tiles(cb_in_id, cb_in_negative_mask_id, index_in, index_mask, dst0);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile<true>(dst0, cb_in_id, index_in);
                        tile_regs_release();

                        index_h_offset += per_core_N;
                    }
                }

                reconfig_data_format_srcb(cb_in_negative_mask_id, cb_x_id);
                add_tiles_init(cb_in_id, cb_x_id);
                // data in cb_x_id has valid data only for current group
                // cb_in_id has cleared data for that group
                // just add them together
                for (uint32_t w = 0; w < block_w_curr; ++w) {
                    index_h_offset = index_b_offset + index_g_offset;
                    uint32_t index_h1_offset = 0;

                    for (uint32_t i = 0; i < block_h; ++i) {
                        tile_regs_acquire();
                        uint32_t index_x = w + index_h1_offset;
                        uint32_t index = w + index_h_offset;

                        add_tiles(cb_in_id, cb_x_id, index, index_x, dst0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(dst0, cb_in_id, index);
                        tile_regs_release();

                        index_h_offset += per_core_N;
                        index_h1_offset += block_w;
                    }
                }
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
        index_h_offset = 0;
        if constexpr (use_negative_mask == false) {
            mul_bcast_rows_init_short(cb_out_id, cb_gamma_id);
            cb_outgamma.reserve_back(per_core_MN);
            cb_gamma.wait_front(per_core_N);
            for (uint32_t i = 0; i < per_core_M; ++i) {
                for (uint32_t j = 0; j < per_core_N; ++j) {
                    tile_regs_acquire();
                    uint32_t index = j + index_h_offset;
                    mul_tiles_bcast_rows(cb_out_id, cb_gamma_id, index, j, dst0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, cb_outgamma_id);
                    tile_regs_release();
                }
                index_h_offset += per_core_N;
            }

            cb_outgamma.push_back(per_core_MN);
            cb_out.pop_front(per_core_MN);
            cb_outgamma.wait_front(per_core_MN);
        } else {
            // cb in has data required for gamma, so we do it inplace
            mul_bcast_rows_init_short(cb_in_id, cb_gamma_id);
            cb_gamma.wait_front(per_core_N);
            cb_in.wait_front(per_core_MN);
            for (uint32_t i = 0; i < per_core_M; i++) {
                for (uint32_t j = 0; j < per_core_N; j++) {
                    tile_regs_acquire();
                    mul_tiles_bcast_rows(cb_in_id, cb_gamma_id, 0, j, dst0);
                    tile_regs_commit();
                    cb_in.pop_front(1);
                    cb_in.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_in_id);
                    cb_in.push_back(1);
                    tile_regs_release();
                }
            }
        }
    }

    if constexpr (do_beta) {
        if constexpr (use_negative_mask == false) {
            index_h_offset = 0;
            add_bcast_rows_init_short(cb_inbeta_id, cb_beta_id);
            cb_outbeta.reserve_back(per_core_MN);
            cb_beta.wait_front(per_core_N);
            for (uint32_t i = 0; i < per_core_M; ++i) {
                for (uint32_t j = 0; j < per_core_N; ++j) {
                    tile_regs_acquire();
                    uint32_t index = j + index_h_offset;
                    add_tiles_bcast_rows(cb_inbeta_id, cb_beta_id, index, j, dst0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(dst0, cb_outbeta_id);
                    tile_regs_release();
                }
                index_h_offset += per_core_N;
            }
            cb_outbeta.push_back(per_core_MN);
            cb_inbeta.pop_front(per_core_MN);
            cb_outbeta.wait_front(per_core_MN);
        } else {
            // cb_in_id has data required for beta, so we do it inplace
            add_bcast_rows_init_short(cb_in_id, cb_beta_id);
            cb_beta.wait_front(per_core_N);
            cb_in.wait_front(per_core_MN);
            for (uint32_t i = 0; i < per_core_M; i++) {
                for (uint32_t j = 0; j < per_core_N; j++) {
                    tile_regs_acquire();
                    add_tiles_bcast_rows(cb_in_id, cb_beta_id, 0, j, dst0);
                    tile_regs_commit();
                    cb_in.pop_front(1);
                    cb_in.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(dst0, cb_in_id);
                    cb_in.push_back(1);
                    tile_regs_release();
                }
            }
        }
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
