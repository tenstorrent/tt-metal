// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes layernorm or rmsnorm, dependent on the RMSNORM define.
 * For layernorm it receives E(x**2) and E(x) and computes the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) and E(x) are contained in a two tile wide tensor containing E(x**2) and E(x) in the left most columns per
 * tile. For rmsnorm it receives E(x**2) and computes teh remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"

ALWI void ACQ() {
    acquire_dst();
}
ALWI void REL() {
    release_dst();
}

namespace NAMESPACE {
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(2);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t do_beta = get_compile_time_arg_val(4);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(5) == 1;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_stats = tt::CB::c_in1;

    constexpr uint32_t cb_eps = tt::CB::c_in4;
    constexpr uint32_t cb_reduce = tt::CB::c_in5;

    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr uint32_t cb_stats_reduced = tt::CB::c_intermed0;   // [E(x**2), E(x)]
    constexpr uint32_t cb_var_eps = tt::CB::c_intermed3;         // var + epsilon (or E(x**2) + epsilon)
    constexpr uint32_t cb_recip_sqrt_var = tt::CB::c_intermed4;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_x_normed = tt::CB::c_intermed6;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

    constexpr uint32_t cb_var = tt::CB::c_intermed2;  // E(x**2) - E(x)**2 or E(x**2)
#ifndef RMSNORM
    // Layernorm-specific CBs
    constexpr uint32_t cb_mean_squared = tt::CB::c_intermed1;  // E(x)**2
    constexpr uint32_t cb_x_minus_mean = tt::CB::c_intermed5;  // x - E(x)

    constexpr uint32_t cb_norm_x_input = cb_x_minus_mean;
    constexpr uint32_t stats_tile_stride = 2;
#else
    constexpr uint32_t cb_norm_x_input = cb_inp;
    constexpr uint32_t stats_tile_stride = 1;
#endif

    constexpr uint32_t cb_gamma = tt::CB::c_in2;
    constexpr uint32_t cb_beta = tt::CB::c_in3;
    uint32_t cb_times_gamma_out = cb_out;
    if constexpr (do_gamma and do_beta) {
        cb_times_gamma_out = tt::CB::c_intermed7;
    }

    binary_op_init_common(cb_inp, cb_inp, cb_stats_reduced);

    cb_wait_front(cb_reduce, 1);  // comes from the reader
    cb_wait_front(cb_eps, 1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        reconfig_data_format(cb_stats, cb_reduce);
        pack_reconfig_data_format(cb_stats_reduced);

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x0), sum(x1**2), sum(x1), ...]
         * RMSNorm packs mean(x**2) into cb_var. Layernorm just uses cb_stats_reduced.
         */
        reduce_init_delta<false>();
        cb_wait_front(cb_stats, stats_tiles_cols);
        cb_reserve_back(cb_stats_reduced, stats_tile_stride);
#ifdef RMSNORM
        cb_reserve_back(cb_var, 1);
#endif
        ACQ();
        // Reduce sum(x**2) first
        for (uint32_t i = 0; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile(cb_stats, cb_reduce, i, 0, 0);
        }
        pack_tile(0, cb_stats_reduced);

#ifndef RMSNORM
        // Reduce sum(x) next
        for (uint32_t i = 1; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile(cb_stats, cb_reduce, i, 0, 1);
        }
        pack_tile(1, cb_stats_reduced);
#else
        pack_tile(0, cb_var);
#endif
        REL();
        cb_push_back(cb_stats_reduced, stats_tile_stride);
        cb_pop_front(cb_stats, stats_tiles_cols);
#ifdef RMSNORM
        cb_push_back(cb_var, 1);
#endif

        reduce_revert_delta();

#ifndef RMSNORM
        /*
         * E[x]**2
         */
        reconfig_data_format(cb_stats_reduced, cb_stats_reduced);
        pack_reconfig_data_format(cb_mean_squared);
        mul_tiles_init();
        cb_reserve_back(cb_mean_squared, onetile);
        cb_wait_front(cb_stats_reduced, stats_tile_stride);
        ACQ();
        mul_tiles(cb_stats_reduced, cb_stats_reduced, 1, 1, 0);
        pack_tile(0, cb_mean_squared);
        REL();

        cb_push_back(cb_mean_squared, 1);

        /*
         * E[x**2] - E[x]**2
         */
        reconfig_data_format(cb_stats_reduced, cb_mean_squared);
        pack_reconfig_data_format(cb_var);
        sub_tiles_init();

        cb_reserve_back(cb_var, onetile);
        cb_wait_front(cb_mean_squared, 1);
        ACQ();
        sub_tiles(cb_stats_reduced, cb_mean_squared, 0, 0, 0);
        pack_tile(0, cb_var);
        REL();
        cb_push_back(cb_var, 1);
        cb_pop_front(cb_mean_squared, 1);

        /*
         * x - E[x]
         */
        reconfig_data_format(cb_inp, cb_stats_reduced);
        pack_reconfig_data_format(cb_x_minus_mean);
        sub_bcast_cols_init_short();
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_inp, blk);
            cb_reserve_back(cb_x_minus_mean, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                sub_tiles_bcast_cols(cb_inp, cb_stats_reduced, wtr, 1, wtr);
                pack_tile(wtr, cb_x_minus_mean);
            }
            REL();
            cb_push_back(cb_x_minus_mean, blk);
            cb_pop_front(cb_inp, blk);
        }
#endif

        // free up CBs
        cb_pop_front(cb_stats_reduced, stats_tile_stride);

        /*
         * 1/sqrt(var + eps)
         */
        cb_wait_front(cb_var, 1);
        cb_reserve_back(cb_recip_sqrt_var, 1);
        reconfig_data_format(cb_var, cb_eps);
        pack_reconfig_data_format(cb_recip_sqrt_var);

        add_tiles_init();
        ACQ();
        add_tiles(cb_var, cb_eps, 0, 0, 0);
        sqrt_tile_init();
        sqrt_tile(0);
        recip_tile_init();
        recip_tile(0);
        pack_tile(0, cb_recip_sqrt_var);
        REL();
        cb_push_back(cb_recip_sqrt_var, 1);
        cb_pop_front(cb_var, 1);

        /*
         * norm x
         * Layernorm: (X - E[X]) * 1/sqrt(Var(X) + eps)
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */

        reconfig_data_format(cb_norm_x_input, cb_recip_sqrt_var);
        pack_reconfig_data_format(cb_x_normed);
        mul_bcast_cols_init_short();
        cb_wait_front(cb_recip_sqrt_var, 1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_norm_x_input, blk);
            cb_reserve_back(cb_x_normed, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_cols(cb_norm_x_input, cb_recip_sqrt_var, wtr, 0, wtr);
                pack_tile(wtr, cb_x_normed);
            }
            REL();
            cb_push_back(cb_x_normed, blk);
            cb_pop_front(cb_norm_x_input, blk);
        }
        cb_pop_front(cb_recip_sqrt_var, 1);

        /*
         * x_normed * gamma
         */
        reconfig_data_format(cb_x_normed, cb_gamma);
        pack_reconfig_data_format(cb_times_gamma_out);
        cb_wait_front(cb_gamma, Wt);
        mul_bcast_rows_init_short();
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_x_normed, blk);
            cb_reserve_back(cb_times_gamma_out, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_rows(cb_x_normed, cb_gamma, wtr, wt + wtr, wtr);
                pack_tile(wtr, cb_times_gamma_out);
            }
            REL();
            cb_push_back(cb_times_gamma_out, blk);
            cb_pop_front(cb_x_normed, blk);
        }

        if constexpr (do_gamma and do_beta) {
            /*
             * x_normed * gamma + beta
             */
            reconfig_data_format(cb_times_gamma_out, cb_beta);
            pack_reconfig_data_format(cb_out);
            cb_wait_front(cb_beta, Wt);
            add_bcast_rows_init_short();
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                cb_wait_front(cb_times_gamma_out, blk);
                cb_reserve_back(cb_out, blk);
                ACQ();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    add_tiles_bcast_rows(cb_times_gamma_out, cb_beta, wtr, wt + wtr, wtr);
                    pack_tile(wtr, cb_out);
                }
                REL();
                cb_push_back(cb_out, blk);
                cb_pop_front(cb_times_gamma_out, blk);
            }
        }
    }
    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_reduce, 1);
    cb_pop_front(cb_gamma, Wt);
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
}  // namespace NAMESPACE
