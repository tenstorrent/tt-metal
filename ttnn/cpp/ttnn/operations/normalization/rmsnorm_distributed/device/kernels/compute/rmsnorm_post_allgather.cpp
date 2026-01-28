// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm, dependent on the RMSNORM define.
 * For rmsnorm we receive E(x**2) and compute the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(2);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t do_beta = get_compile_time_arg_val(4);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(5) == 1;
    // Note: get_compile_time_arg_val(6) is FLOAT32_REDUCTION - unused after library migration
    // Library auto-detects FP32 from ENABLE_FP32_DEST_ACC define
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(7) == 1;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;

    constexpr uint32_t cb_eps = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_5;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    // Note: cb_stats_reduced (c_6) unused after library migration - reduce outputs directly to cb_var
    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;          // var + epsilon (or E(x**2) + epsilon)
    constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_x_normed = tt::CBIndex::c_12;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

    constexpr uint32_t cb_var = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)
    constexpr uint32_t cb_norm_x_input = cb_inp;
    // Note: stats_tile_stride unused after library migration - library handles contiguous access

    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    uint32_t cb_times_gamma_out = cb_out;
    if constexpr (do_gamma and do_beta) {
        cb_times_gamma_out = tt::CBIndex::c_13;
    }

    binary_op_init_common(cb_inp, cb_inp, cb_var);

    cb_wait_front(cb_reduce, 1);  // comes from the reader
    cb_wait_front(cb_eps, 1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into cb_var for rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles CB lifecycle.
         */
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_stats, cb_reduce, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols));

        /*
         * 1/sqrt(var + eps)
         */
        cb_wait_front(cb_var, 1);
        cb_reserve_back(cb_recip_sqrt_var, 1);
        reconfig_data_format(cb_var, cb_eps);
        pack_reconfig_data_format(cb_recip_sqrt_var);

        add_tiles_init(cb_var, cb_eps);
        ACQ();
        add_tiles(cb_var, cb_eps, 0, 0, 0);
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(0);
        pack_tile(0, cb_recip_sqrt_var);
        REL();
        cb_push_back(cb_recip_sqrt_var, 1);
        cb_pop_front(cb_var, 1);

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */

        uint32_t normed_output_cb = cb_x_normed;
        if constexpr (!do_gamma) {
            normed_output_cb = cb_out;
        }

        reconfig_data_format(cb_norm_x_input, cb_recip_sqrt_var);
        pack_reconfig_data_format(normed_output_cb);
        mul_bcast_cols_init_short(cb_norm_x_input, cb_recip_sqrt_var);
        cb_wait_front(cb_recip_sqrt_var, 1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_norm_x_input, blk);
            cb_reserve_back(normed_output_cb, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_cols(cb_norm_x_input, cb_recip_sqrt_var, wtr, 0, wtr);
                pack_tile(wtr, normed_output_cb);
            }
            REL();
            cb_push_back(normed_output_cb, blk);
            cb_pop_front(cb_norm_x_input, blk);
        }
        cb_pop_front(cb_recip_sqrt_var, 1);

        if constexpr (do_gamma) {
            /*
             * x_normed * gamma
             */
            reconfig_data_format(cb_x_normed, cb_gamma);
            pack_reconfig_data_format(cb_times_gamma_out);
            cb_wait_front(cb_gamma, Wt);
            mul_bcast_rows_init_short(cb_x_normed, cb_gamma);
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

            if constexpr (do_beta) {
                /*
                 * x_normed * gamma + beta
                 */
                reconfig_data_format(cb_times_gamma_out, cb_beta);
                pack_reconfig_data_format(cb_out);
                cb_wait_front(cb_beta, Wt);
                add_bcast_rows_init_short(cb_times_gamma_out, cb_beta);
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
    }
    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_reduce, 1);
    if constexpr (do_gamma) {
        cb_pop_front(cb_gamma, Wt);
    }
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
