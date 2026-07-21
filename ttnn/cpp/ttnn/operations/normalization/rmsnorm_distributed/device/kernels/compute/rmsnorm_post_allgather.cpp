// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "api/dataflow/circular_buffer.h"
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
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(7) == 1;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;

    constexpr uint32_t cb_eps_idx = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduce_idx = tt::CBIndex::c_5;

    constexpr uint32_t cb_out_idx = tt::CBIndex::c_14;

    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;          // var + epsilon (or E(x**2) + epsilon)
    constexpr uint32_t cb_recip_sqrt_var_idx = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_x_normed_idx =
        tt::CBIndex::c_12;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

    constexpr uint32_t cb_var_idx = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)
    constexpr uint32_t cb_norm_x_input_idx = cb_inp;

    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta_idx = tt::CBIndex::c_3;
    uint32_t cb_times_gamma_out_idx = cb_out_idx;
    if constexpr (do_gamma and do_beta) {
        cb_times_gamma_out_idx = tt::CBIndex::c_13;
    }

    binary_op_init_common(cb_inp, cb_inp, cb_var_idx);

    CircularBuffer cb_reduce(cb_reduce_idx);
    CircularBuffer cb_eps(cb_eps_idx);
    CircularBuffer cb_var(cb_var_idx);
    CircularBuffer cb_recip_sqrt_var(cb_recip_sqrt_var_idx);
    CircularBuffer cb_norm_x_input(cb_norm_x_input_idx);
    CircularBuffer cb_gamma(cb_gamma_idx);
    CircularBuffer cb_x_normed(cb_x_normed_idx);
    CircularBuffer cb_times_gamma_out(cb_times_gamma_out_idx);
    CircularBuffer cb_beta(cb_beta_idx);
    CircularBuffer cb_out(cb_out_idx);

    cb_reduce.wait_front(1);  // comes from the reader
    cb_eps.wait_front(1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into cb_var_idx for rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles CB lifecycle.
         */
        compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, cb_stats, cb_reduce_idx, cb_var_idx>(
            compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols));

        /*
         * 1/sqrt(var + eps)
         */
        cb_var.wait_front(1);
        cb_recip_sqrt_var.reserve_back(1);
        reconfig_data_format(cb_var_idx, cb_eps_idx);
        pack_reconfig_data_format(cb_recip_sqrt_var_idx);

        add_tiles_init(cb_var_idx, cb_eps_idx);
        ACQ();
        add_tiles(cb_var_idx, cb_eps_idx, 0, 0, 0);
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(0);
        pack_tile(0, cb_recip_sqrt_var_idx);
        REL();
        cb_recip_sqrt_var.push_back(1);
        cb_var.pop_front(1);

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */

        uint32_t normed_output_cb_idx = cb_x_normed_idx;
        if constexpr (!do_gamma) {
            normed_output_cb_idx = cb_out_idx;
        }
        CircularBuffer normed_output_cb(normed_output_cb_idx);

        reconfig_data_format(cb_norm_x_input_idx, cb_recip_sqrt_var_idx);
        pack_reconfig_data_format(normed_output_cb_idx);
        mul_bcast_cols_init_short(cb_norm_x_input_idx, cb_recip_sqrt_var_idx);
        cb_recip_sqrt_var.wait_front(1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_norm_x_input.wait_front(blk);
            normed_output_cb.reserve_back(blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_cols(cb_norm_x_input_idx, cb_recip_sqrt_var_idx, wtr, 0, wtr);
                pack_tile(wtr, normed_output_cb_idx);
            }
            REL();
            normed_output_cb.push_back(blk);
            cb_norm_x_input.pop_front(blk);
        }
        cb_recip_sqrt_var.pop_front(1);

        if constexpr (do_gamma) {
            /*
             * x_normed * gamma
             */
            reconfig_data_format(cb_x_normed_idx, cb_gamma_idx);
            pack_reconfig_data_format(cb_times_gamma_out_idx);
            cb_gamma.wait_front(Wt);
            mul_bcast_rows_init_short(cb_x_normed_idx, cb_gamma_idx);
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                cb_x_normed.wait_front(blk);
                cb_times_gamma_out.reserve_back(blk);
                ACQ();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    mul_tiles_bcast_rows(cb_x_normed_idx, cb_gamma_idx, wtr, wt + wtr, wtr);
                    pack_tile(wtr, cb_times_gamma_out_idx);
                }
                REL();
                cb_times_gamma_out.push_back(blk);
                cb_x_normed.pop_front(blk);
            }

            if constexpr (do_beta) {
                /*
                 * x_normed * gamma + beta
                 */
                reconfig_data_format(cb_times_gamma_out_idx, cb_beta_idx);
                pack_reconfig_data_format(cb_out_idx);
                cb_beta.wait_front(Wt);
                add_bcast_rows_init_short(cb_times_gamma_out_idx, cb_beta_idx);
                for (uint32_t wt = 0; wt < Wt; wt += blk) {
                    cb_times_gamma_out.wait_front(blk);
                    cb_out.reserve_back(blk);
                    ACQ();
                    for (uint32_t wtr = 0; wtr < blk; wtr++) {
                        add_tiles_bcast_rows(cb_times_gamma_out_idx, cb_beta_idx, wtr, wt + wtr, wtr);
                        pack_tile(wtr, cb_out_idx);
                    }
                    REL();
                    cb_out.push_back(blk);
                    cb_times_gamma_out.pop_front(blk);
                }
            }
        }
    }
    cb_eps.pop_front(1);
    cb_reduce.pop_front(1);
    if constexpr (do_gamma) {
        cb_gamma.pop_front(Wt);
    }
    if constexpr (do_beta) {
        cb_beta.pop_front(Wt);
    }
}
