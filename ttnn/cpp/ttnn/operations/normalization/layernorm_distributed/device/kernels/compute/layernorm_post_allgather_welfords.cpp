// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes layernorm or rmsnorm, dependent on the RMSNORM define.
 * For layernorm it receives E(x**2) and E(x) and computes the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) and E(x) are contained in a two tile wide tensor containing E(x**2) and E(x) in the left most columns per
 * tile. For rmsnorm it receives E(x**2) and computes the remaining normalization based on gamma, beta and epsilon.
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
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
#include "debug/dprint_pages.h"
#include "dprint_tensix.h"
// read dest reg
#include "debug/dprint.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    DPRINT << "W: " << W << ENDL();
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(3) / 2;
    constexpr uint32_t do_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t do_beta = get_compile_time_arg_val(5);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(6) == 1;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;

    constexpr uint32_t cb_eps = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_5;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_6;    // [E(x**2), E(x)]
    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;          // var + epsilon (or E(x**2) + epsilon)
    constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_x_normed = tt::CBIndex::c_12;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

    constexpr uint32_t cb_var = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)
#ifndef RMSNORM
    // Layernorm-specific CBs
    constexpr uint32_t cb_mean_squared = tt::CBIndex::c_7;   // E(x)**2
    constexpr uint32_t cb_x_minus_mean = tt::CBIndex::c_11;  // x - E(x)

    constexpr uint32_t cb_norm_x_input = cb_x_minus_mean;
    constexpr uint32_t stats_tile_stride = 2;
#else
    constexpr uint32_t cb_norm_x_input = cb_inp;
    constexpr uint32_t stats_tile_stride = 1;
#endif

    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    uint32_t cb_times_gamma_out = cb_out;
    if constexpr (do_gamma and do_beta) {
        cb_times_gamma_out = tt::CBIndex::c_13;
    }

    DPRINT << "pre_bin init " << ENDL();
    binary_op_init_common(cb_inp, cb_inp, cb_stats_reduced);

    cb_wait_front(cb_reduce, 1);  // comes from the reader
    cb_wait_front(cb_eps, 1);     // comes from the reader
    DPRINT << "post_bin init " << ENDL();

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        // combine_welford_partials()
        DPRINT << "pre_combine " << ENDL();
        norm::kernel_util::compute::combine_welford_partials(
            cb_stats,
            cb_stats_reduced,
            stats_tiles_cols,
            [&](uint32_t b) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        cb_push_back(cb_stats_reduced, 2);
        cb_wait_front(cb_stats_reduced, 2);
        UNPACK(tt::compute::common::print_full_tile(cb_stats_reduced, 0, true));
        UNPACK(tt::compute::common::print_full_tile(cb_stats_reduced, 1, true));
        DPRINT << "here " << ENDL();
        /*
         * x - E[x]
         */
        reconfig_data_format(cb_inp, cb_stats_reduced);
        pack_reconfig_data_format(cb_x_minus_mean);
        sub_bcast_cols_init_short(cb_inp, cb_stats_reduced);
        cb_wait_front(cb_stats_reduced, 1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_inp, blk);
            cb_reserve_back(cb_x_minus_mean, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                sub_tiles_bcast_cols(cb_inp, cb_stats_reduced, wtr, 0, wtr);
                pack_tile(wtr, cb_x_minus_mean);
            }
            REL();
            cb_push_back(cb_x_minus_mean, blk);
            cb_pop_front(cb_inp, blk);
        }
        /*
         * 1/sqrt(var + eps)
         */
        cb_wait_front(cb_stats_reduced, 2);
        cb_reserve_back(cb_recip_sqrt_var, 1);
        reconfig_data_format(cb_stats_reduced, cb_eps);
        pack_reconfig_data_format(cb_recip_sqrt_var);

        add_tiles_init(cb_stats_reduced, cb_eps);
        ACQ();
        add_tiles(cb_stats_reduced, cb_eps, 1, 0, 0);
        rsqrt_tile_init<true>();
        rsqrt_tile<true>(0);
        pack_tile(0, cb_recip_sqrt_var);
        REL();
        cb_push_back(cb_recip_sqrt_var, 1);

        // free up CBs
        cb_pop_front(cb_stats_reduced, stats_tile_stride);

        /*
         * norm x
         * Layernorm: (X - E[X]) * 1/sqrt(Var(X) + eps)
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
    cb_pop_front(cb_gamma, Wt);
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
}  // namespace NAMESPACE
