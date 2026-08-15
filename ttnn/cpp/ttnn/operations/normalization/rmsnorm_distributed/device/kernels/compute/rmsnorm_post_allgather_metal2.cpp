// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm, dependent on the RMSNORM define.
 * For rmsnorm we receive E(x**2) and compute the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 *
 * Metal 2.0 fork of rmsnorm_post_allgather.cpp: same computation, with named kernel arguments and
 * named dataflow-buffer bindings instead of positional compile-time args and CB indices. The legacy
 * file beside this one still serves consumers that have not migrated.
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "experimental/kernel_args.h"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

// RMSNorm normalizes x directly, so the un-normalized input doubles as the normalization operand.
constexpr auto dfb_norm_x_input = dfb::inp;

// The normalized result goes straight to the output unless gamma still has to be applied to it.
// Only the buffers this build binds have handles, so the choice is made at the preprocessor.
#ifdef FUSE_GAMMA
constexpr auto normed_output_dfb = dfb::x_normed;
#else
constexpr auto normed_output_dfb = dfb::out;
#endif

// gamma's product feeds the beta stage when both are applied; otherwise it is already the output.
#if defined(FUSE_GAMMA) && defined(FUSE_BETA)
constexpr auto dfb_times_gamma_out = dfb::times_gamma_out;
#else
constexpr auto dfb_times_gamma_out = dfb::out;
#endif

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::blk);
    constexpr auto stats_tiles_cols = get_arg(args::stats_tiles_cols);
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dtype) == 1;
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::inp, dfb::inp, dfb::var);

    DataflowBuffer dfb_reduce(dfb::reduce);
    DataflowBuffer dfb_eps(dfb::eps);
    DataflowBuffer dfb_var(dfb::var);  // E(x**2)
    DataflowBuffer dfb_recip_sqrt_var(dfb::recip_sqrt_var);
    DataflowBuffer dfb_norm_x(dfb_norm_x_input);
    // Under FUSE_GAMMA this same buffer is the gamma stage's input, so one object drives both roles.
    DataflowBuffer dfb_normed_output(normed_output_dfb);
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb::gamma);
    DataflowBuffer dfb_times_gamma(dfb_times_gamma_out);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb::beta);
#endif
    // beta is applied only in the company of gamma, so the output buffer is driven directly only on
    // that combined path. Without gamma the normalized result is already packed into the output, and
    // dfb_normed_output is the handle for it.
#if defined(FUSE_GAMMA) && defined(FUSE_BETA)
    DataflowBuffer dfb_out(dfb::out);
#endif

    dfb_reduce.wait_front(1);  // comes from the reader
    dfb_eps.wait_front(1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        /*
         * Reduce stats input.
         * dfb::stats = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into dfb::var for the rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles buffer lifecycle.
         */
        compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, dfb::stats, dfb::reduce, dfb::var>(
            compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols));

        /*
         * 1/sqrt(var + eps)
         */
        dfb_var.wait_front(1);
        dfb_recip_sqrt_var.reserve_back(1);
        reconfig_data_format(dfb::var, dfb::eps);
        pack_reconfig_data_format(dfb::recip_sqrt_var);

        add_init(dfb::var, dfb::eps);
        ACQ();
        add_tiles(dfb::var, dfb::eps, 0, 0, 0);
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(0);
        pack_tile(0, dfb::recip_sqrt_var);
        REL();
        dfb_recip_sqrt_var.push_back(1);
        dfb_var.pop_front(1);

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */
        reconfig_data_format(dfb_norm_x_input, dfb::recip_sqrt_var);
        pack_reconfig_data_format(normed_output_dfb);
        mul_bcast_cols_init(dfb_norm_x_input, dfb::recip_sqrt_var);
        dfb_recip_sqrt_var.wait_front(1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            dfb_norm_x.wait_front(blk);
            dfb_normed_output.reserve_back(blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_cols(dfb_norm_x_input, dfb::recip_sqrt_var, wtr, 0, wtr);
                pack_tile(wtr, normed_output_dfb);
            }
            REL();
            dfb_normed_output.push_back(blk);
            dfb_norm_x.pop_front(blk);
        }
        dfb_recip_sqrt_var.pop_front(1);

#ifdef FUSE_GAMMA
        /*
         * x_normed * gamma
         */
        reconfig_data_format(dfb::x_normed, dfb::gamma);
        pack_reconfig_data_format(dfb_times_gamma_out);
        dfb_gamma.wait_front(Wt);
        mul_bcast_rows_init(dfb::x_normed, dfb::gamma);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            dfb_normed_output.wait_front(blk);
            dfb_times_gamma.reserve_back(blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_rows(dfb::x_normed, dfb::gamma, wtr, wt + wtr, wtr);
                pack_tile(wtr, dfb_times_gamma_out);
            }
            REL();
            dfb_times_gamma.push_back(blk);
            dfb_normed_output.pop_front(blk);
        }

#ifdef FUSE_BETA
        /*
         * x_normed * gamma + beta
         */
        reconfig_data_format(dfb_times_gamma_out, dfb::beta);
        pack_reconfig_data_format(dfb::out);
        dfb_beta.wait_front(Wt);
        add_bcast_rows_init(dfb_times_gamma_out, dfb::beta);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            dfb_times_gamma.wait_front(blk);
            dfb_out.reserve_back(blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                add_tiles_bcast_rows(dfb_times_gamma_out, dfb::beta, wtr, wt + wtr, wtr);
                pack_tile(wtr, dfb::out);
            }
            REL();
            dfb_out.push_back(blk);
            dfb_times_gamma.pop_front(blk);
        }
#endif
#endif
    }
    dfb_eps.pop_front(1);
    dfb_reduce.pop_front(1);
#ifdef FUSE_GAMMA
    dfb_gamma.pop_front(Wt);
#endif
#ifdef FUSE_BETA
    dfb_beta.pop_front(Wt);
#endif
}
