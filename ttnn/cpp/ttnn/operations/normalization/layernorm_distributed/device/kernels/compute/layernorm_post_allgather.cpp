
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "chain_llk.hpp"

constexpr auto dfb_norm_x_input = dfb::x_minus_mean;  // x - E(x)
constexpr uint32_t stats_tile_stride = 2;

struct x_minus_mean_node {
    static constexpr LLK_Node node{
        .llk_init = sub_bcast_cols_init,
        .llk = FN_compute(sub_tiles_bcast_cols),
        .DFB_A = dfb::inp,
        .DFB_B = dfb::stats_reduced,
        .DFB_OUT = dfb::x_minus_mean,
        .fixed_DFB_B_index = 1,
        .fixed_dest_reg = 0xFFFF,
    };
};

// The normalized result goes straight to the output unless gamma or beta still has to be applied to
// it. Only the buffers this build binds have handles, so the choice is made at the preprocessor.
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
constexpr auto normed_output_dfb = dfb::x_normed;
#else
constexpr auto normed_output_dfb = dfb::out;
#endif
struct normed_output_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_cols_init,
        .llk = FN_compute(mul_tiles_bcast_cols),
        .DFB_A = dfb_norm_x_input,
        .DFB_B = dfb::recip_sqrt_var,
        .DFB_OUT = normed_output_dfb,
        .fixed_DFB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
    };
};

constexpr auto Wt_file_scope = get_arg(args::Wt);
constexpr auto dfb_length_file_scope = get_arg(args::dfb_length);
// When the whole row fits in one pass, gamma/beta tiles are re-read across iterations rather than
// popped, so the B operand is indexed from the running tile offset instead of the loop counter.
constexpr uint32_t pop_gamma_beta = (Wt_file_scope == dfb_length_file_scope) ? 0xDDDD : 0xFFFF;

// gamma's product feeds the beta stage when both are applied; otherwise it is already the output.
#if defined(FUSE_GAMMA) && defined(FUSE_BETA)
constexpr auto dfb_times_gamma_out = dfb::times_gamma_out;
#else
constexpr auto dfb_times_gamma_out = dfb::out;
#endif
#ifdef FUSE_GAMMA
struct gamma_optional_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_rows_init,
        .llk = FN_compute(mul_tiles_bcast_rows),
        .DFB_A = dfb::x_normed,
        .DFB_B = dfb::gamma,
        .DFB_OUT = dfb_times_gamma_out,
        .fixed_DFB_B_index = pop_gamma_beta,
        .fixed_dest_reg = 0xFFFF,
    };
};
#endif

#ifdef FUSE_GAMMA
constexpr auto dfb_in_beta = dfb_times_gamma_out;
#else
constexpr auto dfb_in_beta = normed_output_dfb;
#endif
#ifdef FUSE_BETA
struct beta_optional_node {
    static constexpr LLK_Node node{
        .llk_init = add_bcast_rows_init,
        .llk = FN_compute(add_tiles_bcast_rows),
        .DFB_A = dfb_in_beta,
        .DFB_B = dfb::beta,
        .DFB_OUT = dfb::out,
        .fixed_DFB_B_index = pop_gamma_beta,
        .fixed_dest_reg = 0xFFFF,
    };
};
#endif

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::blk);
    constexpr auto stats_tiles_cols = get_arg(args::stats_tiles_cols);
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dtype) == 1;
    constexpr bool FLOAT32_REDUCTION = get_arg(args::float32_reduction) == 1;
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1;
    constexpr auto dfb_length = get_arg(args::dfb_length);

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::inp, dfb::inp, dfb::stats_reduced);

    DataflowBuffer dfb_reduce(dfb::reduce);
    DataflowBuffer dfb_eps(dfb::eps);
    DataflowBuffer dfb_stats(dfb::stats);
    DataflowBuffer dfb_stats_reduced(dfb::stats_reduced);
    DataflowBuffer dfb_mean_squared(dfb::mean_squared);  // E(x)**2
    DataflowBuffer dfb_var(dfb::var);                    // E(x**2) - E(x)**2
    DataflowBuffer dfb_recip_sqrt_var(dfb::recip_sqrt_var);

    dfb_reduce.wait_front(1);  // comes from the reader
    dfb_eps.wait_front(1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        reconfig_data_format(dfb::reduce, dfb::stats);
        pack_reconfig_data_format(dfb::stats_reduced);

        /*
         * Reduce stats input.
         * dfb::stats = [sum(x0**2), sum(x0), sum(x1**2), sum(x1), ...]
         * RMSNorm packs mean(x**2) into dfb::var. Layernorm just uses dfb::stats_reduced.
         */
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::stats, dfb::reduce, dfb::stats_reduced);
        dfb_stats.wait_front(stats_tiles_cols);

        tile_regs_acquire();
        // Reduce sum(x**2) first
        for (uint32_t i = 0; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::stats, dfb::reduce, i, 0, 0);
        }
        // Reduce sum(x) next
        for (uint32_t i = 1; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::stats, dfb::reduce, i, 0, 1);
        }
        tile_regs_commit();

        dfb_stats.pop_front(stats_tiles_cols);

        dfb_stats_reduced.reserve_back(stats_tile_stride);

        tile_regs_wait();
        pack_tile(0, dfb::stats_reduced);
        pack_tile(1, dfb::stats_reduced);
        tile_regs_release();

        dfb_stats_reduced.push_back(stats_tile_stride);

        reduce_uninit();

        /*
         * E[x]**2
         */
        reconfig_data_format(dfb::stats_reduced, dfb::stats_reduced);
        pack_reconfig_data_format(dfb::mean_squared);
        mul_init(dfb::stats_reduced, dfb::stats_reduced);
        dfb_stats_reduced.wait_front(stats_tile_stride);

        tile_regs_acquire();
        mul_tiles(dfb::stats_reduced, dfb::stats_reduced, 1, 1, 0);
        tile_regs_commit();

        dfb_mean_squared.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, dfb::mean_squared);
        tile_regs_release();

        dfb_mean_squared.push_back(1);

        /*
         * E[x**2] - E[x]**2
         */
        reconfig_data_format(dfb::stats_reduced, dfb::mean_squared);
        pack_reconfig_data_format(dfb::var);
        sub_init(dfb::stats_reduced, dfb::mean_squared);

        dfb_mean_squared.wait_front(1);

        tile_regs_acquire();
        sub_tiles(dfb::stats_reduced, dfb::mean_squared, 0, 0, 0);
        tile_regs_commit();

        dfb_mean_squared.pop_front(1);

        dfb_var.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, dfb::var);
        tile_regs_release();

        dfb_var.push_back(1);

        /*
         * 1/sqrt(var + eps)
         */
        dfb_var.wait_front(1);
        reconfig_data_format(dfb::var, dfb::eps);
        pack_reconfig_data_format(dfb::recip_sqrt_var);
        add_init(dfb::var, dfb::eps);

        tile_regs_acquire();
        add_tiles(dfb::var, dfb::eps, 0, 0, 0);
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(0);
        tile_regs_commit();

        dfb_var.pop_front(1);

        dfb_recip_sqrt_var.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, dfb::recip_sqrt_var);
        tile_regs_release();

        dfb_recip_sqrt_var.push_back(1);

#if defined(FUSE_GAMMA) && defined(FUSE_BETA)
        /*
         * x_normed * gamma, then + beta
         */
        chain_llk<Wt, dfb_length, true>(
            x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{}, beta_optional_node{});
#elif defined(FUSE_GAMMA)
        chain_llk<Wt, dfb_length, true>(x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{});
#elif defined(FUSE_BETA)
        chain_llk<Wt, dfb_length, true>(x_minus_mean_node{}, normed_output_node{}, beta_optional_node{});
#else
        chain_llk<Wt, dfb_length, true>(x_minus_mean_node{}, normed_output_node{});
#endif

        // free up the buffers
        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);
    }
    dfb_eps.pop_front(1);
    dfb_reduce.pop_front(1);
}
