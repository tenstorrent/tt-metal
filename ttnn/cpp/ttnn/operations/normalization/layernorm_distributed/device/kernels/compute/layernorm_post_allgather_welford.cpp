// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
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
        .fixed_DFB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
        .debug_mode = 1,
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
        .debug_mode = 1,
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
        .debug_mode = 1,
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
        .debug_mode = 1,
    };
};
#endif

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto W = get_arg(args::W);
    constexpr auto blk = get_arg(args::blk);
    constexpr auto stats_tiles_cols = get_arg(args::stats_tiles_cols) / 2;
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dtype) == 1;
    constexpr auto dfb_length = get_arg(args::dfb_length);
    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::inp, dfb::inp, dfb::stats_reduced);

    DataflowBuffer dfb_eps(dfb::eps);
    DataflowBuffer dfb_stats(dfb::stats);
    DataflowBuffer dfb_stats_reduced(dfb::stats_reduced);
    DataflowBuffer dfb_recip_sqrt_var(dfb::recip_sqrt_var);

    dfb_eps.wait_front(1);  // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        norm::kernel_util::compute::combine_welford_partials(
            dfb_stats,
            dfb_stats_reduced,
            stats_tiles_cols,
            [W](uint32_t b) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        dfb_stats_reduced.push_back(2);
        dfb_stats_reduced.wait_front(2);
        /*
         * 1/sqrt(var + eps)
         */

        dfb_stats_reduced.wait_front(2);
        dfb_recip_sqrt_var.reserve_back(1);
        reconfig_data_format(dfb::stats_reduced, dfb::eps);
        pack_reconfig_data_format(dfb::recip_sqrt_var);

        add_init(dfb::stats_reduced, dfb::eps);
        tile_regs_acquire();
        tile_regs_wait();
        add_tiles(dfb::stats_reduced, dfb::eps, 1, 0, 0);
        rsqrt_tile_init<true>();
        rsqrt_tile<true>(0);
        pack_tile(0, dfb::recip_sqrt_var);
        tile_regs_commit();
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
}
