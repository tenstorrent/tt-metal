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

// Pack dest of the normalize stage is a real DFB in every compile: exactly one of
// x_normed (γ∨β) and out (neither) is the staging dest.
constexpr auto normed_output_dfb = engaged_token_between(dfb::x_normed, dfb::out);
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
constexpr auto dfb_times_gamma_out = engaged_token_between(dfb::times_gamma_out, dfb::out);
constexpr auto dfb_in_beta = engaged_token_between(dfb_times_gamma_out, normed_output_dfb);

// Four host configs (none / γ / β / both) share this source. Mixed absences are valid and the
// node list is a compile-time argument pack, so dispatch is an overload set on the tokens.
void apply_gamma_beta(DFBBindingToken, DFBBindingToken) {
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
    chain_llk<Wt_file_scope, dfb_length_file_scope, true>(
        x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{}, beta_optional_node{});
}
void apply_gamma_beta(DFBBindingToken, NullDFBBindingToken) {
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
    chain_llk<Wt_file_scope, dfb_length_file_scope, true>(
        x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{});
}
void apply_gamma_beta(NullDFBBindingToken, DFBBindingToken) {
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
    chain_llk<Wt_file_scope, dfb_length_file_scope, true>(
        x_minus_mean_node{}, normed_output_node{}, beta_optional_node{});
}
void apply_gamma_beta(NullDFBBindingToken, NullDFBBindingToken) {
    chain_llk<Wt_file_scope, dfb_length_file_scope, true>(x_minus_mean_node{}, normed_output_node{});
}

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

        apply_gamma_beta(dfb::gamma, dfb::beta);

        // free up the buffers
        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);
    }

    dfb_eps.pop_front(1);
}
