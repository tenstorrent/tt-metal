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
#include "chain_llk.hpp"

constexpr uint32_t dfb_inp = tt::CBIndex::c_0;
constexpr uint32_t dfb_stats_id = tt::CBIndex::c_1;

constexpr uint32_t dfb_eps_id = tt::CBIndex::c_4;

constexpr uint32_t dfb_out = tt::CBIndex::c_14;

constexpr uint32_t dfb_stats_reduced_id = tt::CBIndex::c_6;    // [E(x**2), E(x)]
constexpr uint32_t dfb_recip_sqrt_var_id = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
constexpr uint32_t dfb_x_normed = tt::CBIndex::c_12;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

// Layernorm-specific CBs
constexpr uint32_t dfb_x_minus_mean = tt::CBIndex::c_11;  // x - E(x)

constexpr uint32_t dfb_norm_x_input = dfb_x_minus_mean;
constexpr uint32_t stats_tile_stride = 2;

struct x_minus_mean_node {
    static constexpr LLK_Node node{
        .llk_init = sub_bcast_cols_init_short,
        .llk = FN_compute(sub_tiles_bcast_cols),
        .CB_A = dfb_inp,
        .CB_B = dfb_stats_reduced_id,
        .CB_OUT = dfb_x_minus_mean,
        .fixed_CB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
        .debug_mode = 1,
    };
};
constexpr uint32_t do_gamma = get_compile_time_arg_val(4);
constexpr uint32_t do_beta = get_compile_time_arg_val(5);
constexpr uint32_t normed_output_dfb =
    do_gamma || do_beta ? dfb_x_normed : dfb_out;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)
struct normed_output_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_cols_init_short,
        .llk = FN_compute(mul_tiles_bcast_cols),
        .CB_A = dfb_norm_x_input,
        .CB_B = dfb_recip_sqrt_var_id,
        .CB_OUT = normed_output_dfb,
        .fixed_CB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
        .debug_mode = 1,
    };
};
constexpr uint32_t dfb_gamma = tt::CBIndex::c_2;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t dfb_length = get_compile_time_arg_val(7);
constexpr uint32_t pop_gamma_beta = Wt == dfb_length ? 0xDDDD : 0xFFFF;
constexpr uint32_t dfb_times_gamma_out = do_beta ? tt::CBIndex::c_13 : dfb_out;
struct gamma_optional_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_rows_init_short,
        .llk = FN_compute(mul_tiles_bcast_rows),
        .CB_A = dfb_x_normed,
        .CB_B = dfb_gamma,
        .CB_OUT = dfb_times_gamma_out,
        .fixed_CB_B_index = pop_gamma_beta,
        .fixed_dest_reg = 0xFFFF,
        .debug_mode = 1,
    };
};
constexpr uint32_t dfb_in_beta = do_gamma ? dfb_times_gamma_out : normed_output_dfb;
constexpr uint32_t dfb_beta = tt::CBIndex::c_3;
struct beta_optional_node {
    static constexpr LLK_Node node{
        .llk_init = add_bcast_rows_init_short,
        .llk = FN_compute(add_tiles_bcast_rows),
        .CB_A = dfb_in_beta,
        .CB_B = dfb_beta,
        .CB_OUT = dfb_out,
        .fixed_CB_B_index = pop_gamma_beta,
        .fixed_dest_reg = 0xFFFF,
        .debug_mode = 1,
    };
};

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(3) / 2;
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(dfb_inp, dfb_inp, dfb_stats_reduced_id);

    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_stats(dfb_stats_id);
    DataflowBuffer dfb_stats_reduced(dfb_stats_reduced_id);
    DataflowBuffer dfb_recip_sqrt_var(dfb_recip_sqrt_var_id);

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
        reconfig_data_format(dfb_stats_reduced_id, dfb_eps_id);
        pack_reconfig_data_format(dfb_recip_sqrt_var_id);

        add_tiles_init(dfb_stats_reduced_id, dfb_eps_id);
        tile_regs_acquire();
        tile_regs_wait();
        add_tiles(dfb_stats_reduced_id, dfb_eps_id, 1, 0, 0);
        rsqrt_tile_init<true>();
        rsqrt_tile<true>(0);
        pack_tile(0, dfb_recip_sqrt_var_id);
        tile_regs_commit();
        tile_regs_release();
        dfb_recip_sqrt_var.push_back(1);

        if constexpr (do_gamma && do_beta) {
            /*
             * x_normed * gamma
             */
            chain_llk<Wt, dfb_length, true>(
                x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{}, beta_optional_node{});

        } else if (do_gamma) {
            chain_llk<Wt, dfb_length, true>(x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{});
        } else if (do_beta) {
            chain_llk<Wt, dfb_length, true>(x_minus_mean_node{}, normed_output_node{}, beta_optional_node{});
        } else {
            chain_llk<Wt, dfb_length, true>(x_minus_mean_node{}, normed_output_node{});
        }

        // free up CBs
        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);
    }

    dfb_eps.pop_front(1);
}
