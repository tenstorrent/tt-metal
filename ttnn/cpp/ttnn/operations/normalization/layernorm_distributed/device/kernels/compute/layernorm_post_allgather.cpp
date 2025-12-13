
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
#include "chain_llk.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
constexpr uint32_t cb_inp = tt::CBIndex::c_0;
constexpr uint32_t cb_stats = tt::CBIndex::c_1;

constexpr uint32_t cb_eps = tt::CBIndex::c_4;

constexpr uint32_t cb_out = tt::CBIndex::c_14;

constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_6;    // [E(x**2), E(x)]
constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
constexpr uint32_t cb_x_normed = tt::CBIndex::c_12;        // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

// Layernorm-specific CBs
constexpr uint32_t cb_x_minus_mean = tt::CBIndex::c_11;  // x - E(x)

constexpr uint32_t cb_norm_x_input = cb_x_minus_mean;
constexpr uint32_t stats_tile_stride = 2;

struct x_minus_mean_node {
    static constexpr LLK_Node node{
        .llk_init = sub_bcast_cols_init_short,
        .llk = FN_compute(sub_tiles_bcast_cols),
        .CB_A = cb_inp,
        .CB_B = cb_stats_reduced,
        .CB_OUT = cb_x_minus_mean,
        .fixed_CB_B_index = 1,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t do_gamma = get_compile_time_arg_val(4);
constexpr uint32_t do_beta = get_compile_time_arg_val(5);
constexpr uint32_t normed_output_cb =
    do_gamma || do_beta ? cb_x_normed : cb_out;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)
struct normed_output_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_cols_init_short,
        .llk = FN_compute(mul_tiles_bcast_cols),
        .CB_A = cb_norm_x_input,
        .CB_B = cb_recip_sqrt_var,
        .CB_OUT = normed_output_cb,
        .fixed_CB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t cb_gamma = tt::CBIndex::c_2;

constexpr uint32_t cb_times_gamma_out = do_beta ? tt::CBIndex::c_13 : cb_out;
struct gamma_optional_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_rows_init_short,
        .llk = FN_compute(mul_tiles_bcast_rows),
        .CB_A = cb_x_normed,
        .CB_B = cb_gamma,
        .CB_OUT = cb_times_gamma_out,
        .fixed_CB_B_index = 0xFFFF,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t cb_in_beta = do_gamma ? cb_times_gamma_out : normed_output_cb;
constexpr uint32_t cb_beta = tt::CBIndex::c_3;
struct beta_optional_node {
    static constexpr LLK_Node node{
        .llk_init = add_bcast_rows_init_short,
        .llk = FN_compute(add_tiles_bcast_rows),
        .CB_A = cb_in_beta,
        .CB_B = cb_beta,
        .CB_OUT = cb_out,
        .fixed_CB_B_index = 0xFFFF,
        .fixed_dest_reg = 0xFFFF,
    };
};
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(2);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t do_beta = get_compile_time_arg_val(4);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(5) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(6) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t cb_length = get_compile_time_arg_val(8);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_reduce = tt::CBIndex::c_5;

    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;  // var + epsilon (or E(x**2) + epsilon)

    constexpr uint32_t cb_var = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)

    // Layernorm-specific CBs
    constexpr uint32_t cb_mean_squared = tt::CBIndex::c_7;  // E(x)**2

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
        reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_stats, cb_reduce, cb_stats_reduced);
        cb_wait_front(cb_stats, stats_tiles_cols);
        cb_reserve_back(cb_stats_reduced, stats_tile_stride);

        ACQ();
        // Reduce sum(x**2) first
        for (uint32_t i = 0; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_stats, cb_reduce, i, 0, 0);
        }
        pack_tile(0, cb_stats_reduced);

        // Reduce sum(x) next
        for (uint32_t i = 1; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_stats, cb_reduce, i, 0, 1);
        }
        pack_tile(1, cb_stats_reduced);

        REL();
        cb_push_back(cb_stats_reduced, stats_tile_stride);
        cb_pop_front(cb_stats, stats_tiles_cols);

        reduce_uninit();

        /*
         * E[x]**2
         */
        reconfig_data_format(cb_stats_reduced, cb_stats_reduced);
        pack_reconfig_data_format(cb_mean_squared);
        mul_tiles_init(cb_stats_reduced, cb_stats_reduced);
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
        sub_tiles_init(cb_stats_reduced, cb_mean_squared);

        cb_reserve_back(cb_var, onetile);
        cb_wait_front(cb_mean_squared, 1);
        ACQ();
        sub_tiles(cb_stats_reduced, cb_mean_squared, 0, 0, 0);
        pack_tile(0, cb_var);
        REL();
        cb_push_back(cb_var, 1);
        cb_pop_front(cb_mean_squared, 1);

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

        if constexpr (do_gamma && do_beta) {
            /*
             * x_normed * gamma
             */
            chain_llk<Wt, cb_length, true>(
                x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{}, beta_optional_node{});

        } else if (do_gamma) {
            chain_llk<Wt, cb_length, true>(x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{});
        } else if (do_beta) {
            chain_llk<Wt, cb_length, true>(x_minus_mean_node{}, normed_output_node{}, beta_optional_node{});
        } else {
            chain_llk<Wt, cb_length, true>(x_minus_mean_node{}, normed_output_node{});
        }

        // free up CBs
        cb_pop_front(cb_stats_reduced, stats_tile_stride);
        cb_pop_front(cb_recip_sqrt_var, 1);
    }
    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_reduce, 1);
}
}  // namespace NAMESPACE
