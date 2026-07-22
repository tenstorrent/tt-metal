
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
#include "api/dataflow/circular_buffer.h"
#include "chain_llk.hpp"

constexpr uint32_t cb_inp = tt::CBIndex::c_0;
constexpr uint32_t cb_stats_idx = tt::CBIndex::c_1;

constexpr uint32_t cb_eps_idx = tt::CBIndex::c_4;

constexpr uint32_t cb_out = tt::CBIndex::c_14;

constexpr uint32_t cb_stats_reduced_idx = tt::CBIndex::c_6;    // [E(x**2), E(x)]
constexpr uint32_t cb_recip_sqrt_var_idx = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
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
        .CB_B = cb_stats_reduced_idx,
        .CB_OUT = cb_x_minus_mean,
        .fixed_CB_B_index = 1,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
constexpr uint32_t do_beta = get_compile_time_arg_val(4);
constexpr uint32_t normed_output_cb =
    do_gamma || do_beta ? cb_x_normed : cb_out;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)
struct normed_output_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_cols_init_short,
        .llk = FN_compute(mul_tiles_bcast_cols),
        .CB_A = cb_norm_x_input,
        .CB_B = cb_recip_sqrt_var_idx,
        .CB_OUT = normed_output_cb,
        .fixed_CB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
constexpr uint32_t cb_length = get_compile_time_arg_val(8);
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t pop_gamma_beta = Wt == cb_length ? 0xDDDD : 0xFFFF;

constexpr uint32_t cb_times_gamma_out = do_beta ? tt::CBIndex::c_13 : cb_out;
struct gamma_optional_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_rows_init_short,
        .llk = FN_compute(mul_tiles_bcast_rows),
        .CB_A = cb_x_normed,
        .CB_B = cb_gamma,
        .CB_OUT = cb_times_gamma_out,
        .fixed_CB_B_index = pop_gamma_beta,
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
        .fixed_CB_B_index = pop_gamma_beta,
        .fixed_dest_reg = 0xFFFF,
    };
};
void kernel_main() {
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

    constexpr uint32_t cb_reduce_idx = tt::CBIndex::c_5;

    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;  // var + epsilon (or E(x**2) + epsilon)

    constexpr uint32_t cb_var_idx = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)

    // Layernorm-specific CBs
    constexpr uint32_t cb_mean_squared_idx = tt::CBIndex::c_7;  // E(x)**2

    compute_kernel_hw_startup(cb_inp, cb_inp, cb_stats_reduced_idx);

    CircularBuffer cb_reduce(cb_reduce_idx);
    CircularBuffer cb_eps(cb_eps_idx);
    CircularBuffer cb_stats(cb_stats_idx);
    CircularBuffer cb_stats_reduced(cb_stats_reduced_idx);
    CircularBuffer cb_mean_squared(cb_mean_squared_idx);
    CircularBuffer cb_var(cb_var_idx);
    CircularBuffer cb_recip_sqrt_var(cb_recip_sqrt_var_idx);

    cb_reduce.wait_front(1);  // comes from the reader
    cb_eps.wait_front(1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        reconfig_data_format(cb_reduce_idx, cb_stats_idx);
        pack_reconfig_data_format(cb_stats_reduced_idx);

        /*
         * Reduce stats input.
         * cb_stats_idx = [sum(x0**2), sum(x0), sum(x1**2), sum(x1), ...]
         * RMSNorm packs mean(x**2) into cb_var_idx. Layernorm just uses cb_stats_reduced_idx.
         */
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_stats_idx, cb_reduce_idx, cb_stats_reduced_idx);
        cb_stats.wait_front(stats_tiles_cols);

        tile_regs_acquire();
        // Reduce sum(x**2) first
        for (uint32_t i = 0; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_stats_idx, cb_reduce_idx, i, 0, 0);
        }
        // Reduce sum(x) next
        for (uint32_t i = 1; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_stats_idx, cb_reduce_idx, i, 0, 1);
        }
        tile_regs_commit();

        cb_stats.pop_front(stats_tiles_cols);

        cb_stats_reduced.reserve_back(stats_tile_stride);

        tile_regs_wait();
        pack_tile(0, cb_stats_reduced_idx);
        pack_tile(1, cb_stats_reduced_idx);
        tile_regs_release();

        cb_stats_reduced.push_back(stats_tile_stride);

        reduce_uninit();

        /*
         * E[x]**2
         */
        reconfig_data_format(cb_stats_reduced_idx, cb_stats_reduced_idx);
        pack_reconfig_data_format(cb_mean_squared_idx);
        mul_init(cb_stats_reduced_idx, cb_stats_reduced_idx);
        cb_stats_reduced.wait_front(stats_tile_stride);

        tile_regs_acquire();
        mul_tiles(cb_stats_reduced_idx, cb_stats_reduced_idx, 1, 1, 0);
        tile_regs_commit();

        cb_mean_squared.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, cb_mean_squared_idx);
        tile_regs_release();

        cb_mean_squared.push_back(1);

        /*
         * E[x**2] - E[x]**2
         */
        reconfig_data_format(cb_stats_reduced_idx, cb_mean_squared_idx);
        pack_reconfig_data_format(cb_var_idx);
        sub_init(cb_stats_reduced_idx, cb_mean_squared_idx);

        cb_mean_squared.wait_front(1);

        tile_regs_acquire();
        sub_tiles(cb_stats_reduced_idx, cb_mean_squared_idx, 0, 0, 0);
        tile_regs_commit();

        cb_mean_squared.pop_front(1);

        cb_var.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, cb_var_idx);
        tile_regs_release();

        cb_var.push_back(1);

        /*
         * 1/sqrt(var + eps)
         */
        cb_var.wait_front(1);
        reconfig_data_format(cb_var_idx, cb_eps_idx);
        pack_reconfig_data_format(cb_recip_sqrt_var_idx);
        add_init(cb_var_idx, cb_eps_idx);

        tile_regs_acquire();
        add_tiles(cb_var_idx, cb_eps_idx, 0, 0, 0);
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(0);
        tile_regs_commit();

        cb_var.pop_front(1);

        cb_recip_sqrt_var.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, cb_recip_sqrt_var_idx);
        tile_regs_release();

        cb_recip_sqrt_var.push_back(1);

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
        cb_stats_reduced.pop_front(stats_tile_stride);
        cb_recip_sqrt_var.pop_front(1);
    }
    cb_eps.pop_front(1);
    cb_reduce.pop_front(1);
}
