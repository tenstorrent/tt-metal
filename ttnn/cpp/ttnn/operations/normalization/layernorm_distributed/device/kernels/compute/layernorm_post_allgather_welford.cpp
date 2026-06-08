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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

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
        .fixed_CB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
        .debug_mode = 1,
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
        .debug_mode = 1,
    };
};
constexpr uint32_t cb_gamma = tt::CBIndex::c_2;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t cb_length = get_compile_time_arg_val(7);
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
        .debug_mode = 1,
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

    binary_op_init_common(cb_inp, cb_inp, cb_stats_reduced);

    cb_wait_front(cb_eps, 1);  // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        norm::kernel_util::compute::combine_welford_partials(
            cb_stats,
            cb_stats_reduced,
            stats_tiles_cols,
            [&](uint32_t b) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        cb_push_back(cb_stats_reduced, 2);
        cb_wait_front(cb_stats_reduced, 2);
        /*
         * 1/sqrt(var + eps)
         */

        // 1/sqrt(var + eps)  — Add reads cb_stats_reduced at index 1 (variance slot;
        // mean is at index 0); cb_eps at index 0.
        // Reconfig audit: explicit reconfig_data_format + add_tiles_init -> Input.
        // Explicit pack_reconfig_data_format -> Output. rsqrt_tile_init<true> -> Legacy::On.
        // Lifecycles: cb_stats_reduced InputLifecycle::HeldBulk + Scalar + compute_kernel_lib::TileOffset::Set
        // (held, popped at line 164 with stats_tile_stride). cb_eps InputLifecycle::CallerManaged + Scalar.
        // cb_recip_sqrt_var OutputLifecycle::Streaming.
        compute_kernel_lib::eltwise_chain(
            1,
            compute_kernel_lib::BinaryFpu<
                cb_stats_reduced,
                cb_eps,
                compute_kernel_lib::BinaryFpuOp::Add,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::InputLifecycle::HeldBulk,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::TileOffset::Set>{1, 0u},
            compute_kernel_lib::
                Rsqrt<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Legacy::On, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<cb_recip_sqrt_var>{});

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
}
