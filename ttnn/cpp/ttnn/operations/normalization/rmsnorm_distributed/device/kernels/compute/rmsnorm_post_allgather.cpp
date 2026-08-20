// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: Production post-allgather factories bind the Metal 2.0 fork beside this file,
// rmsnorm_post_allgather_metal2.cpp. This legacy source remains as a kernel-source composition
// fixture; keep its algorithm aligned with the fork until the fixture is retired.

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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

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

    constexpr uint32_t cb_recip_sqrt_var_idx = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_x_normed_idx =
        tt::CBIndex::c_12;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

    constexpr uint32_t cb_var_idx = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)
    constexpr uint32_t cb_norm_x_input_idx = cb_inp;

    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_times_gamma_out_idx = (do_gamma && do_beta) ? tt::CBIndex::c_13 : cb_out_idx;

    CircularBuffer cb_reduce(cb_reduce_idx);
    CircularBuffer cb_eps(cb_eps_idx);
    CircularBuffer cb_gamma(cb_gamma_idx);
    CircularBuffer cb_beta(cb_beta_idx);

    compute_kernel_hw_startup(cb_inp, cb_inp, cb_var_idx);

    cb_reduce.wait_front(1);  // comes from the reader
    cb_eps.wait_front(1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into cb_var_idx for rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles CB lifecycle.
         */
        ckl::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, cb_stats, cb_reduce_idx, cb_var_idx>(
            ckl::ReduceInputBlockShape::row(stats_tiles_cols));

        // 1/sqrt(var + eps)
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(cb_var_idx),
                ckl::input(cb_eps_idx, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_recip_sqrt_var_idx)>{});

        // X * 1/sqrt(E[X**2] + eps), followed by optional gamma and beta.
        constexpr uint32_t normed_output_cb_idx = do_gamma ? cb_x_normed_idx : cb_out_idx;

        ckl::mul<
            ckl::input(
                cb_norm_x_input_idx,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block),
            ckl::input(cb_recip_sqrt_var_idx, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(normed_output_cb_idx, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/blk));

        if constexpr (do_gamma) {
            // x_normed * gamma
            ckl::mul<
                ckl::input(
                    cb_x_normed_idx,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::InputTileMapping::Block),
                ckl::input(
                    cb_gamma_idx,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Block),
                ckl::output(cb_times_gamma_out_idx, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/blk));

            if constexpr (do_beta) {
                // x_normed * gamma + beta
                ckl::add<
                    ckl::input(
                        cb_times_gamma_out_idx,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::input(
                        cb_beta_idx,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Block),
                    ckl::output(cb_out_idx, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/blk));
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
