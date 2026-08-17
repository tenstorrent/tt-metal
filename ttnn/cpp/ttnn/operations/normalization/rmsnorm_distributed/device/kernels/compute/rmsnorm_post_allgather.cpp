// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as
// rmsnorm_post_allgather_metal2.cpp. Ops ported to Metal 2.0 bind the fork; this file serves
// the consumers still on the legacy API. Until the last of them migrates and
// this file is retired, changes here likely belong in the fork too.

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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
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

    constexpr uint32_t dfb_inp_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_stats_id = tt::CBIndex::c_1;

    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_4;
    constexpr uint32_t dfb_reduce_id = tt::CBIndex::c_5;

    constexpr uint32_t dfb_out_id = tt::CBIndex::c_14;

    constexpr uint32_t dfb_var_eps_id = tt::CBIndex::c_9;          // var + epsilon (or E(x**2) + epsilon)
    constexpr uint32_t dfb_recip_sqrt_var_id = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
    constexpr uint32_t dfb_x_normed_id =
        tt::CBIndex::c_12;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

    constexpr uint32_t dfb_var_id = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)
    constexpr uint32_t dfb_norm_x_input_id = dfb_inp_id;

    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_times_gamma_out_id = (do_gamma && do_beta) ? tt::CBIndex::c_13 : dfb_out_id;

    compute_kernel_hw_startup(dfb_inp_id, dfb_inp_id, dfb_var_id);

    DataflowBuffer(dfb_reduce_id).wait_front(1);  // comes from the reader
    DataflowBuffer(dfb_eps_id).wait_front(1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * Reduce stats input.
         * dfb_stats_id = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into dfb_var_id for rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles DFB lifecycle.
         */
        ckl::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, dfb_stats_id, dfb_reduce_id, dfb_var_id>(
            ckl::ReduceInputBlockShape::row(stats_tiles_cols));

        // 1/sqrt(var + eps)
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_var_id),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb_recip_sqrt_var_id)>{});

        // X * 1/sqrt(E[X**2] + eps), followed by optional gamma and beta.
        constexpr uint32_t normed_output_dfb_id = do_gamma ? dfb_x_normed_id : dfb_out_id;

        ckl::mul<
            ckl::input(dfb_norm_x_input_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(dfb_recip_sqrt_var_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(normed_output_dfb_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/blk));

        if constexpr (do_gamma) {
            ckl::mul<
                ckl::input(dfb_x_normed_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(
                    dfb_gamma_id,
                    ckl::BroadcastDim::Row,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Block),
                ckl::output(dfb_times_gamma_out_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
                ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/blk));

            if constexpr (do_beta) {
                ckl::add<
                    ckl::input(
                        dfb_times_gamma_out_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::AtEnd,
                        ckl::OperandKind::Block),
                    ckl::input(
                        dfb_beta_id,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Block),
                    ckl::output(dfb_out_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/blk));
            }
        }
    }
    DataflowBuffer(dfb_eps_id).pop_front(1);
    DataflowBuffer(dfb_reduce_id).pop_front(1);
    if constexpr (do_gamma) {
        DataflowBuffer(dfb_gamma_id).pop_front(Wt);
    }
    if constexpr (do_beta) {
        DataflowBuffer(dfb_beta_id).pop_front(Wt);
    }
}
