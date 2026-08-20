// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * LayerNorm-only Welford post-allgather.
 * Expects stats with two TILE columns per device (E(x**2), E(x)), applies LN with optional gamma/beta.
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t dfb_inp_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_stats_id = tt::CBIndex::c_1;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_4;
    constexpr uint32_t dfb_stats_reduced_id = tt::CBIndex::c_5;   // [mean, var]
    constexpr uint32_t dfb_recip_sqrt_var_id = tt::CBIndex::c_6;  // 1/sqrt(var+eps)
    constexpr uint32_t dfb_intermediate_id = tt::CBIndex::c_7;    // intermediate result
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_8;

    constexpr uint32_t stats_tile_stride = 2;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);

    constexpr uint32_t do_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t do_beta = get_compile_time_arg_val(5);
    constexpr uint32_t gamma_is_batched = get_compile_time_arg_val(6);
    constexpr uint32_t beta_is_batched = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);

    constexpr uint32_t Wt_round_up_block_sizes = get_compile_time_arg_val(9);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(dfb_inp_id, dfb_inp_id, dfb_stats_reduced_id);

    // combine_welford_partials takes DataflowBuffer&, while the eltwise chains require raw DFB ids.
    DataflowBuffer dfb_stats(dfb_stats_id);
    DataflowBuffer dfb_stats_reduced(dfb_stats_reduced_id);
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_recip_sqrt_var(dfb_recip_sqrt_var_id);
    DataflowBuffer dfb_gamma(dfb_gamma_id);
    DataflowBuffer dfb_beta(dfb_beta_id);

    dfb_eps.wait_front(1);  // broadcast epsilon is ready

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Calculate global tile row and batch index
        uint32_t global_tile_row = tile_row_start + tile_row;
        uint32_t batch_idx = global_tile_row / Ht;
        // Combine per-device stats into mean/variance
        norm::kernel_util::compute::combine_welford_partials(
            dfb_stats,
            dfb_stats_reduced,
            num_devices,
            [W](uint32_t) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        dfb_stats_reduced.push_back(stats_tile_stride);
        dfb_stats_reduced.wait_front(stats_tile_stride);

        // Compute 1/sqrt(var + eps) into dfb_recip_sqrt_var_id
        // combine_welford_partials stores [mean, variance]; tile 1 supplies variance.
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb_stats_reduced_id,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Offset),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{1, 0u},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb_recip_sqrt_var_id)>{});

        // Process tiles across width in blocks
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            // 1) x_minus_mean
            ckl::sub<
                ckl::input(dfb_inp_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::input(dfb_stats_reduced_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::output(dfb_intermediate_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
                ckl::IterationShape::tiles(block_size).block_size(/*block_size=*/block_size));

            // 2) normalize: (x-mean) * inv_std
            constexpr uint32_t norm_target_dfb_id = (do_gamma || do_beta) ? dfb_intermediate_id : dfb_out_id;
            dfb_recip_sqrt_var.wait_front(1);
            // Note that compute and pack are separated because it's possible that
            // norm_target_dfb_id == dfb_intermediate_id (in the case of no gamma/beta), so this
            // must be able to support in-place operations.
            ckl::mul<
                ckl::input(
                    dfb_intermediate_id,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::InputTileMapping::Block),
                ckl::input(dfb_recip_sqrt_var_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::output(norm_target_dfb_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::tiles(block_size).block_size(/*block_size=*/block_size));

            // 3) optional gamma
            if constexpr (do_gamma) {
                constexpr uint32_t gamma_out_dfb_id = do_beta ? dfb_intermediate_id : dfb_out_id;
                dfb_gamma.wait_front(col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(block_size).block_size(/*block_size=*/block_size),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            norm_target_dfb_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb_gamma_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Offset)>{0u, col_tile},
                    ckl::PackTile<ckl::output(
                        gamma_out_dfb_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
            }

            // 4) optional beta (only if gamma was provided)
            if constexpr (do_beta) {
                // Input is always in dfb_intermediate_id, output is always dfb_out_id
                dfb_beta.wait_front(col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(block_size).block_size(/*block_size=*/block_size),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb_intermediate_id,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::AtEnd,
                            ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb_beta_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Offset)>{0u, col_tile},
                    ckl::PackTile<ckl::output(dfb_out_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});
            }
        }

        // free up per-row resources
        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);

        // Check if next tile_row is in a different batch - if so, pop gamma/beta
        if (tile_row + 1 < num_tile_rows) {
            uint32_t next_global_tile_row = tile_row_start + tile_row + 1;
            uint32_t next_batch_idx = next_global_tile_row / Ht;
            if (next_batch_idx != batch_idx) {
                // Pop gamma/beta to prepare for next batch
                if constexpr (do_gamma && gamma_is_batched) {
                    dfb_gamma.pop_front(Wt_round_up_block_sizes);
                }
                if constexpr (do_beta && beta_is_batched) {
                    dfb_beta.pop_front(Wt_round_up_block_sizes);
                }
            }
        }
    }

    // Pop remaining gamma/beta at the end (if batched, only the last batch's data)
    if constexpr (do_gamma) {
        dfb_gamma.pop_front(Wt_round_up_block_sizes);
    }
    if constexpr (do_beta) {
        dfb_beta.pop_front(Wt_round_up_block_sizes);
    }

    dfb_eps.pop_front(1);
}
