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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    constexpr uint32_t cb_eps = tt::CBIndex::c_4;
    constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_5;
    constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_6;
    constexpr uint32_t cb_intermediate = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_8;

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

    compute_kernel_hw_startup(cb_inp, cb_inp, cb_stats_reduced);

    DataflowBuffer(cb_eps).wait_front(1);

    // combine_welford_partials takes DataflowBuffer& (stateless wrappers over the CB id);
    // the raw uint32_t ids above are still used for the eltwise_chain template args and cb_* calls.
    DataflowBuffer cb_stats_cb(cb_stats);
    DataflowBuffer cb_stats_reduced_cb(cb_stats_reduced);

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Calculate global tile row and batch index
        uint32_t global_tile_row = tile_row_start + tile_row;
        uint32_t batch_idx = global_tile_row / Ht;
        // Combine per-device stats into mean/variance
        norm::kernel_util::compute::combine_welford_partials(
            cb_stats_cb,
            cb_stats_reduced_cb,
            num_devices,
            [W](uint32_t) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        DataflowBuffer(cb_stats_reduced).push_back(stats_tile_stride);
        DataflowBuffer(cb_stats_reduced).wait_front(stats_tile_stride);

        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(
                    cb_stats_reduced,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::input(cb_eps, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{1, 0u},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_recip_sqrt_var)>{});

        // Process tiles across width in blocks
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            ckl::sub<
                ckl::input(cb_inp, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(cb_stats_reduced, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::output(cb_intermediate, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size));

            constexpr uint32_t norm_target_cb = (do_gamma || do_beta) ? cb_intermediate : cb_out;
            DataflowBuffer(cb_recip_sqrt_var).wait_front(1);
            ckl::mul<
                ckl::input(
                    cb_intermediate,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::OperandKind::Block),
                ckl::input(cb_recip_sqrt_var, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::output(norm_target_cb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size));

            if constexpr (do_gamma) {
                constexpr uint32_t gamma_out_cb = do_beta ? cb_intermediate : cb_out;
                DataflowBuffer(cb_gamma).wait_front(col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size),
                    ckl::BinaryFpu<
                        ckl::input(
                            norm_target_cb,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::OperandKind::Block),
                        ckl::input(
                            cb_gamma,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row>{0u, col_tile},
                    ckl::PackTile<ckl::output(
                        gamma_out_cb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
            }

            // 4) optional beta (only if gamma was provided)
            if constexpr (do_beta) {
                DataflowBuffer(cb_beta).wait_front(col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_intermediate, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                        ckl::input(
                            cb_beta,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set),
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::Row>{0u, col_tile},
                    ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});
            }
        }

        // free up per-row resources
        DataflowBuffer(cb_stats_reduced).pop_front(stats_tile_stride);
        DataflowBuffer(cb_recip_sqrt_var).pop_front(1);

        // Check if next tile_row is in a different batch - if so, pop gamma/beta
        if (tile_row + 1 < num_tile_rows) {
            uint32_t next_global_tile_row = tile_row_start + tile_row + 1;
            uint32_t next_batch_idx = next_global_tile_row / Ht;
            if (next_batch_idx != batch_idx) {
                // Pop gamma/beta to prepare for next batch
                if constexpr (do_gamma && gamma_is_batched) {
                    DataflowBuffer(cb_gamma).pop_front(Wt_round_up_block_sizes);
                }
                if constexpr (do_beta && beta_is_batched) {
                    DataflowBuffer(cb_beta).pop_front(Wt_round_up_block_sizes);
                }
            }
        }
    }

    // Pop remaining gamma/beta at the end (if batched, only the last batch's data)
    if constexpr (do_gamma) {
        DataflowBuffer(cb_gamma).pop_front(Wt_round_up_block_sizes);
    }
    if constexpr (do_beta) {
        DataflowBuffer(cb_beta).pop_front(Wt_round_up_block_sizes);
    }

    DataflowBuffer(cb_eps).pop_front(1);
}
