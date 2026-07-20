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

    binary_op_init_common(cb_inp, cb_inp, cb_stats_reduced);

    cb_wait_front(cb_eps, 1);

    // combine_welford_partials takes CircularBuffer& (stateless wrappers over the CB id);
    // the raw uint32_t ids above are still used for the eltwise_chain template args and cb_* calls.
    CircularBuffer cb_stats_cb(cb_stats);
    CircularBuffer cb_stats_reduced_cb(cb_stats_reduced);

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
        cb_push_back(cb_stats_reduced, stats_tile_stride);
        cb_wait_front(cb_stats_reduced, stats_tile_stride);

        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                cb_stats_reduced,
                cb_eps,
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None,
                ckl::input(
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{1, 0u},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
            ckl::PackTile<cb_recip_sqrt_var>{});

        // Process tiles across width in blocks
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            ckl::sub<
                cb_inp,
                cb_stats_reduced,
                cb_intermediate,
                ckl::BroadcastDim::Col,
                ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::output(ckl::OutputLifecycle::Bulk)>(
                ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size));

            constexpr uint32_t norm_target_cb = (do_gamma || do_beta) ? cb_intermediate : cb_out;
            cb_wait_front(cb_recip_sqrt_var, 1);
            ckl::mul<
                cb_intermediate,
                cb_recip_sqrt_var,
                norm_target_cb,
                ckl::BroadcastDim::Col,
                ckl::input(ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::output(ckl::OutputLifecycle::Chunked)>(
                ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size));

            if constexpr (do_gamma) {
                constexpr uint32_t gamma_out_cb = do_beta ? cb_intermediate : cb_out;
                cb_wait_front(cb_gamma, col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size),
                    ckl::BinaryFpu<
                        norm_target_cb,
                        cb_gamma,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row,
                        ckl::input(ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
                        ckl::input(
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set)>{0u, col_tile},
                    ckl::PackTile<gamma_out_cb, ckl::output(ckl::OutputLifecycle::Chunked)>{});
            }

            // 4) optional beta (only if gamma was provided)
            if constexpr (do_beta) {
                cb_wait_front(cb_beta, col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size),
                    ckl::BinaryFpu<
                        cb_intermediate,
                        cb_beta,
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::Row,
                        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                        ckl::input(
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set)>{0u, col_tile},
                    ckl::PackTile<cb_out, ckl::output(ckl::OutputLifecycle::Bulk)>{});
            }
        }

        // free up per-row resources
        cb_pop_front(cb_stats_reduced, stats_tile_stride);
        cb_pop_front(cb_recip_sqrt_var, 1);

        // Check if next tile_row is in a different batch - if so, pop gamma/beta
        if (tile_row + 1 < num_tile_rows) {
            uint32_t next_global_tile_row = tile_row_start + tile_row + 1;
            uint32_t next_batch_idx = next_global_tile_row / Ht;
            if (next_batch_idx != batch_idx) {
                // Pop gamma/beta to prepare for next batch
                if constexpr (do_gamma && gamma_is_batched) {
                    cb_pop_front(cb_gamma, Wt_round_up_block_sizes);
                }
                if constexpr (do_beta && beta_is_batched) {
                    cb_pop_front(cb_beta, Wt_round_up_block_sizes);
                }
            }
        }
    }

    // Pop remaining gamma/beta at the end (if batched, only the last batch's data)
    if constexpr (do_gamma) {
        cb_pop_front(cb_gamma, Wt_round_up_block_sizes);
    }
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, Wt_round_up_block_sizes);
    }

    cb_pop_front(cb_eps, 1);
}
