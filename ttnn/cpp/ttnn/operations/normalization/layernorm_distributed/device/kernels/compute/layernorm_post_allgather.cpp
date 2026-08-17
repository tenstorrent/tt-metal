// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

// Each device contributes interleaved partial statistics:
// stats = [sum(x0^2), sum(x0), sum(x1^2), sum(x1), ...]. Even tiles reduce E[x^2]
// and odd tiles reduce E[x].
constexpr uint32_t stats_tile_stride = 2;
constexpr auto Wt_file_scope = get_arg(args::Wt);
constexpr auto dfb_length_file_scope = get_arg(args::dfb_length);

#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
constexpr auto normed_output_dfb = dfb::x_normed;
#else
constexpr auto normed_output_dfb = dfb::out;
#endif

#if defined(FUSE_GAMMA) && defined(FUSE_BETA)
constexpr auto times_gamma_output_dfb = dfb::times_gamma_out;
#else
constexpr auto times_gamma_output_dfb = dfb::out;
#endif

#ifdef FUSE_GAMMA
constexpr auto beta_input_dfb = times_gamma_output_dfb;
#else
constexpr auto beta_input_dfb = normed_output_dfb;
#endif

ALWI void normalize_chunk(const uint32_t num_tiles) {
    const auto shape = ckl::IterationShape::tiles(num_tiles).block_size(ckl::DEST_AUTO_LIMIT);
    // When a whole row fits in one pass, gamma and beta remain resident and are re-read for every
    // row. Chunked rows consume one block at a time.
    constexpr auto gamma_beta_wait =
        Wt_file_scope == dfb_length_file_scope ? ckl::WaitPolicy::Cumulative : ckl::WaitPolicy::PerBlockSize;
    constexpr auto gamma_beta_pop =
        Wt_file_scope == dfb_length_file_scope ? ckl::PopPolicy::None : ckl::PopPolicy::PerBlockSize;

    ckl::eltwise_chain(
        shape,
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(dfb::inp, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::input(
                dfb::stats_reduced,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::None,
                ckl::OperandKind::Scalar,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileOffset::Set)>{0u, 1u},
        ckl::PackTile<ckl::output(
            dfb::x_minus_mean, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});

    // Normalize x, then route through gamma and beta when fused; otherwise write directly to out.
    ckl::mul<
        ckl::input(
            dfb::x_minus_mean, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
        ckl::input(dfb::recip_sqrt_var, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
        ckl::output(normed_output_dfb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);

#ifdef FUSE_GAMMA
    ckl::mul<
        ckl::input(dfb::x_normed, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
        ckl::input(dfb::gamma, ckl::BroadcastDim::Row, gamma_beta_wait, gamma_beta_pop, ckl::OperandKind::Block),
        ckl::output(times_gamma_output_dfb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
#endif
#ifdef FUSE_BETA
    ckl::add<
        ckl::input(
            beta_input_dfb, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
        ckl::input(dfb::beta, ckl::BroadcastDim::Row, gamma_beta_wait, gamma_beta_pop, ckl::OperandKind::Block),
        ckl::output(dfb::out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
#endif
}

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto stats_tiles_cols = get_arg(args::stats_tiles_cols);
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1;
    constexpr auto dfb_length = get_arg(args::dfb_length);

    compute_kernel_hw_startup(dfb::inp, dfb::inp, dfb::stats_reduced);

    DataflowBuffer dfb_reduce(dfb::reduce);
    DataflowBuffer dfb_eps(dfb::eps);
    DataflowBuffer dfb_stats(dfb::stats);
    DataflowBuffer dfb_stats_reduced(dfb::stats_reduced);
    DataflowBuffer dfb_recip_sqrt_var(dfb::recip_sqrt_var);

    dfb_reduce.wait_front(1);
    dfb_eps.wait_front(1);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        reconfig_data_format(dfb::reduce, dfb::stats);
        pack_reconfig_data_format(dfb::stats_reduced);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::stats, dfb::reduce, dfb::stats_reduced);
        dfb_stats.wait_front(stats_tiles_cols);

        tile_regs_acquire();
        for (uint32_t i = 0; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::stats, dfb::reduce, i, 0, 0);
        }
        for (uint32_t i = 1; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb::stats, dfb::reduce, i, 0, 1);
        }
        tile_regs_commit();
        dfb_stats.pop_front(stats_tiles_cols);
        dfb_stats_reduced.reserve_back(stats_tile_stride);
        tile_regs_wait();
        pack_tile(0, dfb::stats_reduced);
        pack_tile(1, dfb::stats_reduced);
        tile_regs_release();
        dfb_stats_reduced.push_back(stats_tile_stride);
        reduce_uninit();

        dfb_stats_reduced.wait_front(stats_tile_stride);

        // variance = E[x^2] - E[x]^2; reciprocal standard deviation = 1/sqrt(variance + eps).
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    dfb::stats_reduced,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::input(
                    dfb::stats_reduced,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set)>{1u, 1u},
            ckl::PackTile<ckl::output(dfb::mean_squared)>{});

        ckl::sub<
            ckl::input(dfb::stats_reduced, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
            ckl::input(dfb::mean_squared, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb::var, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::one_tile());

        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::var),
                ckl::input(dfb::eps, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb::recip_sqrt_var)>{});

        constexpr uint32_t chunk_iterations = Wt / dfb_length;
        constexpr uint32_t leftover_tiles = Wt % dfb_length;
        for (uint32_t chunk = 0; chunk < chunk_iterations; ++chunk) {
            normalize_chunk(dfb_length);
        }
        if constexpr (leftover_tiles > 0) {
            normalize_chunk(leftover_tiles);
        }

        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);
    }
    dfb_eps.pop_front(1);
    dfb_reduce.pop_front(1);
}
