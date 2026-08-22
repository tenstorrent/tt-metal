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

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

// Combine per-device Welford statistics into mean and variance, compute reciprocal standard
// deviation, then apply LayerNorm with optional gamma and beta.
constexpr uint32_t stats_tile_stride = 2;
constexpr auto Wt_file_scope = get_arg(args::Wt);
constexpr auto dfb_length_file_scope = get_arg(args::dfb_length);

// The normalized result goes straight to the output unless gamma or beta still has to be applied to
// it. Only the buffers this build binds have handles, so the choice is made at the preprocessor.
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
constexpr auto normed_output_dfb = dfb::x_normed;
#else
constexpr auto normed_output_dfb = dfb::out;
#endif

// gamma's product feeds the beta stage when both are applied; otherwise it is already the output.
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

    ckl::sub<
        ckl::input(dfb::inp, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
        ckl::input(dfb::stats_reduced, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
        ckl::output(dfb::x_minus_mean, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);

    // Normalize x, then route through gamma and beta when fused; otherwise write directly to out.
    ckl::mul<
        ckl::input(
            dfb::x_minus_mean,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block),
        ckl::input(dfb::recip_sqrt_var, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
        ckl::output(normed_output_dfb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);

#ifdef FUSE_GAMMA
    // x_normed * gamma, then + beta
    ckl::mul<
        ckl::input(
            dfb::x_normed, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
        ckl::input(dfb::gamma, ckl::BroadcastDim::Row, gamma_beta_wait, gamma_beta_pop, ckl::InputTileMapping::Block),
        ckl::output(times_gamma_output_dfb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
#endif
#ifdef FUSE_BETA
    ckl::add<
        ckl::input(
            beta_input_dfb, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
        ckl::input(dfb::beta, ckl::BroadcastDim::Row, gamma_beta_wait, gamma_beta_pop, ckl::InputTileMapping::Block),
        ckl::output(dfb::out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
#endif
}

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto W = get_arg(args::W);
    constexpr auto stats_tiles_cols = get_arg(args::stats_tiles_cols) / 2;
    constexpr auto dfb_length = get_arg(args::dfb_length);

    compute_kernel_hw_startup(dfb::inp, dfb::inp, dfb::stats_reduced);

    DataflowBuffer dfb_eps(dfb::eps);
    DataflowBuffer dfb_stats(dfb::stats);
    DataflowBuffer dfb_stats_reduced(dfb::stats_reduced);
    DataflowBuffer dfb_recip_sqrt_var(dfb::recip_sqrt_var);

    dfb_eps.wait_front(1);  // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        norm::kernel_util::compute::combine_welford_partials(
            dfb_stats,
            dfb_stats_reduced,
            stats_tiles_cols,
            [W](uint32_t) { return static_cast<float>(W); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        dfb_stats_reduced.push_back(stats_tile_stride);
        dfb_stats_reduced.wait_front(stats_tile_stride);

        // combine_welford_partials stores [mean, variance]; tile 1 supplies variance.
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb::stats_reduced,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Offset),
                ckl::input(dfb::eps, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{1u, 0u},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb::recip_sqrt_var)>{});

        constexpr uint32_t chunk_iterations = Wt / dfb_length;
        constexpr uint32_t leftover_tiles = Wt % dfb_length;
        for (uint32_t chunk = 0; chunk < chunk_iterations; ++chunk) {
            normalize_chunk(dfb_length);
        }
        if constexpr (leftover_tiles > 0) {
            normalize_chunk(leftover_tiles);
        }

        // free up the buffers
        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);
    }
    dfb_eps.pop_front(1);
}
