// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

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
    const auto shape = ckl::EltwiseShape::tiles(num_tiles, ckl::DEST_AUTO_LIMIT);
    constexpr auto gamma_beta_wait =
        Wt_file_scope == dfb_length_file_scope ? ckl::WaitPolicy::Cumulative : ckl::WaitPolicy::PerBlockSize;
    constexpr auto gamma_beta_pop =
        Wt_file_scope == dfb_length_file_scope ? ckl::PopPolicy::None : ckl::PopPolicy::PerBlockSize;

    ckl::sub<
        ckl::input(dfb::inp, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
        ckl::input(dfb::stats_reduced, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
        ckl::output(dfb::x_minus_mean, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize),
        ckl::BroadcastDim::Col>(shape);

    ckl::mul<
        ckl::input(
            dfb::x_minus_mean, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
        ckl::input(dfb::recip_sqrt_var, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
        ckl::output(normed_output_dfb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize),
        ckl::BroadcastDim::Col>(shape);

#ifdef FUSE_GAMMA
    ckl::mul<
        ckl::input(dfb::x_normed, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
        ckl::input(dfb::gamma, gamma_beta_wait, gamma_beta_pop, ckl::OperandKind::Block),
        ckl::output(times_gamma_output_dfb, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize),
        ckl::BroadcastDim::Row>(shape);
#endif
#ifdef FUSE_BETA
    ckl::add<
        ckl::input(
            beta_input_dfb, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
        ckl::input(dfb::beta, gamma_beta_wait, gamma_beta_pop, ckl::OperandKind::Block),
        ckl::output(dfb::out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize),
        ckl::BroadcastDim::Row>(shape);
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

    dfb_eps.wait_front(1);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        norm::kernel_util::compute::combine_welford_partials(
            dfb_stats,
            dfb_stats_reduced,
            stats_tiles_cols,
            [W](uint32_t) { return static_cast<float>(W); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        dfb_stats_reduced.push_back(stats_tile_stride);
        dfb_stats_reduced.wait_front(stats_tile_stride);

        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(
                    dfb::stats_reduced,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::input(dfb::eps, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{1u, 0u},
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

        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);
    }
    dfb_eps.pop_front(1);
}
