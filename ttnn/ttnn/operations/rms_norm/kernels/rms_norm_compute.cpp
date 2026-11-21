// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute: one NEO of C. It sees its own slice of every STRIDED ring, so it is the single-threaded
// kernel over this NEO's units (`rows_per_block` tile-rows x Wt tiles each): the dealing lives in the reader,
// the writer and the host split. Every NEO runs the same `rounds`.
//
// Per unit, four named block operations, all kernel-library helpers:
//   sum_squares  square over grid(rows_per_block, Wt), accumulated in DEST per tile-row (one pack per row);
//                the unit is waited cumulatively and NOT popped: it stays resident for normalize.
//   reduce_row   reduce<SUM, REDUCE_ROW> of each row's partial tile (front tile only: WaitAndPopPerTile).
//   inv_rms      rsqrt(row_sum * (1/W) + epsilon) per row: CopyTile, FillScalar + MulBinary (1/W),
//                FillScalar + AddBinary (epsilon), Rsqrt. The scalar-parameter elements (MulUnary, AddUnary)
//                are not used: the scalar goes through a DEST fill instead.
//   normalize    x * inv_rms (FPU, column broadcast) * gamma (row broadcast to DEST, SFPU multiply); the
//                input and gamma are popped per tile, inv_rms held for the unit and popped at its end.
//
// pack_init before every helper whose output ring differs from the previous one's. reduce<> re-aims it itself.
//
// Test hook (CT arg tag_threads): NEO t multiplies its output by t + 1, so a test sees which NEO ran.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);
    const float inv_w = __builtin_bit_cast(float, static_cast<uint32_t>(get_arg(args::inv_w_bits)));
    const float epsilon = __builtin_bit_cast(float, static_cast<uint32_t>(get_arg(args::eps_bits)));
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t rows_per_block = get_arg(args::rows_per_block);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr bool tag_threads = get_arg(args::tag_threads) != 0;
    const uint32_t rounds = (num_units + C - 1) / C;  // the same for every NEO of the cluster

    DataflowBuffer input_tiles(dfb::input_tiles);
    DataflowBuffer gamma_tiles(dfb::gamma_tiles);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer row_stat(dfb::row_stat);
    DataflowBuffer row_sum(dfb::row_sum);
    DataflowBuffer output_tiles(dfb::output_tiles);

    compute_kernel_hw_startup(dfb::input_tiles, dfb::scaler, dfb::output_tiles);

    constexpr auto unit_shape = ckl::IterationShape::grid(rows_per_block, Wt);
    constexpr auto row_shape = ckl::IterationShape::tiles(rows_per_block);

    // sum_squares: the unit's tiles at their index, waited cumulatively (starts on the first landed tile).
    constexpr auto resident_unit = ckl::input(
        dfb::input_tiles, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
    constexpr auto row_partials = ckl::output(
        dfb::row_stat,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Direct,
        ckl::DestAccumulation::PerRow);
    // normalize: the resident unit consumed front-first; inv_rms held for the unit, one tile per row.
    constexpr auto next_input = ckl::input(
        dfb::input_tiles, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::InputTileMapping::Scalar);
    constexpr auto inv_rms_per_row = ckl::input(
        dfb::row_stat,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::AtEnd,
        ckl::InputTileMapping::Col);
    constexpr auto next_gamma = ckl::input(
        dfb::gamma_tiles, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::InputTileMapping::Scalar);

    using NormalizeMul = ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, next_input, inv_rms_per_row, ckl::Dst::D0>;
    using GammaBcast = ckl::UnaryBcast<ckl::BroadcastDim::Row, next_gamma, ckl::Dst::D1>;
    using ApplyGamma = ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>;
    using PackOut = ckl::PackTile<ckl::output(dfb::output_tiles)>;

    for (uint32_t round = 0; round < rounds; ++round) {
        // sum_squares_block: per tile-row, sum_w x_w^2 in DEST -> one row_stat tile.
        pack_init(dfb::row_stat);
        ckl::square<resident_unit, row_partials>(unit_shape, input_tiles, row_stat);

        // reduce_row_block: SUM over the 32 columns of each partial -> row_sum (column 0).
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::row_stat, dfb::scaler, dfb::row_sum>(
            row_stat, scaler, row_sum, ckl::ReduceInputBlockShape::col(rows_per_block));

        // inv_rms_block: rsqrt(sum * (1/W) + epsilon) -> row_stat. Epsilon is added to the mean (rule R7).
        pack_init(dfb::row_stat);
        ckl::eltwise_chain(
            row_shape,
            ckl::CopyTile<ckl::input(dfb::row_sum), ckl::Dst::D0>{row_sum},
            ckl::FillScalar<ckl::Dst::D1>{inv_w},
            ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
            ckl::FillScalar<ckl::Dst::D1>{epsilon},
            ckl::AddBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb::row_stat)>{row_stat});

        // normalize_block: x * inv_rms (col bcast) * gamma (row bcast) -> output_tiles.
        pack_init(dfb::output_tiles);
        if constexpr (tag_threads) {
            const float tag = static_cast<float>(get_my_thread_id() + 1);
            ckl::eltwise_chain(
                unit_shape,
                NormalizeMul{input_tiles, row_stat},
                GammaBcast{gamma_tiles},
                ApplyGamma{},
                ckl::FillScalar<ckl::Dst::D1>{tag},
                ApplyGamma{},
                PackOut{output_tiles});
        } else {
            ckl::eltwise_chain(
                unit_shape, NormalizeMul{input_tiles, row_stat}, GammaBcast{gamma_tiles}, ApplyGamma{}, PackOut{output_tiles});
        }
    }
    // reduce<> waits on the scaler but never pops it: release it after the last round.
    scaler.pop_front(1);
}
