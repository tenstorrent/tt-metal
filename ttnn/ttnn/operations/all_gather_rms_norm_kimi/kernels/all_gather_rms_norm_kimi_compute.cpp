// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute core (q, m): block of block_h x block_w tiles of slice q (op_design.md -> Block Operation Realization).
//   partial_block (a)   (q == p) eltwise_chain BinaryFpu x*x, DestAccumulation::PerRow (fp32 DEST) -> cb_xsq_sum
//   partial_block (b)   (q == p) reduce<SUM, REDUCE_ROW, Accurate fp32 SFPU> -> cb_partial (col 0 = row sums)
//   combine_stats_block eltwise_chain: CopyTile P_0..P_{G-1} -> D0..D{G-1} (UnpackToDestFp32, lossless), then
//                       D0 += D1, D0 += D2, ... in the fixed order q = 0..G-1 (SFPU fp32), * 1/W, + eps, rsqrt
//                       -> cb_rstd
//   expand_rstd_block   unary_bcast<Col> -> cb_rstd_full
//   scale_block         S = rstd_full (Col mapping) * gamma (row broadcast) -> cb_scale (fp32), computed BEFORE the
//                       block lands (the stall shadow)
//   normalize_block     out = x * S, x waited cumulatively as the chunks land -> cb_out
// The one multiply per landed tile is all that is left on the tail.

#include <cstdint>
#include <utility>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "all_gather_rms_norm_kimi_common.hpp"

using namespace compute_kernel_lib;
using namespace agrn;

constexpr uint32_t C = NUM_GEO_CT;
constexpr uint32_t inv_w_bits = get_compile_time_arg_val(C + 0);
constexpr uint32_t eps_bits = get_compile_time_arg_val(C + 1);

constexpr uint32_t cb_x_block = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_out = 2;
constexpr uint32_t cb_partials = 3;
constexpr uint32_t cb_xsq_sum = 4;
constexpr uint32_t cb_rstd = 5;
constexpr uint32_t cb_partial = 6;
constexpr uint32_t cb_rstd_full = 7;
constexpr uint32_t cb_scale = 8;
constexpr uint32_t cb_scaler = 9;

static_assert(G <= DEST_AUTO_LIMIT, "combine_stats_block holds all G partials in DEST at once");

// Partial q (q >= 1) of a row: tile q * h + r of cb_partials (caller-managed window) -> DEST slot q.
template <uint32_t Q>
using PartialLoad = CopyTile<
    input(
        cb_partials,
        WaitPolicy::None,
        PopPolicy::None,
        InputTileMapping::Block,
        DataFormatReconfig::Enabled,
        TileAddressing::Offset),
    static_cast<Dst>(Q)>;
template <uint32_t Q>
using PartialAdd = AddBinary<Dst::D0, static_cast<Dst>(Q), Dst::D0>;

// Q = 0..G-2: load P_0..P_{G-1} into D0..D{G-1}, then D0 += D{Q+1} in increasing q (fixed order, every chip identical).
template <uint32_t... Q>
ALWI void combine_stats_block(uint32_t h, std::integer_sequence<uint32_t, Q...>) {
    eltwise_chain(
        IterationShape::tiles(h),
        PartialLoad<Q>{Q * h}...,
        PartialLoad<G - 1>{(G - 1) * h},
        PartialAdd<Q + 1>{}...,
        MulUnary<>{inv_w_bits},
        AddUnary<>{eps_bits},
        Rsqrt<>{},
        PackTile<output(cb_rstd)>{});
}

void kernel_main() {
    const uint32_t p = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t row_base_idx = 1;
    const BlockId b = block_id(get_absolute_logical_x(), get_absolute_logical_y(), row_base_idx, [](uint32_t i) {
        return get_common_arg_val<uint32_t>(i);
    });
    const uint32_t h = b.h;

    compute_kernel_hw_startup(cb_x_block, cb_x_block, cb_out);

    if (b.q == p) {
        // ---- partial_block (a): sum_w x^2 per tile row, accumulated in fp32 DEST ----
        eltwise_chain(
            IterationShape::grid(h, block_w),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(cb_x_block, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                input(cb_x_block, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                Dst::D0,
                DestAccumulation::PerRow>{},
            PackTile<output(
                cb_xsq_sum,
                ReservePolicy::PerOuter,
                PushPolicy::PerOuter,
                DataFormatReconfig::Enabled,
                TileAddressing::Direct,
                DestAccumulation::PerRow)>{});
        // ---- partial_block (b): collapse the 32 columns of each row tile -> col 0 (fp32 SFPU) ----
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_xsq_sum,
            cb_scaler,
            cb_partial,
            ReduceInputPolicy::BulkWaitBulkPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Accurate>(ReduceInputBlockShape::of(h, 1));
    }

    // ---- combine_stats_block: ss = ((P_0 + P_1) + P_2) + ... ; rstd = rsqrt(ss / W + eps) ----
    cb_wait_front(cb_partials, G * h);  // caller-managed window (None, None) of the G loads
    combine_stats_block(h, std::make_integer_sequence<uint32_t, G - 1>{});
    cb_pop_front(cb_partials, G * h);

    // ---- expand_rstd_block: column-broadcast rstd over the tile ----
    unary_bcast<BroadcastDim::Col, input(cb_rstd), output(cb_rstd_full)>(IterationShape::tiles(h));

    // ---- scale_block: S = rstd (per row) x gamma (per column) ----
    mul<input(cb_rstd_full, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
        input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Row),
        output(cb_scale)>(IterationShape::grid(h, block_w));

    // ---- normalize_block: out = x * S as the chunks land ----
    mul<input(cb_x_block, WaitPolicy::Cumulative, PopPolicy::AtEnd, InputTileMapping::Block),
        input(cb_scale, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
        output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>(
        IterationShape::grid(h, block_w).block_size(chunk_tiles));
}
