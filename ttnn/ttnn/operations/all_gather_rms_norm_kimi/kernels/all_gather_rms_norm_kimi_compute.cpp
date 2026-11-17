// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute core (q, m): block of block_h x block_w tiles of slice q (op_design.md -> Block Operation Realization).
//   partial_block (a)   (q == p) eltwise_chain BinaryFpu x*x, DestAccumulation::PerRow (fp32 DEST) -> cb_xsq_sum
//   partial_block (b)   (q == p) reduce<SUM, REDUCE_ROW, Accurate fp32 SFPU> -> cb_partial (col 0 = row sums)
//   combine_stats_block eltwise_chain: CopyTile P_0..P_{G-1} -> D0..D{G-1} (UnpackToDestFp32, lossless), then
//                       D0 += D1, D0 += D2, ... in the fixed order q = 0..G-1 (SFPU fp32), * 1/W, + eps, rsqrt
//                       -> cb_rstd; COMBINE_COL0: all of it as one pass over the column-0 vectors only
//   gamma_block         xg = x * gamma (row broadcast, fp32 DEST, exact bf16 x bf16 product) -> cb_xg, per landed
//                       chunk; needs no stats, so it runs BEFORE the combine for the whole own block and for the first
//                       remote chunks the host look-ahead table expects landed by then (the stall shadow)
//   expand_rstd_block   unary_bcast<Col> -> cb_rstd_full (one tile per block row)
//   normalize_block     per chunk: out = xg * rstd_full -> cb_out; a chunk not done ahead runs x * gamma * rstd_full
//                       as one chain (DEST reused as srcA, FUSE_LATE_CHUNKS); a block with no chunk ahead runs one
//                       gamma pass + one normalize pass over the whole block (LATE_BLOCK). NORMALIZE_MODE "scale"
//                       instead computes S = rstd_full * gamma after the combine and out = x * S per landed chunk
// After the stats only the combine and one multiply per tile stand between the landed chunks and their DRAM store,
// so the stores (DRAM-bound, ~4 us for the whole output) start right after the combine instead of after a whole-
// block scale. Each chunk is one (rows x cols) window of the block: a run inside one block row, or whole rows.

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
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using namespace compute_kernel_lib;
using namespace agrn;

constexpr uint32_t C = NUM_GEO_CT;
constexpr uint32_t inv_w_bits = get_compile_time_arg_val(C + 0);
constexpr uint32_t eps_bits = get_compile_time_arg_val(C + 1);

// CB indices: defined once on the host (CB_* in the program descriptor), passed in this order
constexpr uint32_t cb_x_block = get_compile_time_arg_val(C + 2);
constexpr uint32_t cb_gamma = get_compile_time_arg_val(C + 3);
constexpr uint32_t cb_out = get_compile_time_arg_val(C + 4);
constexpr uint32_t cb_partials = get_compile_time_arg_val(C + 5);
constexpr uint32_t cb_xsq_sum = get_compile_time_arg_val(C + 6);
constexpr uint32_t cb_rstd = get_compile_time_arg_val(C + 7);
constexpr uint32_t cb_partial = get_compile_time_arg_val(C + 8);
constexpr uint32_t cb_xg = get_compile_time_arg_val(C + 9);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(C + 10);
constexpr uint32_t cb_rstd_full = get_compile_time_arg_val(C + 11);
constexpr bool fuse_late = get_compile_time_arg_val(C + 12) != 0;  // x * gamma * rstd in one chain for late chunks
constexpr bool combine_col0 = get_compile_time_arg_val(C + 13) != 0;  // combine scoped to column 0 (one SFPU pass)
// normalize mode: false = xg (x * gamma ahead of the stats, out = xg * rstd per chunk); true = scale (S = rstd * gamma
// right after the combine, out = x * S per chunk: ONE pass per landed chunk)
constexpr bool scale_mode = get_compile_time_arg_val(C + 14) != 0;
// xg mode, a block with no chunk ahead (all of it lands late): one gamma pass + one normalize pass over the whole
// block (cumulative waits, pushed per chunk) instead of a gamma + normalize pass per chunk
constexpr bool late_block = get_compile_time_arg_val(C + 15) != 0;

static_assert(block_w % chunk_tiles == 0 || chunk_tiles % block_w == 0, "a chunk is a run in one row or whole rows");

// Chunk k of the block as a (rows x cols) window at (r0, c0); chunk tiles are row-major-contiguous in the block.
struct ChunkWindow {
    uint32_t rows, cols, r0, c0;
};
ALWI ChunkWindow chunk_window(uint32_t k) {
    const uint32_t t0 = k * chunk_tiles;
    if constexpr (chunk_tiles <= block_w) {
        return {1, chunk_tiles, t0 / block_w, t0 % block_w};
    } else {
        return {chunk_tiles / block_w, block_w, t0 / block_w, 0};
    }
}

// xg chunk k = x chunk k (landing, caller-managed cumulative wait) * gamma (row broadcast, caller-held) -> cb_xg
ALWI void gamma_chunk(uint32_t k) {
    const ChunkWindow w = chunk_window(k);
    cb_wait_front(cb_x_block, (k + 1) * chunk_tiles);
    eltwise_chain(
        IterationShape::grid(w.rows, w.cols).block_size(chunk_tiles),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(
                cb_x_block,
                WaitPolicy::None,
                PopPolicy::None,
                InputTileMapping::Block,
                DataFormatReconfig::Enabled,
                TileAddressing::Offset),
            input(
                cb_gamma,
                BroadcastDim::Row,
                WaitPolicy::None,
                PopPolicy::None,
                InputTileMapping::Row,
                DataFormatReconfig::Enabled,
                TileAddressing::Offset)>{k * chunk_tiles, w.c0},
        PackTile<output(cb_xg, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
}

// rstd_full tile of a chunk row: caller-held cb_rstd_full, one tile per block row (Col mapping), offset to the chunk
constexpr InputSpec rstd_full_in = input(
    cb_rstd_full,
    WaitPolicy::None,
    PopPolicy::None,
    InputTileMapping::Col,
    DataFormatReconfig::Enabled,
    TileAddressing::Offset);

// out chunk k = xg chunk k (front of cb_xg) * rstd_full of its rows -> cb_out
ALWI void normalize_chunk(uint32_t k) {
    const ChunkWindow w = chunk_window(k);
    eltwise_chain(
        IterationShape::grid(w.rows, w.cols).block_size(chunk_tiles),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(cb_xg, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            rstd_full_in>{0u, w.r0},
        PackTile<output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
}

// out chunk k = (x chunk k * gamma) * rstd_full in one chain: the product stays in DEST (fp32) and is reused as srcA
ALWI void fused_chunk(uint32_t k) {
    const ChunkWindow w = chunk_window(k);
    cb_wait_front(cb_x_block, (k + 1) * chunk_tiles);
    eltwise_chain(
        IterationShape::grid(w.rows, w.cols).block_size(chunk_tiles),
        BinaryFpu<
            BinaryFpuOp::Mul,
            input(
                cb_x_block,
                WaitPolicy::None,
                PopPolicy::None,
                InputTileMapping::Block,
                DataFormatReconfig::Enabled,
                TileAddressing::Offset),
            input(
                cb_gamma,
                BroadcastDim::Row,
                WaitPolicy::None,
                PopPolicy::None,
                InputTileMapping::Row,
                DataFormatReconfig::Enabled,
                TileAddressing::Offset)>{k * chunk_tiles, w.c0},
        DestReuseBinary<BinaryFpuOp::Mul, rstd_full_in, DestReuseType::DEST_TO_SRCA>{w.r0},
        PackTile<output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
}

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

// Column-0 tail of the combine: rstd = rsqrt(((P_0 + P_1) + ...) + P_{G-1}) * 1/W + eps) on the even-parity vectors
// of the two left faces only (8 SFPU vector ops instead of 32 per pass, and one pass instead of G + 2): the partial
// tiles carry their row sums in column 0 and every consumer of cb_rstd reads only column 0 (unary_bcast<Col>). The
// sum keeps the fixed q order. VectorMode::C runs the body on face 0 then face 2; the body advances dst_reg by 8
// (the stock ITERATIONS = 8), skipping the odd-parity vectors that never hold column 0 (examples/sfpu_tile_scope,
// c_skip). Raw sfpi: no helper scopes an SFPU op to a column, and none fuses a G-way DEST sum with rsqrt.
#ifdef TRISC_MATH
sfpi_inline void combine_col0_body(uint32_t inv_w, uint32_t eps) {
    constexpr uint32_t dst_tile_size_sfpi = 32;  // one DEST tile in sfpi rows (as ckernel_sfpu_binary.h)
#pragma GCC unroll 4
    for (uint32_t rg = 0; rg < 4; ++rg) {
        sfpi::vFloat s = sfpi::dst_reg[0];
#pragma GCC unroll 8
        for (uint32_t q = 1; q < G; ++q) {
            s = s + sfpi::dst_reg[q * dst_tile_size_sfpi];
        }
        s = s * ckernel::sfpu::Converter::as_float(inv_w);
        s = s + ckernel::sfpu::Converter::as_float(eps);
        sfpi::dst_reg[0] = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(s);
        sfpi::dst_reg += 2;
    }
}
#endif

struct CombineCol0Rsqrt : UnaryOp<CombineCol0Rsqrt, Dst::D0> {
    static constexpr uint32_t lane_width = G;  // reads D0..D{G-1}
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            combine_col0_body, slot_offset, ckernel::VectorMode::C, inv_w_bits, eps_bits)));
    }
};

// Q = 0..G-2: load P_0..P_{G-1} into D0..D{G-1}, then (combine_col0) one column-0 pass, or the whole-tile chain
// D0 += D{Q+1} in increasing q (fixed order, every chip identical), * 1/W, + eps, rsqrt.
template <uint32_t... Q>
ALWI void combine_stats_block(uint32_t h, std::integer_sequence<uint32_t, Q...>) {
    if constexpr (combine_col0) {
        eltwise_chain(
            IterationShape::tiles(h),
            PartialLoad<Q>{Q * h}...,
            PartialLoad<G - 1>{(G - 1) * h},
            CombineCol0Rsqrt{},
            PackTile<output(cb_rstd)>{});
    } else {
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

    // ---- gamma_block ahead of the stats: the whole own block, the early-stream chunks of a remote one ----
    const uint32_t n_chunks = h * block_w / chunk_tiles;
    // Every member must produce bit-identical output for a slice, so all cores of a slice use one arithmetic: with
    // fuse_late no chunk goes through cb_xg (the fused chain keeps x * gamma in DEST, the cb_xg path rounds it via L1).
    // ahead = this core's entry of the host look-ahead table (common args after row_base, 8 bits per core)
    const uint32_t core_idx = b.q * nb + b.m;
    const uint32_t ahead_tab = get_common_arg_val<uint32_t>(row_base_idx + grid_y + core_idx / 4);
    const uint32_t ahead = (fuse_late || scale_mode) ? 0 : ((ahead_tab >> (8 * (core_idx % 4))) & 0xFF);
    cb_wait_front(cb_gamma, block_w);  // held for every chunk (Row-mapped, offset per chunk)
    for (uint32_t k = 0; k < ahead; ++k) {
        gamma_chunk(k);
    }

    // ---- combine_stats_block: ss = ((P_0 + P_1) + P_2) + ... ; rstd = rsqrt(ss / W + eps) ----
    cb_wait_front(cb_partials, G * h);  // caller-managed window (None, None) of the G loads
    combine_stats_block(h, std::make_integer_sequence<uint32_t, G - 1>{});
    cb_pop_front(cb_partials, G * h);

    // ---- expand_rstd_block: column-broadcast rstd over the tile (one tile per block row) ----
    unary_bcast<BroadcastDim::Col, input(cb_rstd), output(cb_rstd_full)>(IterationShape::tiles(h));

    if constexpr (scale_mode) {
        // ---- scale_block: S = rstd (per row, Col mapping) * gamma (per column, row broadcast) -> cb_xg (fp32) ----
        mul<input(cb_rstd_full, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
            input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Row),
            output(cb_xg)>(IterationShape::grid(h, block_w));
        // ---- normalize_block: out = x * S, x waited cumulatively as the chunks land, pushed per chunk ----
        mul<input(cb_x_block, WaitPolicy::Cumulative, PopPolicy::AtEnd, InputTileMapping::Block),
            input(cb_xg, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>(
            IterationShape::grid(h, block_w).block_size(chunk_tiles));
        return;
    }

    // ---- normalize_block: per chunk, out = xg * rstd (x * gamma * rstd for the chunks not done ahead) ----
    cb_wait_front(cb_rstd_full, h);  // held for every chunk (Col-mapped, offset per chunk)
    if (late_block && !fuse_late && ahead == 0) {
        const IterationShape block = IterationShape::grid(h, block_w).block_size(chunk_tiles);
        mul<input(cb_x_block, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block),
            input(cb_gamma, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row),
            output(cb_xg, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>(block);
        mul<input(cb_xg, WaitPolicy::Cumulative, PopPolicy::AtEnd, InputTileMapping::Block),
            input(cb_rstd_full, WaitPolicy::None, PopPolicy::None, InputTileMapping::Col),
            output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>(block);
    }
    for (uint32_t k = (late_block && !fuse_late && ahead == 0) ? n_chunks : 0; k < n_chunks; ++k) {
        if (k < ahead) {
            normalize_chunk(k);
        } else if constexpr (fuse_late) {
            fused_chunk(k);
        } else {
            gamma_chunk(k);
            normalize_chunk(k);
        }
    }
    cb_pop_front(cb_rstd_full, h);
    cb_pop_front(cb_gamma, block_w);
    cb_pop_front(cb_x_block, h * block_w);
}
