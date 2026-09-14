// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (TRISC unpack/math/pack).
//
// Per block of `rows x core_w_tiles` tiles (op_design.md Block schedule):
//   [tilize_x_block]   RM input: 32*rows sticks -> rows*Wc tiles                  (tilize helper)
//   A sumsq_block      x*x accumulated in DEST per tile-row -> rows fp32 tiles      (sum_of_squares / eltwise_chain)
//   B collapse_block   within-tile row sum -> rows column-0-valid tiles            (reduce, ReduceTile)
//   C combine_block    root (or Cw==1): sum of the num_partials slots -> rsqrt(sum/W + eps) on column 0 (reduce,
//   AccumulateViaAdd+Skip, fused col-0 sfpi post-op) D normalize_block  x (.) bcast_col(rstd) -> cb_normed |
//   cb_output_tiles        (mul, BroadcastDim::Col) E scale_block      normed (.) bcast_row(gamma_slice) ->
//   cb_output_tiles        (mul, BroadcastDim::Row) [untilize_x_block] RM output: rows*Wc tiles -> sticks (untilize
//   helper)
//
// R3 (WIDTH_SHARDED): x and out are the resident shard buffers. The chains address them with
// TileOffset::Strided (base = block_idx * block_rows * shard_w_tiles, row stride = shard_w_tiles)
// under caller-managed (None, None) CB policies: the shard is published once by the reader and
// never popped; the output shard is reserved once up front and pushed once at the end. The walk
// covers only this core's core_w_tiles valid columns, so a ragged (padded) last shard never feeds
// its padding into the sum of squares.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"  // _calculate_sqrt_body_: the rsqrt body rsqrt_tile runs
#endif

using namespace compute_kernel_lib;

// Perf ablation (measurement only, never set in production): RMS_NORM_ABLATE_FINALIZE stubs the
// combine post-op's SFPU payload (mean, +eps, rsqrt) while the reduce's DEST window, pack and CB
// scaffolding stay intact, so the ablated variant measures the collective without the finalize.
#ifndef RMS_NORM_ABLATE_FINALIZE
#define RMS_NORM_ABLATE_FINALIZE 0
#endif

// RAW LLK — combine post-op (perf_experiments/finalize_col0_sfpu; measured 990.6 -> 227.6 ns per call at
// 16-bit DEST, 966.6 -> 221.8 at fp32 DEST, whole-op -1080 ns ablation ceiling on the perf-flagged decode
// shape). The reduce's Skip contract leaves DEST holding the raw column-0-valid SUM, and the only consumer
// reads column 0 (BroadcastDim::Col), so the three stock full-tile passes (mul_unary_tile, add_unary_tile,
// rsqrt_tile: VectorMode::RC x ITERATIONS=8 = 96 vector ops, two inits) are replaced by ONE sfpi pass over
// the 8 even-parity vectors of the left faces: SFPMAD(sum, 1/W, eps) -> the stock
// _calculate_sqrt_body_<APPROX, RECIPROCAL> -> RNE to bf16 under 16-bit DEST. Same precision knobs (APPROX,
// DST_ACCUM_MODE) and fewer intermediate truncations (max rel err 3.8e-3 vs 5.1e-3 at 16-bit DEST).
// The face walk is hand-rolled with an opaque trip count: the LLK VectorMode::C branch is a 2-trip loop the
// compiler unrolls, and two interleaved MAD+rsqrt bodies overflow the SFPU register file (sfpi cannot
// spill -> ICE). Columns 1..31 of the rstd tile keep the raw sum; nothing reads them.
#ifdef TRISC_MATH
sfpi_inline void finalize_rstd_col0_body(uint32_t inv_w_bits, uint32_t eps_bits) {
    const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
#pragma GCC unroll 4
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        x = x * inv_w + eps;
        sfpi::vFloat y = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(x);
        if constexpr (!DST_ACCUM_MODE) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg += 2;  // skip the odd-parity vector (columns 1,3,..,15 — never column 0)
    }
}
#endif

ALWI void finalize_rstd_col0(uint32_t dst, uint32_t inv_w_bits, uint32_t eps_bits) {
    MATH(({
        uint32_t num_faces;
        asm volatile("li %0, 2" : "=r"(num_faces));  // opaque 2: keeps the face loop a loop (see above)
        _llk_math_eltwise_sfpu_start_(dst);
        for (uint32_t face = 0; face < num_faces; ++face) {  // faces 0 and 2 == VectorMode::C
            finalize_rstd_col0_body(inv_w_bits, eps_bits);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
        _llk_math_eltwise_sfpu_done_();
    }));
}

void kernel_main() {
    // ---- named compile-time args ----
    constexpr uint32_t cb_x_tiles = get_named_compile_time_arg_val("CB_X_TILES");
    constexpr uint32_t cb_x_sticks = get_named_compile_time_arg_val("CB_X_STICKS");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("CB_SCALER");
    constexpr uint32_t cb_sumsq_partial = get_named_compile_time_arg_val("CB_SUMSQ_PARTIAL");
    constexpr uint32_t cb_partial_collapsed = get_named_compile_time_arg_val("CB_PARTIAL_COLLAPSED");
    constexpr uint32_t cb_gather = get_named_compile_time_arg_val("CB_GATHER");
    constexpr uint32_t cb_rstd_handoff = get_named_compile_time_arg_val("CB_RSTD_HANDOFF");
    constexpr uint32_t cb_rstd = get_named_compile_time_arg_val("CB_RSTD");
    constexpr uint32_t cb_gamma_tiles = get_named_compile_time_arg_val("CB_GAMMA_TILES");
    constexpr uint32_t cb_gamma_sticks = get_named_compile_time_arg_val("CB_GAMMA_STICKS");
    constexpr uint32_t cb_normed = get_named_compile_time_arg_val("CB_NORMED");
    constexpr uint32_t cb_output_tiles = get_named_compile_time_arg_val("CB_OUTPUT_TILES");
    constexpr uint32_t cb_out_sticks = get_named_compile_time_arg_val("CB_OUT_STICKS");
    constexpr bool input_rm = get_named_compile_time_arg_val("INPUT_RM") != 0;
    constexpr uint32_t gamma_mode = get_named_compile_time_arg_val("GAMMA_MODE");  // 0 none, 1 TILE, 2 RM
    constexpr bool has_gamma = gamma_mode != 0;
    constexpr bool sharded = get_named_compile_time_arg_val("SHARDED") != 0;
    constexpr uint32_t shard_w_tiles = get_named_compile_time_arg_val("SHARD_W_TILES");  // R3 in-shard row stride
    constexpr uint32_t num_w_splits = get_named_compile_time_arg_val("NUM_W_SPLITS");
    constexpr uint32_t core_w_tiles = get_named_compile_time_arg_val("CORE_W_TILES");

    // Cw == 1: B packs straight into the gather CB and C straight into the rstd CB (no handoffs).
    constexpr uint32_t cb_collapse_out = (num_w_splits == 1) ? cb_gather : cb_partial_collapsed;
    constexpr uint32_t cb_rstd_out = (num_w_splits == 1) ? cb_rstd : cb_rstd_handoff;
    constexpr uint32_t cb_norm_out = has_gamma ? cb_normed : cb_output_tiles;

    // ---- runtime args ----
    const bool is_root = get_arg_val<uint32_t>(0) != 0;
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);
    const uint32_t block_rows = get_arg_val<uint32_t>(2);
    const uint32_t last_block_rows = get_arg_val<uint32_t>(3);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);
    const uint32_t tensor_row_tiles = get_arg_val<uint32_t>(6);
    const uint32_t num_partials = get_arg_val<uint32_t>(7);  // gather slots per row = active W slices

    compute_kernel_hw_startup(cb_x_tiles, cb_scaler, cb_output_tiles);

    // ---- publish_x_shard (R3): the whole resident shard is the block source; reserve the whole output shard ----
    if constexpr (sharded) {
        MaybeDeviceZoneScope("compute_shard_wait");
        cb_wait_front(cb_x_tiles, tensor_row_tiles * shard_w_tiles);
        cb_reserve_back(cb_output_tiles, tensor_row_tiles * shard_w_tiles);
    }

    // combine_block post-op: DEST holds the raw cross-core SUM (Skip: no SFPU armed) -> rsqrt(sum/W + eps) on
    // column 0. The post-op owns its SFPU init (Skip contract): rsqrt_tile_init programs the sqrt constants.
    const auto finalize_rstd = [inv_w_bits, eps_bits]([[maybe_unused]] uint32_t dst) {
        if constexpr (RMS_NORM_ABLATE_FINALIZE) {
            return;
        }
        rsqrt_tile_init();
        finalize_rstd_col0(dst, inv_w_bits, eps_bits);
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
        const auto shape = IterationShape::grid(rows, core_w_tiles);
        // R3: this block's tiles inside the resident shard — rows of shard_w_tiles, core_w_tiles valid each.
        [[maybe_unused]] const StridedTileRange shard_range{block_idx * block_rows * shard_w_tiles, shard_w_tiles};

        if constexpr (input_rm) {
            // The helper waits per tile-row on cb_x_sticks, so this zone is tilize occupancy (wait + work).
            MaybeDeviceZoneScope("compute_x_tilize");
            tilize<core_w_tiles, cb_x_sticks, cb_x_tiles>(rows);
        }

        // Split the x arrival wait from sumsq: sum_of_squares waits Upfront on the same count, so the
        // explicit wait is idempotent and A's zone below is payload only.
        if constexpr (!sharded) {
            MaybeDeviceZoneScope("compute_x_wait");
            cb_wait_front(cb_x_tiles, rows * core_w_tiles);
        }

        // A: sumsq_block — x*x accumulated in DEST per tile-row; x stays resident for D.
        {
            MaybeDeviceZoneScope("compute_sumsq");
            if constexpr (sharded) {
                eltwise_chain(
                    shape,
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Strided),
                        input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Strided),
                        Dst::D0,
                        DestAccumulation::PerRow>{shard_range, shard_range},
                    PackTile<output(
                        cb_sumsq_partial,
                        ReservePolicy::PerOuter,
                        PushPolicy::PerOuter,
                        DataFormatReconfig::Enabled,
                        PackRelu::Disabled,
                        L1Accumulation::Disabled,
                        DestAccumulation::PerRow)>{});
            } else {
                sum_of_squares<
                    input(cb_x_tiles, WaitPolicy::Upfront, PopPolicy::None, OperandKind::Block),
                    row_output(cb_sumsq_partial)>(shape);
            }
        }

        // B: collapse_block — within-tile row sum -> column-0-valid partial per tile-row.
        {
            MaybeDeviceZoneScope("compute_collapse");
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sumsq_partial,
                cb_scaler,
                cb_collapse_out,
                ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(rows, 1));
        }

        // C: combine_block — root (or Cw == 1): sum the Cw already-collapsed slots, then finalize.
        if (num_w_splits == 1 || is_root) {
            {
                // Cw > 1: the gather rendezvous (all peers' partials landed) — the reduce below waits on
                // the same count, so hoisting the wait makes C's zone payload only.
                MaybeDeviceZoneScope("compute_gather_wait");
                cb_wait_front(cb_gather, rows * num_partials);
            }
            MaybeDeviceZoneScope("compute_combine");
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_gather,
                cb_scaler,
                cb_rstd_out,
                ReduceInputPolicy::BulkWaitBulkPop,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ReduceAlgorithm::AccumulateViaAdd,
                ReduceWithinTile::Skip>(
                ReduceInputBlockShape::of(rows, num_partials),
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                finalize_rstd);
        }

        // D: normalize_block — x (.) bcast_col(rstd); releases x (interleaved) and rstd.
        {
            // Cw > 1: the rstd multicast round trip; the chain waits Upfront on the same count.
            MaybeDeviceZoneScope("compute_rstd_wait");
            cb_wait_front(cb_rstd, rows);
        }

        // load_gamma_slice (RM gamma), first block only: one 32-stick block -> Wc gamma tiles, resident for the
        // kernel. Placed here rather than at kernel start because the reader publishes gamma after x (reader
        // order), and no later than here because cb_gamma_sticks aliases cb_normed's allocation: the sticks
        // must be consumed before D packs the first normed tile.
        if constexpr (gamma_mode == 2) {
            if (block_idx == 0) {
                MaybeDeviceZoneScope("compute_gamma_tilize");
                tilize<core_w_tiles, cb_gamma_sticks, cb_gamma_tiles>(1);
            }
        }
        {
            MaybeDeviceZoneScope("compute_normalize");
            if constexpr (sharded) {
                if constexpr (has_gamma) {
                    eltwise_chain(
                        shape,
                        BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(
                                cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Strided),
                            input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col)>{
                            shard_range},
                        PackTile<output(cb_normed)>{});
                } else {
                    eltwise_chain(
                        shape,
                        BinaryFpu<
                            BinaryFpuOp::Mul,
                            input(
                                cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Strided),
                            input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col)>{
                            shard_range},
                        PackTile<output(cb_output_tiles, ReservePolicy::None, PushPolicy::None, TileOffset::Strided)>{
                            shard_range});
                }
            } else {
                mul<input(cb_x_tiles, WaitPolicy::None, PopPolicy::AtEnd, OperandKind::Block),
                    input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col),
                    output(cb_norm_out)>(shape);
            }
        }

        // E: scale_block — normed (.) bcast_row(gamma slice); gamma stays resident.
        if constexpr (has_gamma) {
            MaybeDeviceZoneScope("compute_scale");
            if constexpr (sharded) {
                eltwise_chain(
                    shape,
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_normed),
                        input(
                            cb_gamma_tiles,
                            BroadcastDim::Row,
                            WaitPolicy::Upfront,
                            PopPolicy::None,
                            OperandKind::Row)>{},
                    PackTile<output(cb_output_tiles, ReservePolicy::None, PushPolicy::None, TileOffset::Strided)>{
                        shard_range});
            } else {
                mul<input(cb_normed),
                    input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, OperandKind::Row),
                    output(cb_output_tiles)>(shape);
            }
        }

        if constexpr (input_rm) {
            MaybeDeviceZoneScope("compute_untilize");
            untilize<core_w_tiles, cb_output_tiles, cb_out_sticks>(rows);
        }
    }

    if constexpr (sharded) {
        cb_push_back(cb_output_tiles, tensor_row_tiles * shard_w_tiles);
    }
}
