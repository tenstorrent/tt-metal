// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute (TRISC unpack/math/pack).
//
// Per block of `rows x core_w_tiles` tiles (op_design.md Block schedule):
//   [tilize_x_block]   RM input: 32*rows sticks -> rows*Wc tiles                  (tilize helper)
//   A sumsq_block      x*x accumulated in DEST per tile-row -> rows fp32 tiles      (sum_of_squares / eltwise_chain)
//   B collapse_block   within-tile row sum -> rows column-0-valid tiles            (reduce, ReduceTile)
//   C combine_block    root (or Cw==1): sum of the num_partials slots -> *1/W, +eps, rsqrt (reduce,
//   AccumulateViaAdd+Skip) D normalize_block  x (.) bcast_col(rstd) -> cb_normed | cb_output_tiles        (mul,
//   BroadcastDim::Col) E scale_block      normed (.) bcast_row(gamma_slice) -> cb_output_tiles        (mul,
//   BroadcastDim::Row) [untilize_x_block] RM output: rows*Wc tiles -> sticks                          (untilize helper)
//
// R3 (WIDTH_SHARDED): x and out are the resident shard buffers. The chains address them with
// TileOffset::Set (base = block_idx * block_rows * core_w_tiles) under caller-managed (None, None)
// CB policies: the shard is published once by the reader and never popped; the output shard is
// reserved once up front and pushed once at the end.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

using namespace compute_kernel_lib;

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

    // ---- load_gamma_slice (RM gamma): one 32-stick block -> Wc gamma tiles, resident for the kernel ----
    if constexpr (gamma_mode == 2) {
        tilize<core_w_tiles, cb_gamma_sticks, cb_gamma_tiles>(1);
    }

    // ---- publish_x_shard (R3): the whole resident shard is the block source; reserve the whole output shard ----
    if constexpr (sharded) {
        cb_wait_front(cb_x_tiles, tensor_row_tiles * core_w_tiles);
        cb_reserve_back(cb_output_tiles, tensor_row_tiles * core_w_tiles);
    }

    // combine_block post-op: DEST holds the raw cross-core SUM (Skip: no SFPU armed) -> mean -> +eps -> rsqrt.
    const auto finalize_rstd = [inv_w_bits, eps_bits](uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, inv_w_bits);
        add_unary_tile(dst, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst);
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows = (block_idx + 1 < num_blocks_this_core) ? block_rows : last_block_rows;
        const auto shape = IterationShape::grid(rows, core_w_tiles);
        [[maybe_unused]] const uint32_t shard_base = block_idx * block_rows * core_w_tiles;

        if constexpr (input_rm) {
            tilize<core_w_tiles, cb_x_sticks, cb_x_tiles>(rows);
        }

        // A: sumsq_block — x*x accumulated in DEST per tile-row; x stays resident for D.
        if constexpr (sharded) {
            eltwise_chain(
                shape,
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Set),
                    input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Set),
                    Dst::D0,
                    DestAccumulation::PerRow>{shard_base, shard_base},
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

        // B: collapse_block — within-tile row sum -> column-0-valid partial per tile-row.
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_sumsq_partial,
            cb_scaler,
            cb_collapse_out,
            ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(rows, 1));

        // C: combine_block — root (or Cw == 1): sum the Cw already-collapsed slots, then finalize.
        if (num_w_splits == 1 || is_root) {
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
        if constexpr (sharded) {
            if constexpr (has_gamma) {
                eltwise_chain(
                    shape,
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Set),
                        input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col)>{
                        shard_base},
                    PackTile<output(cb_normed)>{});
            } else {
                eltwise_chain(
                    shape,
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_x_tiles, WaitPolicy::None, PopPolicy::None, OperandKind::Block, TileOffset::Set),
                        input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col)>{
                        shard_base},
                    PackTile<output(cb_output_tiles, ReservePolicy::None, PushPolicy::None, TileOffset::Set)>{
                        shard_base});
            }
        } else {
            mul<input(cb_x_tiles, WaitPolicy::None, PopPolicy::AtEnd, OperandKind::Block),
                input(cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, OperandKind::Col),
                output(cb_norm_out)>(shape);
        }

        // E: scale_block — normed (.) bcast_row(gamma slice); gamma stays resident.
        if constexpr (has_gamma) {
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
                    PackTile<output(cb_output_tiles, ReservePolicy::None, PushPolicy::None, TileOffset::Set)>{
                        shard_base});
            } else {
                mul<input(cb_normed),
                    input(cb_gamma_tiles, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, OperandKind::Row),
                    output(cb_output_tiles)>(shape);
            }
        }

        if constexpr (input_rm) {
            untilize<core_w_tiles, cb_output_tiles, cb_out_sticks>(rows);
        }
    }

    if constexpr (sharded) {
        cb_push_back(cb_output_tiles, tensor_row_tiles * core_w_tiles);
    }
}
