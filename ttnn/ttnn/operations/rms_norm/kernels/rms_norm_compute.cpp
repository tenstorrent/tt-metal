// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm compute. Every phase is a kernel_lib helper call over one block (block_rows x core_w_tiles):
//
//   tilize_gamma         (RM γ, once)      tilize          cb_gamma_rm -> cb_gamma
//   tilize_x_block       (RM)              tilize          cb_x_rm     -> cb_x
//   sumsq_block                            eltwise_chain   cb_x (cumulative wait, NOT popped) -> cb_sumsq_raw
//                                          (x*x folded into DEST per tile-row: DestAccumulation::PerRow)
//   collapse_block                         reduce SUM ROW  cb_sumsq_raw -> cb_rinv_col (LOCAL) | cb_partial (W_SPLIT)
//                                          (AccumulateViaAdd + within-tile Collapse)
//   combine_block        (W_SPLIT root)    reduce SUM ROW  cb_gathered -> cb_rinv_col (AccumulateViaAdd, Skip)
//   finalize_rinv_block  (LOCAL / root)    eltwise_chain   cb_rinv_col -> bcast col, *(1/W), +eps, rsqrt
//                                                          -> cb_rinv (LOCAL) | cb_rinv_bcast (root)
//   normalize_block                        eltwise_chain   y = (x * γ[row bcast]) * rinv  -> cb_y (TILE)
//                                                          | in place into cb_x (RM, per-tile credit)
//   untilize_y_block     (RM)              untilize        cb_x (y) -> cb_out_rm
//
// AccumulateViaAdd reduces never read their scaler CB on the tile-aligned path (their inputs are already whole
// summed tiles); the template still wants a slot id distinct from in/out, so cb_x is passed and never pushed as
// a scaler.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

constexpr uint32_t cb_x_rm = 0;
constexpr uint32_t cb_x = 1;
constexpr uint32_t cb_gamma_rm = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_sumsq_raw = 4;
constexpr uint32_t cb_partial = 5;
constexpr uint32_t cb_gathered = 6;
constexpr uint32_t cb_rinv_col = 7;
constexpr uint32_t cb_rinv_bcast = 8;
constexpr uint32_t cb_rinv = 9;
constexpr uint32_t cb_y = 10;
constexpr uint32_t cb_out_rm = 11;

using namespace compute_kernel_lib;

constexpr bool is_rm = get_compile_time_arg_val(0) != 0;
constexpr bool is_wsplit = get_compile_time_arg_val(1) != 0;
constexpr bool has_gamma = get_compile_time_arg_val(2) != 0;
constexpr bool gamma_is_rm = get_compile_time_arg_val(3) != 0;
constexpr uint32_t block_rows = get_compile_time_arg_val(4);
constexpr uint32_t core_w_tiles = get_compile_time_arg_val(5);
constexpr uint32_t w_split = get_compile_time_arg_val(6);
constexpr bool rsqrt_approx = get_compile_time_arg_val(7) != 0;
constexpr Approx kRsqrtApprox = rsqrt_approx ? Approx::Fast : Approx::Exact;

// Normalize output: TILE -> separate cb_y window; RM -> in place into cb_x (compute is its only producer/consumer).
constexpr uint32_t cb_norm_out = is_rm ? cb_x : cb_y;

template <uint32_t out_cb>
ALWI void collapse_block(uint32_t rows) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_sumsq_raw,
        cb_x /* scaler slot: never waited or read on the aligned AccumulateViaAdd path */,
        out_cb,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Collapse>(ReduceInputBlockShape::of(rows, 1));
}

ALWI void combine_block(uint32_t rows) {
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_gathered,
        cb_x /* scaler slot: unused */,
        cb_rinv_col,
        ReduceInputPolicy::BulkWaitBulkPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::AccumulateViaAdd,
        ReduceWithinTile::Skip>(ReduceInputBlockShape::of(rows, w_split));
}

template <uint32_t out_cb>
ALWI void finalize_rinv_block(uint32_t rows, uint32_t inv_w_bits, uint32_t eps_bits) {
    // rinv = rsqrt(Σx² * (1/W) + eps), broadcast from column 0 to the whole tile.
    eltwise_chain(
        IterationShape::tiles(rows),
        UnaryBcast<BroadcastDim::Col, input(cb_rinv_col)>{},
        MulUnary<>{inv_w_bits},
        AddUnary<>{eps_bits},
        Rsqrt<kRsqrtApprox>{},
        PackTile<output(out_cb)>{});
}

ALWI void normalize_block(uint32_t rows) {
    // TILE: cb_x was waited by sumsq; pop the whole block at the end (frees the slot for block b+2).
    // RM:   in place on cb_x with real per-tile credit (pop 1 -> reserve 1 into the slot just freed).
    constexpr InputSpec x_in = is_rm ? input(cb_x, WaitPolicy::PerTile, PopPolicy::PerTile)
                                     : input(cb_x, WaitPolicy::None, PopPolicy::AtEnd, InputTileMapping::Block);
    if constexpr (has_gamma) {
        eltwise_chain(
            IterationShape::grid(rows, core_w_tiles),
            BinaryFpu<
                BinaryFpuOp::Mul,
                x_in,
                input(cb_gamma, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row)>{},
            DestReuseBinary<
                BinaryFpuOp::Mul,
                input(cb_rinv, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                DestReuseType::DEST_TO_SRCA>{},
            PackTile<output(cb_norm_out)>{});
    } else {
        eltwise_chain(
            IterationShape::grid(rows, core_w_tiles),
            BinaryFpu<
                BinaryFpuOp::Mul,
                x_in,
                input(cb_rinv, BroadcastDim::None, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)>{},
            PackTile<output(cb_norm_out)>{});
    }
}

void kernel_main() {
    const uint32_t line_rows = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);

    if constexpr (is_rm) {
        compute_kernel_hw_startup(cb_x_rm, cb_x);
    } else {
        compute_kernel_hw_startup(cb_x, cb_x, cb_sumsq_raw);
    }

    const uint32_t num_blocks = (line_rows + block_rows - 1) / block_rows;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t row0 = b * block_rows;
        const uint32_t rows = (line_rows - row0 < block_rows) ? (line_rows - row0) : block_rows;

        if constexpr (is_rm) {
            tilize<core_w_tiles, cb_x_rm, cb_x>(rows);  // tilize_x_block
        }

        // sumsq_block: raw Σ_W x² per tile-row (32x32 lanes), x stays resident in cb_x.
        eltwise_chain(
            IterationShape::grid(rows, core_w_tiles),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(cb_x, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block),
                input(cb_x, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block),
                Dst::D0,
                DestAccumulation::PerRow>{},
            PackTile<output(
                cb_sumsq_raw,
                ReservePolicy::PerOuter,
                PushPolicy::PerOuter,
                DataFormatReconfig::Enabled,
                TileAddressing::Direct,
                DestAccumulation::PerRow)>{});

        if constexpr (is_wsplit) {
            collapse_block<cb_partial>(rows);
            if (is_root) {
                combine_block(rows);
                finalize_rinv_block<cb_rinv_bcast>(rows, inv_w_bits, eps_bits);
            }
        } else {
            collapse_block<cb_rinv_col>(rows);
            finalize_rinv_block<cb_rinv>(rows, inv_w_bits, eps_bits);
        }

        if constexpr (has_gamma) {
            if (b == 0) {
                if constexpr (gamma_is_rm) {
                    tilize<core_w_tiles, cb_gamma_rm, cb_gamma>(1, 1);  // tilize_gamma (1 valid row page)
                }
                cb_wait_front(cb_gamma, core_w_tiles);  // resident for the whole kernel
            }
        }

        normalize_block(rows);

        if constexpr (is_rm) {
            untilize<core_w_tiles, cb_x, cb_out_rm>(rows);  // untilize_y_block
        }
    }

    if constexpr (has_gamma) {
        if (num_blocks > 0) {
            cb_pop_front(cb_gamma, core_w_tiles);
        }
    }
}
