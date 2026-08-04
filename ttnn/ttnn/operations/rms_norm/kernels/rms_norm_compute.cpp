// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for rms_norm (UNPACK / MATH / PACK).
//
//   out = x * rsqrt( (1/W) * sum_w x^2 + eps ) * gamma
//
// One loop nest covers both regimes (op_design.md section 7).  Per row-block:
//
//   pass A   [RM] tilize            cb_input_sticks -> cb_input_tiles
//            square                 cb_input_tiles  -> cb_x_squared
//            accumulate_reduce_block cb_x_squared   -> cb_row_stat   (sum x^2)
//   finalize transform_in_place     cb_row_stat     -> cb_row_stat   (1/rms)
//   pass B   mul<Col>               cb_input_tiles, cb_row_stat -> NORM_OUT
//            mul<Row>               cb_normalized, cb_gamma_tiles -> cb_output_tiles
//            [RM] untilize          cb_output_tiles -> cb_output_sticks
//
// RESIDENT (NUM_W_CHUNKS == 1): cb_input_tiles is HELD across pass A and pass B
// (pass A pops nothing), so x is read from DRAM once.  STREAM: each pass pops
// its chunk and the reader re-reads x for pass B.
//
// Every phase is a kernel_lib helper.  The only raw LLK is inside the
// transform_in_place lambda (x1/W, +eps, rsqrt) — that helper's documented
// calling convention, and the family explicitly routes multi-instruction
// finalizers like rsqrt-with-eps here rather than to a chain
// (streaming_reduce_helpers.hpp:75-78).
//
// Explicit cb_pop_front calls on cb_row_stat / cb_gamma_tiles / cb_scaler are
// the sanctioned pattern for operands whose lifetime spans more calls than any
// single PopPolicy can express (op_design.md section 6.1).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_x_squared = 2;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_row_stat = 4;
constexpr uint32_t cb_gamma_sticks = 5;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_normalized = 7;
constexpr uint32_t cb_output_tiles = 8;
constexpr uint32_t cb_output_sticks = 9;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(5);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(6);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(7);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(8);
    constexpr uint32_t REDUCE_BULK = get_compile_time_arg_val(9);

    const uint32_t num_rows = get_arg_val<uint32_t>(0);  // tile-rows owned by this core

    constexpr bool RM = (IS_TILE == 0);
    constexpr bool HAS_G = (HAS_GAMMA != 0);
    constexpr bool G_RM = (GAMMA_IS_RM != 0);
    // X_RESIDENT == GAMMA_RESIDENT == (NUM_W_CHUNKS == 1): one source of truth.
    constexpr bool X_RESIDENT = (NUM_W_CHUNKS == 1);

    // srcA at boot is whichever CB the first helper unpacks from.
    constexpr uint32_t CB_A = RM ? cb_input_sticks : cb_input_tiles;
    compute_kernel_hw_startup(CB_A, cb_scaler, cb_output_tiles);

    // ---- policy / shape knobs --------------------------------------------
    constexpr auto REDUCE_POLICY =
        (REDUCE_BULK != 0) ? ckl::ReduceInputPolicy::BulkWaitBulkPop : ckl::ReduceInputPolicy::WaitAndPopPerTile;
    // Non-tile-aligned W: the reader emitted [full scaler, partial scaler];
    // route the partial one to the last width tile so pad lanes contribute 0.
    constexpr auto PARTIAL_SCALER =
        (PARTIAL_W != 0) ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();
    // RESIDENT holds x across both passes -> pass A must not pop it.
    constexpr auto PASS_A_POP = X_RESIDENT ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd;
    constexpr uint32_t NORM_OUT = HAS_G ? cb_normalized : cb_output_tiles;

    // ---- gamma: resident for the whole core's assignment (RESIDENT) -------
    if constexpr (HAS_G && X_RESIDENT && G_RM) {
        ckl::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1);
    }

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        // ================= pass A: sum(x^2) over the whole width ===========
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            if constexpr (RM) {
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            ckl::square<
                ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, PASS_A_POP, ckl::OperandKind::Block),
                ckl::output(cb_x_squared)>(ckl::EltwiseShape::grid(rows, WT_CHUNK));

            ckl::accumulate_reduce_block<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_x_squared,
                cb_scaler,
                cb_row_stat,
                REDUCE_POLICY>(ckl::ReduceInputBlockShape::of(rows, WT_CHUNK), c, NUM_W_CHUNKS, PARTIAL_SCALER);
        }

        // ================= finalize: 1/rms = rsqrt(sum/W + eps) ============
        // Pops before reserving, so the `rows`-page accumulator CB suffices.
        for (uint32_t i = 0; i < rows; ++i) {
            ckl::transform_in_place(cb_row_stat, [](uint32_t dst) {
                binop_with_scalar_tile_init();
                mul_unary_tile(dst, INV_W_BITS);  // x (1/W): the LOGICAL width (R1)
                add_unary_tile(dst, EPS_BITS);
                rsqrt_tile_init();
                rsqrt_tile(dst);
            });
        }

        // ================= pass B: scale ===================================
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            if constexpr (RM && !X_RESIDENT) {
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            if constexpr (HAS_G && !X_RESIDENT && G_RM) {
                ckl::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1);
            }

            // x * (1/rms). The stat is a REDUCE_ROW result: column-shaped, so it
            // broadcasts back ACROSS columns (BroadcastDim::Col) and must be
            // operand B. OperandKind::Col indexes it by row only, and it is not
            // popped -- every width chunk of this block re-reads it.
            ckl::mul<
                ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(cb_row_stat, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col),
                ckl::output(NORM_OUT),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(rows, WT_CHUNK));

            if constexpr (HAS_G) {
                // gamma is row-shaped (1 x W, valid in row 0) -> broadcasts DOWN
                // rows (BroadcastDim::Row), indexed by column (OperandKind::Row).
                ckl::mul<
                    ckl::input(cb_normalized, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::input(cb_gamma_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row),
                    ckl::output(cb_output_tiles),
                    ckl::BroadcastDim::Row>(ckl::EltwiseShape::grid(rows, WT_CHUNK));
            }

            if constexpr (RM) {
                ckl::untilize<WT_CHUNK, cb_output_tiles, cb_output_sticks>(rows);
            }

            if constexpr (HAS_G && !X_RESIDENT) {
                cb_pop_front(cb_gamma_tiles, WT_CHUNK);
            }
        }

        cb_pop_front(cb_row_stat, rows);
    }

    cb_pop_front(cb_scaler, (PARTIAL_W != 0) ? 2 : 1);
    if constexpr (HAS_G && X_RESIDENT) {
        cb_pop_front(cb_gamma_tiles, WT_CHUNK);
    }
}
