// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for rms_norm.
//
//   RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + epsilon) * gamma
//
// Parameterized row-parallel streaming reduction. Per assigned tile-row:
//   Pass 1 (stream W in W_BLOCK_TILES chunks):
//     (rm) tilize -> square -> accumulate_reduce_block<SUM,ROW>  =>  Sum(x^2)/W
//   Finalize (1 tile, in place):
//     transform_in_place: add epsilon (SFPU) then rsqrt (SFPU)    =>  1/rms
//   Pass 2 (stream W again, re-reading x):
//     (rm) tilize -> [ (gamma) tilize gamma ] -> mul<Col> x*(1/rms)
//       -> [ mul<Row> *gamma ] -> [ (rm) untilize ]
//
// The Sum(x^2) accumulator (cb_sumsq) is a single-thread compute scratch: the
// reduce writes it, transform_in_place rewrites it in place, the pass-2 mul
// reads it held (never popped mid-row), and it is popped once at row end.
//
// Every phase is a kernel_lib helper. The only raw LLK is the epsilon+rsqrt
// finalizer inside transform_in_place's documented lambda hook (its intended
// use: "a chain like mul_unary_tile, add_unary_tile, rsqrt_tile").
//
// HELPER NOTE: the streaming-reduce wrapper accumulate_reduce_block() is stale
// against the current reduce() (it forwards CB ids as RUNTIME args, but reduce()
// now takes them as TEMPLATE args, per reduce_helpers_compute.hpp:482 and its
// examples at :43-51) — a fresh compile of the wrapper fails. So the per-block
// accumulating reduce is expressed directly on the lower-level reduce() helper
// (same behaviour the wrapper documents: Accumulate::at(cb, b) each block, and
// the partial scaler routed only to the last block). transform_in_place, which
// does not touch reduce(), is used as-is.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_gamma_tiles = 4;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t cb_xsq = 24;
constexpr uint32_t cb_sumsq = 25;
constexpr uint32_t cb_norm = 26;
constexpr uint32_t TILE_H = 32;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(1);
    constexpr uint32_t HAS_PARTIAL_W = get_compile_time_arg_val(2);
    constexpr uint32_t origin_H = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_image = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t W_BLOCK_TILES = get_compile_time_arg_val(6);
    constexpr uint32_t num_w_blocks = get_compile_time_arg_val(7);
    (void)Wt;

    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    uint32_t eps_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::row(W_BLOCK_TILES);
    constexpr auto wshape = ckl::EltwiseShape::tiles(W_BLOCK_TILES);

    for (uint32_t t = 0; t < num_tile_rows; ++t) {
        // Every tile-row is processed as a full 32-row tile in both regimes
        // (the RM reader zero-pads H-padding rows; TILE gets ttnn's zero
        // padding). Padding rows reduce to 0 and are dropped by the writer.

        // ---------- Pass 1: mean(x^2) -> cb_sumsq ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            if constexpr (IS_ROW_MAJOR) {
                ckl::tilize<W_BLOCK_TILES, cb_input_rm, cb_input_tiles>(1, TILE_H);
            }
            ckl::square<cb_input_tiles, cb_xsq>(wshape);
            // Accumulating SUM reduce over W: fresh at b==0, reload+add after; the partial
            // scaler zeros the non-tile-aligned W tail only on the last block's last tile.
            const bool is_last = (b + 1 == num_w_blocks);
            ckl::ReducePartialScaler part = (HAS_PARTIAL_W && is_last) ? ckl::ReducePartialScaler::last_tile_at(1)
                                                                       : ckl::ReducePartialScaler::none();
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_sumsq>(
                reduce_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_sumsq, b),
                ckl::NoOp{},
                part);
        }

        // ---------- Finalize: 1/sqrt(mean + eps), in place ----------
        ckl::transform_in_place(cb_sumsq, [eps_bits](uint32_t dst) {
            binop_with_scalar_tile_init();
            add_unary_tile(dst, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(dst);
        });

        // ---------- Pass 2: x * (1/rms) * gamma ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            if constexpr (IS_ROW_MAJOR) {
                ckl::tilize<W_BLOCK_TILES, cb_input_rm, cb_input_tiles>(1, TILE_H);
            }
            if constexpr (HAS_GAMMA) {
                // gamma is one ROW_MAJOR stick -> a row-0-valid tile (asymmetric tilize, 1 input page).
                ckl::tilize<W_BLOCK_TILES, cb_gamma_rm, cb_gamma_tiles>(1, 1);
                // x * (1/rms): B (cb_sumsq) is a per-row scalar valid in col 0, held across the whole
                // row (HeldBulk, never popped here), broadcast across columns (Col).
                ckl::mul<
                    cb_input_tiles,
                    cb_sumsq,
                    cb_norm,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Scalar>(wshape);
                // * gamma: gamma weight is valid in row 0, broadcast down all rows (Row).
                ckl::mul<cb_norm, cb_gamma_tiles, cb_output_tiles, ckl::BroadcastDim::Row>(wshape);
            } else {
                ckl::mul<
                    cb_input_tiles,
                    cb_sumsq,
                    cb_output_tiles,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Scalar>(wshape);
            }
            if constexpr (IS_ROW_MAJOR) {
                ckl::untilize<W_BLOCK_TILES, cb_output_tiles, cb_output_rm>(1);
            }
        }

        // Release the per-row 1/rms held across pass 2.
        cb_pop_front(cb_sumsq, 1);
    }

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
