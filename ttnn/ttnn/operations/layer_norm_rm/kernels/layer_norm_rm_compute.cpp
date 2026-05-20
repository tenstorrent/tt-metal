// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for layer_norm_rm.
//
// Per tile-row, three streaming passes:
//   Pass 1: reduce<SUM, REDUCE_ROW>(cb_input_tiles, cb_scaler, cb_mean)
//   Pass 2: sub<COL>(cb_input_tiles, cb_mean, cb_centered)
//           square_in_place(cb_centered)
//           reduce<SUM, REDUCE_ROW>(cb_centered, cb_scaler, cb_inv_std)
//           transform_in_place(cb_inv_std) → rsqrt(var + eps)
//   Pass 3: sub<COL>(cb_input_tiles, cb_mean, cb_centered)
//           mul_in_place<COL>(cb_centered, cb_inv_std)
//           (optional) mul_in_place<ROW>(cb_centered, cb_gamma_tiles)
//           (optional) add_in_place<ROW>(cb_centered, cb_beta_tiles)
//           drain → cb_output  (copy_tiles for TILE / untilize for RM)
//
// For ROW_MAJOR input, each pass first tilizes 32 sticks of cb_input_sticks
// (asymmetric tilize: num_blocks=1, total_input_pages=32) into cb_input_tiles.
//
// Gamma/beta are tilized ONCE at startup from their 1-stick CB into their
// Wt-tile CB; they persist (WaitUpfrontNoPop, never popped) through every
// Pass 3 and are popped at kernel exit.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_gamma_sticks = 2;
constexpr uint32_t cb_beta_sticks = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_output = 16;
constexpr uint32_t cb_gamma_tiles = 24;
constexpr uint32_t cb_beta_tiles = 25;
constexpr uint32_t cb_mean = 26;
constexpr uint32_t cb_inv_std = 27;
constexpr uint32_t cb_centered = 28;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t total_tile_rows = get_compile_time_arg_val(1);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(2) != 0;
    constexpr bool IS_RM_INPUT = get_compile_time_arg_val(3) != 0;
    constexpr bool IS_RM_OUTPUT = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(5) != 0;
    constexpr bool HAS_BETA = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t eps_bits = get_compile_time_arg_val(7);

    // Single full hardware init for the kernel — helpers will reconfig as needed.
    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output);

    // ---------------- Pre-pass: tilize gamma / beta (one tile-row each) ----------------
    if constexpr (HAS_GAMMA) {
        // Asymmetric: 1 input page → Wt output tiles. Only row 0 of each output
        // tile carries data; rows 1-31 are L1 garbage but unused by ROW broadcast.
        ckl::tilize<Wt, cb_gamma_sticks, cb_gamma_tiles>(/*num_blocks=*/1, /*total_input_pages=*/1);
    }
    if constexpr (HAS_BETA) {
        ckl::tilize<Wt, cb_beta_sticks, cb_beta_tiles>(/*num_blocks=*/1, /*total_input_pages=*/1);
    }

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(/*rows=*/1, /*cols=*/Wt, /*batches=*/1);
    constexpr auto bin_shape = ckl::BinaryInputBlockShape::of(/*rows=*/1, /*cols=*/Wt);
    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    // ---------------- Per-tile-row loop ----------------
    for (uint32_t tr = 0; tr < total_tile_rows; ++tr) {
        // ============ Pass 1: mean ============
        if constexpr (IS_RM_INPUT) {
            ckl::tilize<Wt, cb_input_sticks, cb_input_tiles>(/*num_blocks=*/1, /*total_input_pages=*/32);
        }
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            cb_input_tiles,
            cb_scaler,
            cb_mean,
            reduce_block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            ckl::NoOp{},
            partial_scaler);

        // ============ Pass 2: variance → inv_std ============
        if constexpr (IS_RM_INPUT) {
            ckl::tilize<Wt, cb_input_sticks, cb_input_tiles>(/*num_blocks=*/1, /*total_input_pages=*/32);
        }
        // sub<COL>: A = cb_input_tiles (per-tile streaming), B = cb_mean (WaitUpfrontNoPop, 1 tile)
        ckl::sub<
            ckl::BroadcastDim::COL,
            ckl::BinaryInputPolicy::WaitAndPopPerTile,
            ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_input_tiles, cb_mean, cb_centered, bin_shape);

        ckl::square_in_place(cb_centered, bin_shape);

        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            cb_centered,
            cb_scaler,
            cb_inv_std,
            reduce_block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            ckl::NoOp{},
            partial_scaler);

        // variance → 1/sqrt(variance + eps) in place. Lambda runs in DST[0].
        ckl::transform_in_place(cb_inv_std, [](uint32_t dst) {
            binop_with_scalar_tile_init();
            add_unary_tile(dst, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(dst);
        });

        // ============ Pass 3: normalize + (gamma) + (beta) + drain ============
        if constexpr (IS_RM_INPUT) {
            ckl::tilize<Wt, cb_input_sticks, cb_input_tiles>(/*num_blocks=*/1, /*total_input_pages=*/32);
        }
        ckl::sub<
            ckl::BroadcastDim::COL,
            ckl::BinaryInputPolicy::WaitAndPopPerTile,
            ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_input_tiles, cb_mean, cb_centered, bin_shape);

        ckl::mul_in_place<ckl::BroadcastDim::COL, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_inv_std, bin_shape);

        if constexpr (HAS_GAMMA) {
            ckl::mul_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_centered, cb_gamma_tiles, bin_shape);
        }
        if constexpr (HAS_BETA) {
            ckl::add_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_centered, cb_beta_tiles, bin_shape);
        }

        // Drain cb_centered → cb_output.
        if constexpr (IS_RM_OUTPUT) {
            ckl::untilize<Wt, cb_centered, cb_output>(/*num_blocks=*/1);
        } else {
            ckl::copy_tiles<ckl::CopyInputPolicy::WaitAndPop>(cb_centered, cb_output, Wt);
        }

        // Release per-row persistent CBs (cb_mean, cb_inv_std were WaitUpfrontNoPop'd).
        cb_pop_front(cb_mean, 1);
        cb_pop_front(cb_inv_std, 1);
    }

    // ---------------- Teardown ----------------
    if constexpr (HAS_GAMMA) {
        cb_pop_front(cb_gamma_tiles, Wt);
    }
    if constexpr (HAS_BETA) {
        cb_pop_front(cb_beta_tiles, Wt);
    }
    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
