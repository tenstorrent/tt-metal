// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 4: 2-pass: pass1=tilize+reduce_mean, pass2=tilize+sub_mean+square+reduce_var+add_eps+rsqrt

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB indices
constexpr uint32_t cb_input_rm = 0;       // c_0: RM sticks from reader
constexpr uint32_t cb_tilized = 1;        // c_1: Tilized tiles
constexpr uint32_t cb_reduce_scaler = 2;  // c_2: Reduce scaler (1/W)
constexpr uint32_t cb_eps_scalar = 3;     // c_3: Epsilon tile
constexpr uint32_t cb_mean = 24;          // c_24: Row-reduced mean (1 tile)
constexpr uint32_t cb_centered = 25;      // c_25: Centered tiles (x - mean)
constexpr uint32_t cb_var = 26;           // c_26: Row-reduced variance (1 tile)
constexpr uint32_t cb_rsqrt_var = 27;     // c_27: rsqrt(var+eps) (1 tile)
constexpr uint32_t cb_output_tiles = 16;  // c_16: Pre-untilize output tiles
constexpr uint32_t cb_output_rm = 17;     // c_17: Untilized RM output

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

void kernel_main() {
    // Runtime args
    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    if (nblocks == 0) {
        return;
    }

    // Hardware startup: input CB for srcA/srcB, output CB for pack
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm);

    for (uint32_t block = 0; block < nblocks; block++) {
        // ==================== PASS 1: Compute mean ====================

        // Phase 1: Tilize c_0 -> c_1
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 2: Reduce mean - SUM along row with 1/W scaler
        // c_1 consumed per-tile, c_24 gets 1 tile (col vector, persists for pass 2)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ==================== PASS 2: Variance + rsqrt ====================

        // Phase 3: Tilize (re-read from reader's second pass)
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 4: sub(x, mean) with COL broadcast -> c_25 (centered)
        // A=c_1 (tilized, consumed per-tile), B=c_24 (mean, NoWaitNoPop - persists)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Pop mean after use (no longer needed for this tile-row)
        cb_pop_front(cb_mean, 1);

        // Phase 5: Square centered -> c_1 (reuse, c_1 is free after phase 4)
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 6: Reduce variance - SUM along row with 1/W scaler
        // c_1 consumed per-tile, c_26 gets 1 tile (variance col vector)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_tilized, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 7: add(var, eps) with SCALAR broadcast + rsqrt post-op -> c_27
        // A=c_26 (variance, consumed), B=c_3 (epsilon, WaitUpfrontNoPop - persists across blocks)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var,
            cb_eps_scalar,
            cb_rsqrt_var,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Untilize rsqrt_var (1 tile) -> c_17 for output
        // For this stage, output is the rsqrt(var+eps) column vector
        compute_kernel_lib::untilize<
            1,
            cb_rsqrt_var,
            cb_output_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
