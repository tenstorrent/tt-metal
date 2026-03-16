// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Runs on math RISC-V core, performs FPU/SFPU operations
//
// Stage 3 (rms_normalize): tilize -> square -> reduce_row -> add_eps+rsqrt -> re-tilize -> mul<COL> -> untilize
//   Full normalization: output = x * rsqrt(mean(x^2) + eps)

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// CB indices
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_in = 1;
constexpr uint32_t cb_x_sq = 2;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_eps = 9;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_rm = 17;
constexpr uint32_t cb_reduce_out = 24;
constexpr uint32_t cb_rms_inv = 25;

// Compile-time args
constexpr uint32_t is_rm_input = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t Ht = get_compile_time_arg_val(3);
constexpr uint32_t NC = get_compile_time_arg_val(4);

void kernel_main() {
    uint32_t num_rows = get_arg_val<uint32_t>(0);

    // Hardware init: srcA=cb_in, srcB=cb_scaler, out=cb_out
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Phase 1: Tilize (RM input only) - pass 1 data
        if constexpr (is_rm_input) {
            compute_kernel_lib::tilize<
                Wt,
                cb_in_rm,
                cb_in,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(1);
        }

        // Phase 2: Square (cb_in -> cb_x_sq)
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_in, cb_x_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 3: Reduce Row (cb_x_sq -> cb_reduce_out)
        // Uses SUM with scaler 1/W to compute mean(x^2)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_x_sq, cb_scaler, cb_reduce_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt, 1));

        // Phase 4: Add epsilon + rsqrt (cb_reduce_out + cb_eps -> cb_rms_inv)
        // cb_eps is persistent (WaitUpfrontNoPop for B) across all rows
        // rsqrt is applied as a post-op callback
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_reduce_out,
            cb_eps,
            cb_rms_inv,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 5: Re-tilize (RM input only, pass 2 data)
        if constexpr (is_rm_input) {
            compute_kernel_lib::tilize<
                Wt,
                cb_in_rm,
                cb_in,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(1);
        }

        // Phase 6: Normalize - broadcast multiply input * rsqrt(mean+eps)
        // cb_in: Wt tiles (pass 2 data), popped per tile
        // cb_rms_inv: 1 tile, waited upfront, NOT popped (persists for all Wt tiles)
        // Output: cb_out (Wt tiles)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_in, cb_rms_inv, cb_out, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Pop cb_rms_inv after normalization (it was WaitUpfrontNoPop)
        cb_pop_front(cb_rms_inv, 1);

        // Phase 8: Untilize (RM output only)
        if constexpr (is_rm_input) {
            compute_kernel_lib::untilize<
                Wt,
                cb_out,
                cb_out_rm,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
        }
    }

    // Pop cb_eps at kernel end (it was WaitUpfrontNoPop across all rows)
    cb_pop_front(cb_eps, 1);
}
