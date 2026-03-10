// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 4: tilize -> reduce_mean -> sub_mean -> square -> reduce_var -> add_eps+rsqrt -> mul_rsqrt -> untilize
// Full normalization: (x - mean) / sqrt(var + eps)

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// CB indices
constexpr uint32_t cb_input_rm = 0;      // c_0: RM sticks for tilize
constexpr uint32_t cb_tilized = 1;       // c_1: Tilized input tiles
constexpr uint32_t cb_scaler = 8;        // c_8: Reduce scaler (1/W)
constexpr uint32_t cb_eps = 9;           // c_9: Epsilon tile
constexpr uint32_t cb_output_rm = 16;    // c_16: Untilized RM output
constexpr uint32_t cb_mean = 24;         // c_24: Row mean
constexpr uint32_t cb_centered = 25;     // c_25: x - mean
constexpr uint32_t cb_centered_sq = 26;  // c_26: (x - mean)^2
constexpr uint32_t cb_var = 27;          // c_27: Row variance
constexpr uint32_t cb_rsqrt = 28;        // c_28: rsqrt(var + eps)
constexpr uint32_t cb_normalized = 31;   // c_31: Normalized output

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

void kernel_main() {
    // Hardware startup: srcA=c_0, srcB=c_8, ocb=c_16
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_output_rm);

    // Runtime args
    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    // Wait for program-lifetime tiles (pushed once by reader)
    cb_wait_front(cb_eps, 1);

    // Main loop: process nblocks tile-rows
    for (uint32_t block = 0; block < nblocks; ++block) {
        // Phase 1: Tilize (RM -> tile) c_0 -> c_1
        compute_kernel_lib::
            tilize<cb_input_rm, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);

        // Phase 2: Reduce mean (row-wise sum * 1/W)
        // c_1: Wt tiles, WaitUpfrontNoPop (persists for Phase 3)
        // c_8: 1/W scaler, waited by helper, not popped (program lifetime)
        // c_24: 1 tile mean, pushed
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean (broadcast COL)
        // c_1: Wt tiles, NoWaitNoPop (already waited in Phase 2)
        // c_24: 1 tile mean, WaitUpfrontPopAtEnd (consumed)
        // c_25: Wt tiles centered, pushed
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        // Manual pop c_1 -- NoWaitNoPop leaves c_1 tiles
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square centered values
        // c_25: Wt tiles, WaitUpfrontNoPop (persists for Phase 7)
        // c_26: Wt tiles centered^2, pushed
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_centered_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce variance (row-wise sum * 1/W)
        // c_26: Wt tiles, default WaitAndPopPerTile (consumed)
        // c_8: 1/W scaler
        // c_27: 1 tile variance, pushed
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_centered_sq, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Add epsilon + rsqrt
        // c_27: 1 tile variance, WaitAndPopPerTile (consumed)
        // c_9: 1 tile epsilon, NoWaitNoPop (program lifetime, manually managed)
        // c_28: 1 tile rsqrt(var+eps), pushed
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_var, cb_eps, cb_rsqrt, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply by rsqrt (broadcast COL)
        // c_25: Wt tiles centered, NoWaitPopAtEnd (already waited in Phase 4, pop at end)
        // c_28: 1 tile rsqrt, WaitAndPopPerTile (COL broadcast, consumed)
        // c_31: Wt tiles normalized, pushed
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_rsqrt, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 10: Untilize (tile -> RM) c_31 -> c_16
        compute_kernel_lib::untilize<
            Wt,
            cb_normalized,
            cb_output_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(1);
    }

    // Pop program-lifetime tiles
    cb_pop_front(cb_eps, 1);
}
