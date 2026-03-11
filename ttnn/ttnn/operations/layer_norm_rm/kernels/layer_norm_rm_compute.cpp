// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 3: tilize, reduce_row(mean), sub(COL), square, reduce_row(var),
//          add_eps+rsqrt, mul_inv_std(COL), untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB indices
constexpr uint32_t cb_rm_input = 0;    // RM input sticks
constexpr uint32_t cb_scaler = 8;      // Reduce scaler (1/W)
constexpr uint32_t cb_eps = 9;         // Epsilon constant tile
constexpr uint32_t cb_rm_output = 16;  // RM output sticks
constexpr uint32_t cb_tilized = 24;    // Tilized input / reused intermediate
constexpr uint32_t cb_mean = 25;       // Row mean / reused for variance (1 tile)
constexpr uint32_t cb_centered = 26;   // x - mean (Wt tiles)
constexpr uint32_t cb_squared = 27;    // (x - mean)^2 (Wt tiles)
constexpr uint32_t cb_inv_std = 28;    // rsqrt(var + eps) (1 tile)

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t max_blocks = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

void kernel_main() {
    // Hardware startup
    compute_kernel_hw_startup(cb_tilized, cb_scaler, cb_rm_output);

    // Get actual num_blocks for this core from runtime args
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Wait for persistent CBs (scaler and eps persist entire program)
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    // Main loop: per block
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize (c_0 -> c_24)
        compute_kernel_lib::
            tilize<cb_rm_input, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);

        // Phase 2: Reduce Row for mean (c_24, c_8 -> c_25)
        // Scaler CB has 1/W; reduce<SUM> computes: sum(x_i * (1/W)) = mean
        // WaitUpfrontNoPop: c_24 tiles persist for Phase 3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 3: Sub mean with COL broadcast (c_24 - c_25 -> c_26)
        // c_24: already waited from Phase 2, NoWaitNoPop -> caller pops
        // c_25: freshly pushed by Phase 2, WaitUpfrontPopAtEnd -> helper waits+pops
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_tilized, Wt);  // Manual pop for NoWaitNoPop A

        // Phase 4: Square centered (c_26 -> c_27)
        // WaitUpfrontNoPop: c_26 tiles persist for Phase 7 (multiply by inv_std)
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_squared, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce Row for variance (c_27, c_8 -> c_25)
        // Default WaitAndPopPerTile: c_27 tiles consumed
        // c_25 reused (was freed in Phase 3)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_squared, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 6: Add epsilon + rsqrt (c_25 + c_9 -> c_28)
        // c_25 (variance): freshly pushed by Phase 5, WaitAndPopPerTile
        // c_9 (eps): persistent, NoWaitNoPop (already waited at startup)
        // Post-op: rsqrt applied in DEST before pack -> c_28 = rsqrt(var + eps)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_mean, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::of(1, 1), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply centered by inv_std with COL broadcast (c_26 * c_28 -> c_24)
        // c_26: already waited from Phase 4, NoWaitNoPop -> caller pops
        // c_28: freshly pushed by Phase 6, WaitUpfrontPopAtEnd -> helper waits+pops
        // Output to c_24 (reused, was freed in Phase 3)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_centered, cb_inv_std, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_centered, Wt);  // Manual pop for NoWaitNoPop A

        // Phase 10: Untilize normalized output (c_24 -> c_16)
        compute_kernel_lib::
            untilize<Wt, cb_tilized, cb_rm_output, compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(
                1);
    }

    // Pop persistent CBs
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
}
