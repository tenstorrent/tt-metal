// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel (full)
// Runs on math RISC-V core (TRISC), performs tile operations.
//
// Compile-time args:
//   [0] num_rows   - total tile-rows to process (N_outer * Ht)
//   [1] Wt         - tiles per row (W / 32)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB indices
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_gamma_rm = 1;
constexpr uint32_t cb_beta_rm = 2;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_eps = 9;
constexpr uint32_t cb_out_rm = 16;
constexpr uint32_t cb_input_tiled = 24;
constexpr uint32_t cb_mean = 25;
constexpr uint32_t cb_centered = 26;
constexpr uint32_t cb_var_sq = 27;
constexpr uint32_t cb_inv_std = 28;
constexpr uint32_t cb_gamma_tiled = 29;
constexpr uint32_t cb_beta_tiled = 30;
constexpr uint32_t cb_normed = 31;

void kernel_main() {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // Hardware init: srcA=cb_input_rm, srcB=cb_scaler, output=cb_out_rm
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_out_rm);

    // One-time setup: Tilize gamma (c_1 -> c_29) and beta (c_2 -> c_30)
    compute_kernel_lib::
        tilize<cb_gamma_rm, cb_gamma_tiled, compute_kernel_lib::tilize_config::InitUninitMode::InitOnly>(Wt, 1);
    compute_kernel_lib::
        tilize<cb_beta_rm, cb_beta_tiled, compute_kernel_lib::tilize_config::InitUninitMode::UninitOnly>(Wt, 1);

    // Wait for persistent CBs (loaded once, never popped)
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_gamma_tiled, Wt);
    cb_wait_front(cb_beta_tiled, Wt);

    // Per-row loop
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Phase 0: Tilize input from cb_input_rm to cb_input_tiled
        compute_kernel_lib::tilize<cb_input_rm, cb_input_tiled>(Wt, 1);

        // Phase 1: Reduce mean across W dimension
        // SUM reduction with 1/W scaler gives mean
        // WaitUpfrontNoPop: c_24 tiles persist for Phase 2
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_input_tiled, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 2: Subtract mean from input (broadcast COL)
        // c_24 already waited from Phase 1 (NoWaitNoPop)
        // c_25 waited and popped at end (WaitUpfrontPopAtEnd)
        // Output to c_26 (Bulk push)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_input_tiled, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        // Manual pop of c_24 after sub (it was kept via NoPop)
        cb_pop_front(cb_input_tiled, Wt);

        // Phase 3: Square centered values
        // c_26 WaitUpfrontNoPop: persists for Phase 5 (mul inv_std)
        // Output to c_27 (Bulk push)
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_centered, cb_var_sq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 4a: Reduce variance (SUM of squares with 1/W scaler) to c_inv_std
        // c_27 BulkWaitBulkPop: wait all Wt tiles, process, pop all
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_var_sq, cb_scaler, cb_inv_std, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 4b: Add epsilon (SCALAR broadcast) and apply rsqrt
        // c_28 (inv_std) has variance, c_9 has epsilon
        // Output to c_mean (c_25, reused as temp) with rsqrt post-op
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_inv_std,
            cb_eps,
            cb_mean,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 5: Multiply centered values by inv_std (broadcast COL)
        // c_26 already waited from Phase 3 (NoWaitNoPop), manual pop after
        // c_25 (cb_mean, now has inv_std) waited and popped at end
        // Output to c_31 (Bulk push)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_centered, cb_mean, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        // Manual pop c_26 after mul (it was kept via NoPop from Phase 3)
        cb_pop_front(cb_centered, Wt);

        // Phase 6: Multiply by gamma (broadcast ROW)
        // c_31 WaitUpfrontPopAtEnd: consumed
        // c_29 (gamma) NoWaitNoPop: persists (program lifetime)
        // Output to c_26 (reused, Bulk push)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_normed, cb_gamma_tiled, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 7: Add beta (broadcast ROW)
        // c_26 WaitUpfrontPopAtEnd: consumed
        // c_30 (beta) NoWaitNoPop: persists (program lifetime)
        // Output to c_24 (reused, Bulk push)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_centered, cb_beta_tiled, cb_input_tiled, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 8: Untilize final output to cb_out_rm
        compute_kernel_lib::untilize<Wt, cb_input_tiled, cb_out_rm>(1);
    }
}
