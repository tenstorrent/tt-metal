// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Phase 1:  Tilize c_0 -> c_24
// Phase 2:  Reduce mean (SUM REDUCE_ROW) c_24 -> c_25
// Phase 3:  Subtract mean (SUB COL) c_24, c_25 -> c_26
// Phase 4:  Square centered c_26 -> c_27
// Phase 5:  Reduce variance (SUM REDUCE_ROW) c_27 -> c_28
// Phase 6:  Add epsilon + rsqrt c_28, c_10 -> c_29
// Phase 7:  Multiply by inverse std c_26, c_29 -> c_30
// Phase 10: Untilize c_30 -> c_16

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t c_0 = 0;    // RM input sticks (from reader)
constexpr uint32_t c_8 = 8;    // Reduce scaler (persistent, 1/W)
constexpr uint32_t c_10 = 10;  // Epsilon (persistent)
constexpr uint32_t c_16 = 16;  // Untilized output (to writer)
constexpr uint32_t c_24 = 24;  // Tilized input
constexpr uint32_t c_25 = 25;  // Mean col vector
constexpr uint32_t c_26 = 26;  // Centered (x - mean)
constexpr uint32_t c_27 = 27;  // Squared centered
constexpr uint32_t c_28 = 28;  // Variance col vector (fp32)
constexpr uint32_t c_29 = 29;  // Inverse std
constexpr uint32_t c_30 = 30;  // Pre-untilize (normalized)

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // Hardware initialization - required before using helpers
    compute_kernel_hw_startup(c_0, c_8, c_16);

    // Wait for persistent scalar CBs once (pushed by reader at program start)
    cb_wait_front(c_8, 1);   // reduce scaler (1/W)
    cb_wait_front(c_10, 1);  // epsilon

    // Per-row loop
    for (uint32_t row = 0; row < num_rows_per_core; row++) {
        // Phase 1: Tilize c_0 -> c_24
        compute_kernel_lib::tilize<
            c_0,
            c_24,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 2: Reduce mean (SUM REDUCE_ROW) c_24 -> c_25
        cb_wait_front(c_24, Wt);
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                c_24, c_8, c_25, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean (SUB COL) c_24, c_25 -> c_26
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            c_24, c_25, c_26, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        cb_pop_front(c_24, Wt);  // free tilized input
        cb_pop_front(c_25, 1);   // free mean

        // Phase 4: Square centered c_26 -> c_27
        cb_wait_front(c_26, Wt);
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            c_26, c_27, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        // c_26 NOT popped -- persists for Phase 7

        // Phase 5: Reduce variance (SUM REDUCE_ROW) c_27 -> c_28
        cb_wait_front(c_27, Wt);
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                c_27, c_8, c_28, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        cb_pop_front(c_27, Wt);  // free squared centered

        // Phase 6: Add epsilon + rsqrt c_28, c_10 -> c_29
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            c_28, c_10, c_29, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply by inverse std c_26, c_29 -> c_30
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            c_26, c_29, c_30, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        cb_pop_front(c_26, Wt);  // free centered
        cb_pop_front(c_29, 1);   // free inv_std

        // Phase 10: Untilize c_30 -> c_16
        compute_kernel_lib::untilize<
            Wt,
            c_30,
            c_16,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
