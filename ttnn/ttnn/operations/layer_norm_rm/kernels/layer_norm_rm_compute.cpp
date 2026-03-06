// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_reduce_scaler = 8;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_tilized = 24;
constexpr uint32_t cb_mean = 25;
constexpr uint32_t cb_centered = 26;
constexpr uint32_t cb_normed = 30;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    const uint32_t N = get_arg_val<uint32_t>(0);

    if (N == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_in, cb_reduce_scaler, cb_out);

    // Wait for constant CBs
    cb_wait_front(cb_reduce_scaler, 1);

    for (uint32_t tr = 0; tr < N; tr++) {
        // Phase 1: Tilize input (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<cb_in, cb_tilized>(Wt, 1);

        // Phase 2: Reduce mean - SUM with 1/W scaler (WaitUpfrontNoPop so cb_tilized persists)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: sub<COL> cb_tilized - cb_mean -> cb_centered
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Untilize centered to output
        compute_kernel_lib::untilize<Wt, cb_centered, cb_out>(1);
    }
}
