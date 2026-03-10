// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 2 (subtract_mean): tilize -> reduce_mean -> sub_mean -> untilize

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // CB indices
    constexpr uint32_t cb_input_rm = 0;    // c_0: RM sticks from reader
    constexpr uint32_t cb_scaler = 8;      // c_8: reduce scaler (1/W)
    constexpr uint32_t cb_tilized = 16;    // c_16: tilized data (multi-use)
    constexpr uint32_t cb_output_rm = 17;  // c_17: untilized output for writer
    constexpr uint32_t cb_reduce_out = 24;  // c_24: reduce output (mean)
    constexpr uint32_t cb_centered = 25;    // c_25: centered values

    // Hardware init - must come first
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_tilized);

    for (uint32_t block = 0; block < num_blocks_per_core; block++) {
        // Phase 1: Tilize (c_0 -> c_16)
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Wt, 1);

        // Phase 2: Reduce Mean (c_16 -> c_24)
        // WaitUpfrontNoPop: waits for Wt tiles in c_16, does NOT pop them (needed for Phase 3)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_tilized, cb_scaler, cb_reduce_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract Mean (c_16, c_24 -> c_25)
        // A: c_16 already waited from Phase 2 (NoWaitPopAtEnd - pops at end)
        // B: c_24 freshly pushed (WaitUpfrontPopAtEnd - waits then pops at end)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_tilized, cb_reduce_out, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 10: Untilize (c_25 -> c_17) -- centered values
        compute_kernel_lib::untilize<Wt, cb_centered, cb_output_rm>(1);
    }
}
