// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 1: Identity passthrough - tilize c_0 -> c_1, then untilize c_1 -> c_17.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB indices
constexpr uint32_t cb_input_rm = 0;       // c_0: RM sticks from reader
constexpr uint32_t cb_tilized = 1;        // c_1: Tilized tiles
constexpr uint32_t cb_reduce_scaler = 2;  // c_2: Reduce scaler
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
        // Phase 1: Tilize c_0 -> c_1
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // For Stage 1: pass tilized data directly to output (c_1 -> c_16)
        // We need to copy tiles from c_1 to c_16. But since we just need identity,
        // we can untilize directly from c_1 to c_17.
        compute_kernel_lib::untilize<
            Wt,
            cb_tilized,
            cb_output_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
