// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Runs on math RISC-V core, performs FPU/SFPU operations
//
// Stage 1: Identity passthrough via tilize + untilize.
// Later stages will add compute phases between tilize and untilize.

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

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);

    // Runtime args
    uint32_t N = get_arg_val<uint32_t>(0);

    // Hardware startup - sets srcA=cb_in, srcB=cb_reduce_scaler, output=cb_out
    compute_kernel_hw_startup(cb_in, cb_reduce_scaler, cb_out);

    for (uint32_t row = 0; row < N; row++) {
        // Phase 1: Tilize input (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<cb_in, cb_tilized>(Wt, 1);

        // Stage 1: skip all compute phases, go directly to untilize

        // Phase 9: Untilize (cb_tilized -> cb_out)
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_out>(1);
    }
}
