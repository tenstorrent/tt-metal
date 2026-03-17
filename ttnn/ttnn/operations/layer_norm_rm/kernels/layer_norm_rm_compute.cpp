// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 1 (data_pipeline): tilize input, copy through to output, untilize.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace NAMESPACE {

void MAIN {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // ========== Runtime args ==========
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // ========== CB IDs ==========
    constexpr uint32_t cb_input_rm = 0;  // c_0: RM sticks from reader
    constexpr uint32_t cb_tilized = 1;   // c_1: tilized input
    constexpr uint32_t cb_out_rm = 16;   // c_16: untilized output for writer

    // ========== Hardware init ==========
    // Three-arg form: srcA=cb_input_rm, srcB=cb_input_rm (not used for tilize), ocb=cb_out_rm
    compute_kernel_hw_startup(cb_input_rm, cb_input_rm, cb_out_rm);

    // ========== Main loop ==========
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Phase 1: Tilize c_0 -> c_1 (Wt tiles)
        compute_kernel_lib::tilize<Wt, cb_input_rm, cb_tilized>(1);

        // Phase 10: Untilize c_1 -> c_16 (identity passthrough for stage 1)
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_out_rm>(1);
    }
}

}  // namespace NAMESPACE
