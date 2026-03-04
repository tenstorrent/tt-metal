// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel
// Stage 1: tilize + untilize passthrough (identity)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_tilized = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_17;

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t nblocks_per_core = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

void kernel_main() {
    // Hardware init for tilize: srcA=srcB=cb_in, output=cb_tilized
    compute_kernel_hw_startup(cb_in, cb_tilized);

    for (uint32_t block = 0; block < nblocks_per_core; ++block) {
        // Phase 1: Tilize (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<cb_in, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(
            Wt, 1);

        // Phase 10: Untilize (cb_tilized -> cb_out)
        compute_kernel_lib::
            untilize<Wt, cb_tilized, cb_out, compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(1);
    }
}
