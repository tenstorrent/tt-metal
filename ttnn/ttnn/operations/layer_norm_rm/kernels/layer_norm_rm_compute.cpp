// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Compute Kernel
//
// Stage 1 (data_pipeline): tilize(cb_in_rm -> cb_tilized) + untilize(cb_tilized -> cb_out_rm)
//
// Compile-time args:
//   [0] Wt               — tiles per row (W / 32)
//   [1] nblocks_per_core — tile-rows this core processes
//   [2] has_gamma        — 1 if gamma present, 0 otherwise
//   [3] has_beta         — 1 if beta present, 0 otherwise
//   [4] epsilon_packed   — epsilon as uint32 IEEE-754 bits

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 9;
constexpr uint32_t cb_out_rm = 17;
}  // namespace

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    // CT arg [1] is nblocks_per_core (legacy, now read from runtime args)
    // has_gamma, has_beta, epsilon_packed are used in later stages

    // ========== Runtime args ==========
    const uint32_t nblocks_per_core = get_arg_val<uint32_t>(0);

    // ========== Hardware startup ==========
    // 3-arg form: srcA=cb_in_rm, srcB=cb_scaler, ocb=cb_out_rm
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm);

    for (uint32_t block = 0; block < nblocks_per_core; ++block) {
        // Phase 1: Tilize (cb_in_rm -> cb_tilized)
        compute_kernel_lib::tilize<
            cb_in_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Stage 1: Direct passthrough — untilize from cb_tilized
        // Phase 8: Untilize (cb_tilized -> cb_out_rm)
        compute_kernel_lib::untilize<
            Wt,
            cb_tilized,
            cb_out_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
