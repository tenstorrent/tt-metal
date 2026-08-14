// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gather_strip bench compute — IDENTICAL in structure to the production
// tilize_compute.cpp: ONE `compute_kernel_lib::tilize` call for the whole core,
// so the LLK init/uninit is amortized across every block.
//
// The ONLY thing the strip layout changes is the two numbers that call takes:
//   ROW   mode: block_width_tiles = WT_CHUNK,     num_blocks = blocks
//   STRIP mode: block_width_tiles = PAGE_TILES,   num_blocks = blocks * slices
// The library helper itself is used verbatim in both arms — no substitution, no
// raw LLK. That is the compute-side contract this bench exists to prove.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = 0;
    constexpr uint32_t cb_output_tiles = 16;

    constexpr uint32_t compute_wt = get_compile_time_arg_val(0);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);

    if (num_blocks == 0) {
        return;
    }

    using namespace compute_kernel_lib::tilize_config;

    MaybeDeviceZoneScope("compute_tilize");
    compute_kernel_lib::tilize<
        compute_wt,
        cb_input_sticks,
        cb_output_tiles,
        InitUninitMode::InitAndUninit,
        WaitMode::WaitBlock,
        ReconfigureRegisterDatatypeMode::NoReconfigure,
        Fp32Mode::Fast>(num_blocks);
}
