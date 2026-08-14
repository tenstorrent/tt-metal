// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute (TRISC). One helper call for the whole core: the library
// tilize helper loops num_blocks tile-rows internally, so the LLK init/uninit
// is amortized across every block this core owns (master.md Part 1
// `compute_block_size`).
//
// WT_CHUNK (block_width_tiles) is the W block factor from op_design.md §1.4 and
// arrives as a compile-time arg — the helper needs it as a template parameter.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = 0;
    constexpr uint32_t cb_output_tiles = 16;

    constexpr uint32_t wt_chunk = get_compile_time_arg_val(0);
    constexpr uint32_t needs_cast = get_compile_time_arg_val(1);
    // Classification ablation (op_design.md §9.1): keep the per-block CB
    // handshake the helper would do, drop the tilize math. Always 0 in production.
    constexpr uint32_t ablate_compute = get_compile_time_arg_val(2);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);

    if (num_blocks == 0) {
        return;
    }

    using namespace compute_kernel_lib::tilize_config;

    if constexpr (ablate_compute) {
        for (uint32_t b = 0; b < num_blocks; ++b) {
            cb_wait_front(cb_input_sticks, wt_chunk);
            cb_reserve_back(cb_output_tiles, wt_chunk);
            cb_push_back(cb_output_tiles, wt_chunk);
            cb_pop_front(cb_input_sticks, wt_chunk);
        }
        return;
    }

    if constexpr (needs_cast) {
        // A real value-preserving cast: reconfigure both unpack and pack.
        compute_kernel_lib::tilize<
            wt_chunk,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            Fp32Mode::Fast>(num_blocks);
    } else {
        // Same format in and out — skip the ~150 ns register reconfiguration.
        compute_kernel_lib::tilize<
            wt_chunk,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            Fp32Mode::Fast>(num_blocks);
    }
}
