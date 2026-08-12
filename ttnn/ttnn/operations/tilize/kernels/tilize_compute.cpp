// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute — the single compute phase, entirely helper-driven.
//
// `block_width_tiles` IS the block-factor knob (tilize_helpers.hpp:188): two
// instantiations, one at WT_BLOCK for the full-width column-blocks and one at
// WT_TAIL for the tail column-block. Because a core's contiguous block range
// crosses the full/tail boundary at most once, the two runtime counts
// (n_full, n_tail) cover it with no per-core kernel variant.
//
// Knob settings, each a decision (op_design.md §7.1):
//   init_uninit_mode = InitAndUninit on BOTH calls — the two calls use different
//       block_width_tiles and tilize_init takes the width, so each needs its own
//       init.
//   wait_mode        = WaitBlock — per-block wait is what lets the reader run
//       ahead of compute; WaitUpfront would serialize the core behind the reader.
//   reconfig_mode    = NoReconfigure when there is nothing to cast (the
//       reconfigure exists only to drive a dtype= cast and is otherwise a fixed
//       ~150 ns waste), UnpackAndPackReconfigure on a real cast.
//   fp32_mode        = Fast — tilize's consumers are FPU ops, which re-truncate
//       to tf32 anyway; Lossless would only cost a slower path.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t wt_block = get_compile_time_arg_val(2);
    constexpr uint32_t wt_tail = get_compile_time_arg_val(3);
    constexpr bool needs_cast = get_compile_time_arg_val(4) == 1;

    const uint32_t n_full = get_arg_val<uint32_t>(0);
    const uint32_t n_tail = get_arg_val<uint32_t>(1);

    using namespace compute_kernel_lib::tilize_config;
    constexpr ReconfigureRegisterDatatypeMode reconfig_mode =
        needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                   : ReconfigureRegisterDatatypeMode::NoReconfigure;

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);

    // `tilize` ASSERTs num_blocks > 0, so both calls are guarded.
    if (n_full > 0) {
        compute_kernel_lib::tilize<
            wt_block,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            reconfig_mode,
            Fp32Mode::Fast>(n_full);
    }
    if (n_tail > 0) {
        compute_kernel_lib::tilize<
            wt_tail,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            reconfig_mode,
            Fp32Mode::Fast>(n_tail);
    }
}
