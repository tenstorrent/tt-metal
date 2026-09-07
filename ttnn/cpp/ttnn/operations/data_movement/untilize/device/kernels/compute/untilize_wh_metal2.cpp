// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of untilize_wh.cpp, which lives beside it. Ops ported to Metal 2.0
// bind this file; the original serves the consumers still on the legacy API. Until the last of them
// migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::src, dfb::out) and named args are this fork's interface — shared with
// the sibling untilize_metal2.cpp and untilize_variable_num_blocks_metal2.cpp forks, so a factory can
// bind any of the three untilize compute kernels with one vocabulary.

#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto block_size_col = get_arg(args::block_size_col);
    constexpr auto block_size_row = get_arg(args::block_size_row);
    constexpr auto third_dim = get_arg(args::third_dim);

    // Each region binds the buffer set matching its block width, so which buffers this instance
    // works on is decided by the host binding rather than by a fixed index.
    compute_kernel_hw_startup(dfb::src, dfb::out);
    compute_kernel_lib::untilize<
        block_size_row,
        dfb::src,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
        block_size_col * third_dim);
}
