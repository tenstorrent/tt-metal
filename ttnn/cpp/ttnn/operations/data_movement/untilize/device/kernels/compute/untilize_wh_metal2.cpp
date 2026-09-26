// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This forked untilize_wh.cpp during the Metal 2.0 migration. Its last legacy binder has
// migrated and the original is retired, so this is the only copy and the _metal2 suffix is historical.
//
// The binding names below (dfb::src, dfb::out) and named args are this fork's interface — shared with
// the sibling untilize_metal2.cpp / untilize_variable_num_blocks_metal2.cpp forks so a factory can
// bind any of the untilize compute kernels with one vocabulary.

#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t block_size_col = get_arg(args::block_size_col);
    const uint32_t block_size_row = get_arg(args::block_size_row);
    const uint32_t third_dim = get_arg(args::third_dim);

    // Each region binds the buffer set matching its block width: `src` / `out` are that set's input
    // and output buffers -- see BlockBufferSet in data_movement/common.
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
