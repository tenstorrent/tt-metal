// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of untilize_variable_num_blocks.cpp, which lives beside it. Ops
// ported to Metal 2.0 bind this file; the original serves the consumers still on the legacy API. Until
// the last of them migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::src, dfb::out) and named args are this fork's interface — shared with
// the sibling untilize_metal2.cpp fork so a factory can bind either compute kernel with one vocabulary.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto per_core_block_cnt = get_arg(args::per_core_block_cnt);

    // For uneven nd-sharding, the host assigns 0 blocks to cores that fall outside
    // the populated shard set. The kernel_lib untilize asserts num_blocks > 0,
    // so we early-return on those idle cores.
    if (per_core_block_cnt == 0) {
        return;
    }

    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    compute_kernel_hw_startup(dfb::src, dfb::out);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::src,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
