// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of untilize_variable_num_blocks.cpp. Identical compute logic; the CB indices become
// dfb:: bindings, the per-block tile count a named compile-time arg, and the variable per-core block
// count a named runtime arg. The legacy copy is retained for the not-yet-ported sharded untilize
// factories that still instantiate it.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);

    // For uneven nd-sharding, the host assigns 0 blocks to cores that fall outside
    // the populated shard set. The kernel_lib untilize asserts num_blocks > 0,
    // so we early-return on those idle cores.
    if (per_core_block_cnt == 0) {
        return;
    }

    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::in,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
