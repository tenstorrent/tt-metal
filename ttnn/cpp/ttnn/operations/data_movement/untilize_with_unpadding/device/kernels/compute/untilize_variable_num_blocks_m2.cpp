// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// //
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp.
// The legacy source is shared, so it is forked here (not edited in place) and ported to Metal 2.0
// named bindings for untilize_with_unpadding's multi-core ND-sharded factory.
// Logic, loops and numeric paths are UNCHANGED; only the access mechanism moves to named bindings:
//   CB ids               -> dfb::in / dfb::out
//   per_core_block_tile_cnt CTA -> named CTA (get_arg(args::...))
//   num_input_blocks_to_process RTA -> named RTA (get_arg(args::num_input_blocks_to_process))

#include <cstdint>

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg(args::num_input_blocks_to_process);

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
