// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of untilize.cpp.
//
// Bindings (named, from host KernelSpec):
//   dfb::src                       — DFB endpoint (CONSUMER) — tiled input
//   dfb::dst                       — DFB endpoint (PRODUCER) — row-major output
//   args::per_core_block_tile_cnt  — CTA: tiles per block (template arg for untilize)
//   args::per_core_block_cnt       — RTA: number of blocks this core processes
//
// per_core_block_cnt is an RTA (was CTA in legacy) so a single compute KernelSpec
// handles full + cliff cores; satisfies the DFB "one producer / one consumer"
// invariant (dataflow_buffer_spec.hpp:46).

#include <cstdint>

#include "api/compute/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    auto per_core_block_cnt = get_arg(args::per_core_block_cnt);

    // For uneven nd-sharding, the host assigns 0 blocks to cores outside the populated
    // shard set. The kernel_lib untilize asserts num_blocks > 0; early-return on idle cores.
    if (per_core_block_cnt == 0) {
        return;
    }

    // dfb::src / dfb::dst implicitly convert to uint32_t for LLK APIs on WH/BH.
    compute_kernel_hw_startup(dfb::src, dfb::dst);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::src,
        dfb::dst,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
