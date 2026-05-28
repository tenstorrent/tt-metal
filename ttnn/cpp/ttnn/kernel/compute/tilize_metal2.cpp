// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of tilize.cpp.
//
// Tilizes rows of input data into tiled-format output, block by block.
//
// Bindings (named, from host KernelSpec):
//   dfb::src                       — DFB endpoint (CONSUMER) — row-major input
//   dfb::dst                       — DFB endpoint (PRODUCER) — tiled output
//   args::per_core_block_tile_cnt  — CTA: tiles per block (template arg for compute_kernel_lib::tilize)
//   args::per_core_block_cnt       — RTA: number of blocks this core processes
//
// Why per_core_block_cnt is an RTA (was CTA in legacy): Metal 2.0 enforces "one
// producer / one consumer" per DFB. If we needed N compute KernelSpecs (one per
// distinct block-count, e.g., full vs cliff cores), each would consume from the
// same SRC_DFB and violate the invariant. By keeping per_core_block_cnt as an
// RTA, a single compute KernelSpec serves all cores with per-core RTA variation.

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    auto per_core_block_cnt = get_arg(args::per_core_block_cnt);

    // dfb::src / dfb::dst implicitly convert to uint32_t for LLK APIs on WH/BH.
    compute_kernel_hw_startup(dfb::src, dfb::dst);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<dfb::src>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        dfb::src,
        dfb::dst,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(per_core_block_cnt);
}
