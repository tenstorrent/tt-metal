// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp.
// The legacy fold DRAM (tiled) factory file-path-instantiated the untilize op's compute kernel. That
// kernel is shared with the untilize op (still on the legacy API), so it cannot be Metal-2.0-ified in
// place without breaking untilize. This fold-owned fork carries the Metal 2.0 rewrite (named args +
// DFB bindings) for fold's use only. See METAL2_PORT_REPORT.md (Open items for downstream).

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    compute_kernel_hw_startup(dfb::src, dfb::out);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::src,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
