// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of untilize.cpp.
//
// The legacy copy next to this file stays in place for the ~10 factories that have not yet
// migrated to the Metal 2.0 host API. Delete it once the last of them ports; until then, any
// bug fix to the legacy copy should be evaluated for this one.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::in,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
