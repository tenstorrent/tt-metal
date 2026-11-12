// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the multi-core untilize whose block count follows the common row count; the row width
// stays a compile-time arg.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

void kernel_main() {
    // RUNTIME ARGS
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t num_tile_rows = get_arg(args::num_tile_rows);

    // COMPILE TIME ARGS
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    const auto [row_start, row_end] = ttnn::runtime_length::chunked_range(core_idx, num_tile_rows, num_cores);
    const uint32_t per_core_block_cnt = row_end - row_start;

    if (per_core_block_cnt == 0) {
        return;
    }

    compute_kernel_hw_startup(dfb::src, dfb::out);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::src,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(per_core_block_cnt);
}
