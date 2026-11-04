// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of untilize.cpp. Identical compute logic; the CB indices and loop counts are
// sourced from Metal 2.0 named bindings (dfb::in / dfb::out) and named compile-time args instead of
// positional CTAs / CB-index CTAs.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);
    compute_kernel_lib::untilize<
        per_core_block_tile_cnt,
        dfb::in,
        dfb::out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
        in, out, per_core_block_cnt);
}
