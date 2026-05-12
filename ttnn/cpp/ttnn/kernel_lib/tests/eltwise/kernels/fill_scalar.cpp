// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FillScalar validation kernel — output tile filled with FILL_VALUE define.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"

#ifndef FILL_VALUE
#define FILL_VALUE 1.0f
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    // Fill-only chain (no CB-reader) — boot the engine using out_cb on both sides
    // (matches the legacy `EltwiseChainPipelineInit::run()` no-reader fallback).
    eltwise_chain_with_init(
        num_tiles, FillScalar<Dst::D0>{FILL_VALUE}, PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
