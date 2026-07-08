// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Case F — compile-time elision. Same CB across consecutive chain elements on srca; fold's `if constexpr`
// guard elides every srca-side emission past element 0. Output PackTile is the only emission point.
//
// Chain shape: CopyTile(CbA, D0) -> CopyTile(CbA, D0) -> CopyTile(CbA, D0) -> PackTile(CbOut).
// At each CopyTile past element 0: curr_a == prev_a == CbA, reconf_a evaluates to false at compile
// time, no LLK call emitted. This kernel verifies the refactored fold preserves the elision path.
//
// Same-dtype here is fine — the case has no format delta to exercise. Net semantic = CbA (each CopyTile
// overwrites D0 with the same value).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_a, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
