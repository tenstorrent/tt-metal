// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Init-hoisting equality — HOISTED side: one eltwise_chain over N tiles. The chain is uniform
// (single CopyTile math-MOP + single SFPU op, consistent CBs), so the per-element init is emitted
// ONCE at boot (hoist_compute_init), not per tile. Paired with hoist_per_tile.cpp; for identical
// input the two outputs must be bit-identical — hoisting is a pure emission optimization.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(EltwiseShape::tiles(n), CopyTile<cb_in, Dst::D0>{}, Exp<>{}, PackTile<cb_out>{});
}
