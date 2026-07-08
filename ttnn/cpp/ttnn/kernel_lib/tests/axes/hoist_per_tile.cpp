// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Init-hoisting equality — PER-TILE side (G5 / init-hoisting).
//
// N separate eltwise_chain(EltwiseShape::single(), ...) calls, each emitting its own init for its single tile. This is
// the NON-hoisted-across-tiles shape: the same op (Exp) is re-initialized N times instead of once.
//
// Paired with hoist_single_call.cpp (one call over N tiles, init hoisted once). For identical input
// the two outputs must be bit-identical — proving the hoist (init once for N tiles) and the per-tile
// init produce exactly the same result for same-format data.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    for (uint32_t i = 0; i < n; ++i) {
        eltwise_chain(EltwiseShape::single(), CopyTile<cb_in, Dst::D0>{}, Exp<>{}, PackTile<cb_out>{});
    }
}
