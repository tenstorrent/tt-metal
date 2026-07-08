// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OptionalChainElement gating (untested mechanism).
//
// CopyTile -> OptionalChainElement<ON, Negative> -> PackTile. When ON the negate runs (out = -A);
// when OFF the element is a compile-time no-op (out = A). Proves the gate both applies and elides,
// and that the false stub of a DEST-only op is a valid (no-op) chain element. `cond` is a CT arg.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t cond = get_compile_time_arg_val(1);
    constexpr bool ON = (cond != 0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0>{},
        OptionalChainElement<ON, Negative<Dst::D0>>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{});
}
