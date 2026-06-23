// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SA: SetupOwner::Caller on a chain whose setup is NOT fully boot-hoistable — must not compile.
//
// Exp and Sqrt are two DIFFERENT SFPU ops, so the chain's SFPU init is non-uniform
// (chain_hoist_sfpu_v is false) → the one-time setup can't all be emitted once at boot. There is
// then no single "before the loop" the caller can own, so SetupOwner::Caller is meaningless and
// the helper must reject it. (Reconfig is None throughout so the OTHER Caller assert can't fire
// first — this isolates the boot-hoistable guard.)

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain<SetupOwner::Caller>(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
        Exp<>{},
        Sqrt<>{},  // different SFPU op than Exp -> non-uniform -> not boot-hoistable
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
}
