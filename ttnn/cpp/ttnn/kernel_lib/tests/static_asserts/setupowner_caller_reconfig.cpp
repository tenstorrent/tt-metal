// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SA: SetupOwner::Caller with a chain that still requests reconfig — must not compile.
//
// This chain IS fully boot-hoistable (uniform: one CopyTile math-MOP + one Exp SFPU + one pack),
// so the boot-hoistable guard passes. But CopyTile's default reconfig is Input (and PackTile's is
// Output), i.e. the element asks the chain to emit a reconfig. Under SetupOwner::Caller the chain
// emits NO reconfig — the caller owns the setup — so a non-None reconfig knob is inert and lies
// about what runs inside the helper. The helper must reject it and force the caller to declare None.

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
        CopyTile<cb_in, Dst::D0>{},  // default CopyTileReconfig::Input -> requests reconfig
        Exp<>{},
        PackTile<cb_out>{});  // default PackTileReconfig::Output
}
