// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: SetupOwner::Caller with a chain that still requests reconfig.
//
// The chain is fully boot-hoistable so that guard passes, but input/output specs enable reconfig
// by default. Under Caller the chain emits no reconfig, so an enabled operand spec is inert and
// misleading — the helper rejects it and forces the caller to disable reconfig.
// MUST fail to compile with "enabled operand reconfig".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain<SetupOwner::Caller>(EltwiseShape::tiles(n), CopyTile<cb_in, Dst::D0>{}, Exp<>{}, PackTile<cb_out>{});
}
