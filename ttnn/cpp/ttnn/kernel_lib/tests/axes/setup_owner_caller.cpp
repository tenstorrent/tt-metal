// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SetupOwner::Caller functional test (positive).
//
// SetupOwner::Caller means the CALLER owns the chain's one-time setup and emits it itself, so the
// chain emits none of it. That setup is raw LLK here in the caller — exactly what a SetupOwner::Chain
// call would hoist for this chain: CopyTile's copy_tile_init + Exp's exp_tile_init. (PackTile::init is
// empty and the CB-bound elements use reconfig None, so there's nothing else to emit.) The chain is
// then written ONCE and walks all n tiles under Caller. If Caller wrongly re-emitted setup, or if the
// raw setup didn't match what the chain expects, exp(x) would be wrong.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    using namespace compute_kernel_lib;

    // Caller-owned setup, as raw LLK (mirrors the chain's hoistable init for CopyTile + Exp + PackTile):
    compute_kernel_hw_startup(cb_in, cb_out);  // BIG hw init — always caller-owned
    copy_tile_init(cb_in);                     // == CopyTile::init()
    exp_tile_init();                           // == Exp<>::init()

    // Chain written once; SetupOwner::Caller emits no init/reconfig — it reuses the setup above.
    eltwise_chain<SetupOwner::Caller>(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
        Exp<>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
}
