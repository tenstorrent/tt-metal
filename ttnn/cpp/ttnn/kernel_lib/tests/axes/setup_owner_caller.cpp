// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SetupOwner::Caller functional test (positive): the caller owns the chain's one-time setup and
// emits it as raw LLK (copy_tile_init + exp_tile_init) — exactly what SetupOwner::Chain would hoist.
// The chain then walks all n tiles emitting none of it. If Caller wrongly re-emitted setup, or the
// raw setup didn't match what the chain expects, exp(x) would be wrong.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    using namespace compute_kernel_lib;

    // Caller-owned setup as raw LLK (mirrors the chain's hoistable init):
    compute_kernel_hw_startup(cb_in, cb_out);  // BIG hw init — always caller-owned
    copy_tile_init(cb_in);                     // == CopyTile::init()
    exp_tile_init();                           // == Exp<>::init()

    // Chain written once; SetupOwner::Caller emits no init/reconfig — it reuses the setup above.
    eltwise_chain<SetupOwner::Caller>(
        EltwiseShape::tiles(n),
        CopyTile<input(cb_in, InputLifecycle::Streaming, DataFormatReconfig::Disabled), Dst::D0>{},
        Exp<>{},
        PackTile<output(cb_out, OutputLifecycle::Streaming, DataFormatReconfig::Disabled)>{});
}
