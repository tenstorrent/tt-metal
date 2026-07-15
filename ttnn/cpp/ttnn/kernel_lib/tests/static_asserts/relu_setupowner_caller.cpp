// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: SetupOwner::Caller with a packer-ReLU PackTile. Under Caller the chain
// emits no one-time setup, so it never programs (or restores) STACC_RELU — a non-None ReLU knob is
// inert and misleading. All reconfig knobs are None here, so the ReLU is the ONLY thing that makes
// the chain request setup, isolating the guard.
// MUST fail to compile with "non-None reconfig knob".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    using namespace compute_kernel_lib;
    eltwise_chain<SetupOwner::Caller>(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
        PackTile<
            cb_out,
            OutputLifecycle::Streaming,
            PackTileReconfig::None,
            Dst::D0,
            TileOffset::Unset,
            PackTileL1Accumulation::Disabled,
            PackRelu::Zero>{});
}
