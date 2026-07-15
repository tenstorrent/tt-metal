// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: packer ReLU on an L1-accumulating PackTile. Whether the packer ReLU clamps
// the accumulated sum or the pre-accumulation DEST value is unverified, so the combination is
// forbidden (per-element PackTile static_assert).
// MUST fail to compile with "packer ReLU combined with L1 accumulation".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0>{},
        PackTile<
            cb_out,
            OutputLifecycle::L1Accumulation,
            PackTileReconfig::None,
            Dst::D0,
            TileOffset::Unset,
            PackTileL1Accumulation::Enabled,
            PackRelu::Zero>{});
}
