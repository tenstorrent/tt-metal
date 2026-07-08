// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tile-offset index test (G3 / OK-08, the TileOffset axis).
//
// Identity copy with a Block walker whose tile index is `base + i` (TileOffset::Set). The chain
// inflates its wait/pop counts by `base` and reads tiles [base, base+n) (eltwise_chain.hpp:294-326).
// `base` is a runtime constructor arg. Output[i] must equal input[base + i] -> a dropped base add
// reads from tile 0 and the golden mismatch is exact. TileOffset::Set requires a Bulk-family
// lifecycle (the static_assert in CopyTile), so this uses Bulk.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    const uint32_t base = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block, TileOffset::Set>{
            base},
        PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
}
