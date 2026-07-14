// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: TileOffset::Set requires a Bulk-family / CallerManaged
// lifecycle. A runtime base offset can't compose with iter-dependent (Streaming) wait/pop counts.
// Scalar+Streaming is itself legal, so ONLY the TileOffset guard fires here (clean isolation).
// MUST fail to compile with "TileOffset::Set requires".

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
        CopyTile<
            cb_in,
            Dst::D0,
            InputLifecycle::Streaming,
            CopyTileReconfig::Input,
            OperandKind::Scalar,
            TileOffset::Set>{0},
        PackTile<cb_out>{});
}
