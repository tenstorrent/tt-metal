// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tile-offset index test (the TileOffset axis): identity copy with a Block walker at tile `base + i`
// (TileOffset::Set); the chain inflates its wait/pop counts by `base` and reads tiles [base, base+n).
// `base` is a runtime ctor arg. output[i] must equal input[base+i] — a dropped base add reads tile 0
// and mismatches. TileOffset::Set requires a Bulk-family lifecycle (CopyTile static_assert), so uses Bulk.

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
        CopyTile<
            cb_in,
            Dst::D0,
            input(InputLifecycle::Bulk, OperandKind::Block, DataFormatReconfig::Enabled, TileOffset::Set)>{base},
        PackTile<cb_out, output(OutputLifecycle::Bulk)>{});
}
