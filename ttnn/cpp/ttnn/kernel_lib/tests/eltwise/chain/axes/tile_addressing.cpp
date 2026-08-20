// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tile-offset index test (the TileAddressing axis): identity copy with a Block walker at tile `base + i`
// (TileAddressing::Offset); the chain inflates its wait/pop counts by `base` and reads tiles [base, base+n).
// `base` is a runtime ctor arg. output[i] must equal input[base+i] — a dropped base add reads tile 0
// and mismatches. TileAddressing::Offset requires a Bulk-family lifecycle (CopyTile static_assert), so uses Bulk.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    const uint32_t base = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        IterationShape::tiles(n),
        CopyTile<
            input(
                cb_in,
                WaitPolicy::Upfront,
                PopPolicy::AtEnd,
                InputTileMapping::Block,
                DataFormatReconfig::Enabled,
                TileAddressing::Offset),
            Dst::D0>{base},
        PackTile<output(cb_out, ReservePolicy::Upfront, PushPolicy::AtEnd)>{});
}
