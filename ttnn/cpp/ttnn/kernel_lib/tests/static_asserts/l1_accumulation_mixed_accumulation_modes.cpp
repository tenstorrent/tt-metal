// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    using namespace compute_kernel_lib;
    using PreloadedPack = PackTile<
        cb_out,
        OutputLifecycle::L1AccumulationCallerManaged,
        PackTileReconfig::Output,
        Dst::D0,
        TileOffset::Unset,
        PackTileL1Accumulation::Enabled>;
    using SeedFirstPack = PackTile<
        cb_out,
        OutputLifecycle::L1AccumulationCallerManaged,
        PackTileReconfig::Output,
        Dst::D1,
        TileOffset::Unset,
        PackTileL1Accumulation::SeedFirst>;
    eltwise_chain(EltwiseShape::tiles(n), CopyTile<cb_in>{}, PreloadedPack{}, SeedFirstPack{});
}
