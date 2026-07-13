// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Functional coverage for both L1-accumulation output lifecycles. One seed-first chain overwrites
// the accumulator with the first input tile, enables L1 accumulation for the remaining tiles, and
// restores overwrite mode before publishing. The final chain copies the accumulated tile to the
// externally visible output CB.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_acc = tt::CBIndex::c_15;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr bool caller_managed = get_compile_time_arg_val(1) != 0;
    static_assert(n > 1);

    compute_kernel_hw_startup(cb_in, cb_acc);

    using namespace compute_kernel_lib;
    CircularBuffer accumulator(cb_acc);

    using L1ManagedPack = PackTile<
        cb_acc,
        OutputLifecycle::L1Accumulation,
        PackTileReconfig::None,
        Dst::D0,
        TileOffset::Unset,
        PackTileL1Accumulation::SeedFirst>;
    using L1CallerManagedPack = PackTile<
        cb_acc,
        OutputLifecycle::L1AccumulationCallerManaged,
        PackTileReconfig::None,
        Dst::D0,
        TileOffset::Unset,
        PackTileL1Accumulation::SeedFirst>;

    if constexpr (caller_managed) {
        accumulator.reserve_back(1);
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
            L1CallerManagedPack{});
        accumulator.push_back(1);
    } else {
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
            L1ManagedPack{});
    }

    eltwise_chain(
        EltwiseShape::single(),
        CopyTile<cb_acc, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
}
