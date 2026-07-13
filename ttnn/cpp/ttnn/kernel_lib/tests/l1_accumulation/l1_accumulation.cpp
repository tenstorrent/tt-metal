// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Functional coverage for both L1-accumulation output lifecycles. The first input tile seeds a
// one-page accumulator CB normally. After cycling that page through push/pop, the L1-accumulating
// chain adds every remaining input tile into the same physical L1 tile. The final chain copies the
// accumulated tile to the externally visible output CB.

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

    // Seed the accumulator without L1 accumulation, then cycle the one-page CB so its write pointer
    // returns to the same physical tile while preserving the packed data in L1.
    accumulator.reserve_back(1);
    eltwise_chain(
        EltwiseShape::single(),
        CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
        PackTile<cb_acc, OutputLifecycle::CallerManaged, PackTileReconfig::None>{});
    accumulator.push_back(1);
    accumulator.wait_front(1);
    accumulator.pop_front(1);

    using L1ManagedPack = PackTile<
        cb_acc,
        OutputLifecycle::L1Accumulation,
        PackTileReconfig::None,
        Dst::D0,
        TileOffset::Unset,
        PackTileL1Accumulation::Enabled>;
    using L1CallerManagedPack = PackTile<
        cb_acc,
        OutputLifecycle::L1AccumulationCallerManaged,
        PackTileReconfig::None,
        Dst::D0,
        TileOffset::Unset,
        PackTileL1Accumulation::Enabled>;

    if constexpr (caller_managed) {
        accumulator.reserve_back(1);
        eltwise_chain(
            EltwiseShape::tiles(n - 1),
            CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
            L1CallerManagedPack{});
        accumulator.push_back(1);
    } else {
        eltwise_chain(
            EltwiseShape::tiles(n - 1),
            CopyTile<cb_in, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::None>{},
            L1ManagedPack{});
    }

    eltwise_chain(
        EltwiseShape::single(),
        CopyTile<cb_acc, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
}
