// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Model a reduction of partial results arriving from two sources (for example, local and fabric
// inputs in an all-reduce). Each row of tile pairs contributes to sticky D0 and produces one
// reduced output tile.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_local = tt::CBIndex::c_0;
    constexpr uint32_t cb_remote = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr bool caller_managed = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t num_outputs = get_compile_time_arg_val(3);
    static_assert(n > 0);
    static_assert(block_size > 0);
    static_assert(num_outputs > 0);

    compute_kernel_hw_startup(cb_local, cb_remote, cb_out);

    using namespace compute_kernel_lib;
    using Accumulate = BinaryFpu<
        cb_local,
        cb_remote,
        BinaryFpuOp::Add,
        BroadcastDim::None,
        InputLifecycle::Bulk,
        InputLifecycle::Bulk,
        BinaryDataFormatReconfig::Input,
        Dst::D0,
        OperandKind::Block,
        OperandKind::Block,
        TileOffset::Unset,
        TileOffset::Unset,
        DestAccumulation::Enabled>;
    using ManagedPack = PackTile<cb_out, OutputLifecycle::DestAccumulation, PackTileReconfig::Output, Dst::D0>;
    using CallerManagedPack =
        PackTile<cb_out, OutputLifecycle::DestAccumulationCallerManaged, PackTileReconfig::Output, Dst::D0>;

    CircularBuffer output(cb_out);
    using ManagedChain = EltwiseChain<Accumulate, ManagedPack>;
    using CallerManagedChain = EltwiseChain<Accumulate, CallerManagedPack>;
    // This reduction has no transient DEST lanes. Keep the zero-width capacity calculation
    // unbounded without ever instantiating a division by transient_lane_width.
    static_assert(chain_transient_lane_width_v<ManagedChain> == 0);
    static_assert(chain_transient_lane_width_v<CallerManagedChain> == 0);
    static_assert(chain_max_block_v<ManagedChain> == ~uint32_t{0});
    static_assert(chain_max_block_v<CallerManagedChain> == ~uint32_t{0});

    if constexpr (caller_managed) {
        output.reserve_back(num_outputs);
        eltwise_chain(EltwiseShape::grid(num_outputs, n, block_size), Accumulate{}, CallerManagedPack{});
        output.push_back(num_outputs);
    } else {
        eltwise_chain(EltwiseShape::grid(num_outputs, n, block_size), Accumulate{}, ManagedPack{});
    }
}
