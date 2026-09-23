// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative lifecycle probe: a staged CB-window owner cannot share the CB with
// another reader that pops it.  The compile-time order argument exercises both
// directions of the collision check.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);
    constexpr bool owner_first = get_compile_time_arg_val(1) != 0;

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    using WindowOwner = CopyTile<input(cb_in, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block), Dst::D0>;
    using PeerPopper =
        CopyTile<input(cb_in, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Scalar), Dst::D1>;

    if constexpr (owner_first) {
        eltwise_chain(IterationShape::tiles(total_tiles), WindowOwner{}, PeerPopper{}, PackTile<output(cb_out)>{});
    } else {
        eltwise_chain(IterationShape::tiles(total_tiles), PeerPopper{}, WindowOwner{}, PackTile<output(cb_out)>{});
    }
}
