// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OptionalChainElement gating a PackTile fan-out (the memory #16 false-stub case).
//
// CopyTile -> PackTile(cb_out0) -> OptionalChainElement<ON, PackTile(cb_out1)>. When ON, DEST is
// packed to BOTH cb_out0 and cb_out1 (fan-out; distinct CBs so no pack collision). When OFF, the
// optional PackTile's FALSE stub must still be a valid chain element — it must expose pack_dst_slot
// (the bug that was fixed: writer_pair_collide reads it). So OFF must compile and write only cb_out0.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t cond = get_compile_time_arg_val(1);
    constexpr bool ON = (cond != 0);

    compute_kernel_hw_startup(cb_in, cb_out0);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_in, Dst::D0>{},
        PackTile<cb_out0, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{},
        OptionalChainElement<ON, PackTile<cb_out1, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>>{});
}
