// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: two PackTile elements writing the same (CB, DEST slot) would race. The
// chain static_asserts on the collision (chain_pack_writes_collide).
// MUST fail to compile with "eltwise_chain: two PackTile elements collide on (dfb, dst_slot)".

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
        CopyTile<cb_in, Dst::D0>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{});
}
