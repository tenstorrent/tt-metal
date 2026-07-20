// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Out-of-bounds block_size probe: block-capable identity copy (Bulk + Block reader).
//
// block_size is a RUNTIME EltwiseShape field, so it can't be static_asserted; the chain clamps it
// to chain_max_block_v = DEST_AUTO_LIMIT / chain_lane_width. An oversized block_size can't overflow
// DEST — it only changes loop iteration count, not coverage. So {1, 4, 1000} must all reproduce the
// input exactly (the driving pytest feeds those).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n, blk),
        CopyTile<cb_in, Dst::D0, input(InputLifecycle::Bulk, OperandKind::Block)>{},
        PackTile<cb_out, output(OutputLifecycle::Bulk), Dst::D0>{});
}
