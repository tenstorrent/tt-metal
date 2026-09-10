// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A block-capable Exp chain that deliberately uses D1.  Its per-lane DEST footprint is two
// slots (D0 is unused but reserved by the lane layout), making a four-tile block the half-sync
// BF16 capacity boundary.  The Python test drives a partial final block through this path.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        IterationShape::tiles(n).block_size(block_size),
        CopyTile<input(cb_in, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block), Dst::D1>{},
        Exp<Approx::Exact, Dst::D1>{},
        PackTile<output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize), Dst::D1>{});
}
