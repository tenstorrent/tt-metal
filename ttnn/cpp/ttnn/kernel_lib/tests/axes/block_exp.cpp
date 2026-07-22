// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-capable exp(x): a Bulk + Block reader stages the window upfront so the chain processes
// block_size tiles per inner iter across DEST lanes. block_size is a compile-time arg.
//
// Blocking is a loop-structure optimization — it must NOT change the per-tile result, so exp(x) is
// identical across block_size; larger blocks should cut loop/DEST-sync overhead (the perf signal).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n, blk),
        CopyTile<input(cb_in, InputLifecycle::Bulk, OperandKind::Block), Dst::D0>{},
        Exp<>{},
        PackTile<output(cb_out, OutputLifecycle::Bulk)>{});
}
