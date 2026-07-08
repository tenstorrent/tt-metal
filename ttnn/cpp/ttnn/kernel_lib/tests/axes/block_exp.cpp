// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blocking workload (G4 / blocking perf + correctness).
//
// Block-capable exp(x): a Bulk + Block CB-reader stages the window upfront so the chain processes
// block_size tiles per inner iter across DEST lanes (DEST[D0 + j*chain_lane_width], j in
// [0, block_size)). block_size is a compile-time arg.
//
// Blocking is a loop-structure optimization: it changes how many outer iterations run and how many
// DEST lanes are filled per iter, but NOT the per-tile result. So exp(x) must be identical across
// block_size, and larger blocks should reduce loop/DEST-sync overhead (the perf signal).

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
        CopyTile<cb_in, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
        Exp<>{},
        PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
}
