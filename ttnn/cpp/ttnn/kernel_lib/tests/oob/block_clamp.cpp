// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Out-of-bounds block_size probe (G6 / OOB-03, DB-04/DB-05).
//
// Block-capable identity copy: a Bulk + Block CB-reader stages the whole window upfront, so the
// chain honors block_size (processes block_size tiles per inner iter across DEST lanes j at slot
// D0 + j*chain_lane_width). block_size is supplied as a compile-time arg.
//
// The point: block_size is a RUNTIME field of EltwiseShape, so it can't be static_asserted. The
// chain instead clamps it at runtime to chain_max_block_v = DEST_AUTO_LIMIT / chain_lane_width
// (eltwise_chain.inl:2024-2033) — an oversized value can NOT overflow DEST; it only makes the
// outer loop take more iterations. Total tile coverage (and thus the output) is unchanged.
//
// So an over-large block_size must still produce the exact identity copy. The driving pytest
// feeds {1, 4, 1000} and asserts all three reproduce the input.

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
        CopyTile<cb_in, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
        PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output, Dst::D0>{});
}
