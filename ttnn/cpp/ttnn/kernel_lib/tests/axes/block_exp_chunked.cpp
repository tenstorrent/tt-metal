// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-capable exp(in) via a CHUNKED reader: waits/pops block_size tiles per chunk, so the CB
// holds only ~block_size pages regardless of n. Unlike block_exp.cpp (Bulk+Block), this lets the
// perf sweep run very large n without exhausting L1 while still exercising the block path.
//
// CT args: [n, block_size]. Host sizes cb_in/cb_out to a small multiple of block_size.

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
        CopyTile<input(cb_in, InputLifecycle::Chunked, OperandKind::Block), Dst::D0>{},
        Exp<>{},
        PackTile<output(cb_out, OutputLifecycle::Chunked)>{});
}
