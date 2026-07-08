// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block-capable Exp via a CHUNKED reader, so the tile count can scale arbitrarily.
//
// Unlike the Bulk+Block variant (block_exp.cpp), a Chunked reader waits/pops block_size tiles per
// chunk, so the CB only needs ~block_size pages resident regardless of total n. That lets the perf
// sweep run very large n (thousands of tiles) without exhausting L1, while still exercising the
// block path (block_size tiles per inner iter across DEST lanes). out = exp(in).
//
// CT args: [n, block_size]. The host sizes cb_in/cb_out to a small multiple of block_size.

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
        CopyTile<cb_in, Dst::D0, InputLifecycle::Chunked, CopyTileReconfig::Input, OperandKind::Block>{},
        Exp<>{},
        PackTile<cb_out, OutputLifecycle::Chunked, PackTileReconfig::Output>{});
}
