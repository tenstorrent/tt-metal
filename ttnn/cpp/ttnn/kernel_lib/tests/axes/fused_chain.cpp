// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Realistic fused chain for the Chunked-vs-Bulk perf comparison: out = exp(A + B) * C
// (BinaryFpu add -> Exp -> DestReuseBinary mul, all in D0).
//
// Both lifecycles keep a BOUNDED CB (footprint set by `batch`/`block_size`, not N), so N scales to
// thousands:
//   - Bulk (life=0):    outer loop over `batch`-tile windows, a re-initialised Bulk chain per batch.
//   - Chunked (life=1): one chain over all N, waiting/popping `block_size` per chunk (no re-init).
//
// CT args: [n, block_size, life, batch].

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t life = get_compile_time_arg_val(2);   // 0 = Bulk (batched), 1 = Chunked
    constexpr uint32_t batch = get_compile_time_arg_val(3);  // Bulk batch window (tiles per chain call)

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);  // one boot covers every batch

    using namespace compute_kernel_lib;
    if constexpr (life == 0) {  // Bulk, batched over the whole N with a bounded `batch` window
        for (uint32_t off = 0; off < n; off += batch) {
            eltwise_chain(
                EltwiseShape::tiles(batch, blk),
                BinaryFpu<
                    cb_a,
                    cb_b,
                    BinaryFpuOp::Add,
                    BroadcastDim::None,
                    InputLifecycle::Bulk,
                    InputLifecycle::Bulk,
                    BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Block,
                    OperandKind::Block>{},
                Exp<>{},
                DestReuseBinary<
                    cb_c,
                    BinaryFpuOp::Mul,
                    DestReuseType::DEST_TO_SRCA,
                    InputLifecycle::Bulk,
                    DestReuseReconfig::Input,
                    Dst::D0,
                    Dst::D0,
                    OperandKind::Block>{},
                PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
        }
    } else {  // Chunked: single call over all N, bounded CB via per-chunk wait/pop
        eltwise_chain(
            EltwiseShape::tiles(n, blk),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                InputLifecycle::Chunked,
                InputLifecycle::Chunked,
                BinaryDataFormatReconfig::Input,
                Dst::D0,
                OperandKind::Block,
                OperandKind::Block>{},
            Exp<>{},
            DestReuseBinary<
                cb_c,
                BinaryFpuOp::Mul,
                DestReuseType::DEST_TO_SRCA,
                InputLifecycle::Chunked,
                DestReuseReconfig::Input,
                Dst::D0,
                Dst::D0,
                OperandKind::Block>{},
            PackTile<cb_out, OutputLifecycle::Chunked, PackTileReconfig::Output>{});
    }
}
