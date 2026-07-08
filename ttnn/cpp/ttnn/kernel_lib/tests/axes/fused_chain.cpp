// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Realistic fused chain for the Chunked-vs-Bulk perf comparison: out = exp(A + B) * C.
//   D0 = A + B            (BinaryFpu, FPU add)
//   D0 = exp(D0)          (Exp, SFPU unary)
//   D0 = DEST(D0) * C     (DestReuseBinary Mul, DEST_TO_SRCA: C -> srcb, DEST -> srca)
//   pack D0
//
// CRUCIAL: neither lifecycle stages the whole N-tile window. Both keep a BOUNDED CB and process N
// over multiple iterations:
//   - Bulk (life=0): an OUTER LOOP over batches of `batch` tiles — each iter is a Bulk eltwise_chain
//     over a `batch`-tile window (CB ~ batch pages), re-initialised per batch. This is how you'd
//     actually run Bulk at large N (you cannot hold 1024 tiles in L1).
//   - Chunked (life=1): a SINGLE eltwise_chain over all N, waiting/popping `block_size` per chunk
//     (CB ~ block_size pages). No re-init, reader/compute overlap per chunk.
// So CB footprint is bounded by `batch` / `block_size`, NOT by N — N scales to thousands.
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

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);  // one boot covers every batch (moreh inner-loop pattern)

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
