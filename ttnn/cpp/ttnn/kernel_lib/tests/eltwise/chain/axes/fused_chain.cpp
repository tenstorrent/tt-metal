// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Realistic fused chain for the PerBlockSize-vs-Bulk perf comparison: out = exp(A + B) * C
// (BinaryFpu add -> Exp -> DestReuseBinary mul, all in D0).
//
// Both lifecycles keep a BOUNDED CB (footprint set by `batch`/`block_size`, not N), so N scales to
// thousands:
//   - Bulk (life=0):    outer loop over `batch`-tile windows, a re-initialised Bulk chain per batch.
//   - PerBlockSize (life=1): one chain over all N, waiting/popping once per block-size group (no re-init).
//
// CT args: [n, block_size, life, batch].

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);

    using FastExpD1 = compute_kernel_lib::Exp<compute_kernel_lib::Approx::Fast, compute_kernel_lib::Dst::D1>;
    static_assert(FastExpD1::lane_width == 2);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t life = get_compile_time_arg_val(2);   // 0 = Bulk (batched), 1 = PerBlockSize
    constexpr uint32_t batch = get_compile_time_arg_val(3);  // Bulk batch window (tiles per chain call)

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);  // one boot covers every batch

    using namespace compute_kernel_lib;
    if constexpr (life == 0) {  // Bulk, batched over the whole N with a bounded `batch` window
        for (uint32_t off = 0; off < n; off += batch) {
            eltwise_chain(
                IterationShape::tiles(batch).block_size(blk),
                BinaryFpu<
                    BinaryFpuOp::Add,
                    input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                    input(cb_b, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)>{},
                Exp<>{},
                DestReuseBinary<
                    BinaryFpuOp::Mul,
                    input(cb_c, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                    DestReuseType::DEST_TO_SRCA>{},
                PackTile<output(cb_out, ReservePolicy::Upfront, PushPolicy::AtEnd)>{});
        }
    } else {  // PerBlockSize: single call over all N, bounded CB via per-block-size wait/pop
        eltwise_chain(
            IterationShape::tiles(n).block_size(blk),
            BinaryFpu<
                BinaryFpuOp::Add,
                input(cb_a, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                input(cb_b, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block)>{},
            Exp<>{},
            DestReuseBinary<
                BinaryFpuOp::Mul,
                input(cb_c, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                DestReuseType::DEST_TO_SRCA>{},
            PackTile<output(cb_out, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    }
}
