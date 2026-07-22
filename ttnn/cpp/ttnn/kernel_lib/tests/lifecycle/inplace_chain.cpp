// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// In-place chain lifecycle: the chain reads AND writes the SAME CB (cb_x), overwriting a resident
// buffer with no second CB. Run under --dev. CT args: [n, life, blk].
//
// Hazard: CB self-deadlock — the packer's reserve can't succeed while the reader's tiles still
// occupy cb_x. Per-iter order is wait->read->POP then RESERVE->pack->push, so in-place is safe ONLY
// when the output reserves incrementally (per-tile/chunk) AND the input pops incrementally. The
// `life` cases are exactly the safe pairs; each must pass (no-hang + correct values):
//   0  BulkDrain + Streaming (Scalar)   wait all n upfront, then pop1/reserve1/push1 per tile
//   1  Chunked   + Chunked   (Block)    wait/pop K, then reserve/push K, per chunk
//   2  Streaming + Streaming (Scalar)   wait1/pop1, then reserve1/push1, per tile
// Any upfront-reserve output (Bulk / ReserveAllPush*) would deadlock and is deliberately not built.
// In-place safety is a pure function of this (InputLifecycle, OutputLifecycle) pairing; the compute
// elements between read and pack (DestReuseBinary, SFPU ops, ...) are DEST-internal and don't affect
// it — every case here holds the compute constant (Exp) and varies only the lifecycle pair.
//
// 3 stages keep the writer off cb_x: stage 0 Bulk-fills cb_x from the reader, stage A is the
// in-place transform under test, stage B Bulk-copies cb_x to cb_out for the writer.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_src = tt::CBIndex::c_0;   // reader fills this
    constexpr uint32_t cb_x = tt::CBIndex::c_1;     // the in-place buffer (read AND written by stage A)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;  // writer drains this

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t life = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);  // block_size for the Chunked case

    compute_kernel_hw_startup(cb_src, cb_out);

    using namespace compute_kernel_lib;

    // Stage 0: Bulk-fill cb_x from the reader; after this cb_x is owned entirely by compute.
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<input(cb_src, InputLifecycle::Bulk, OperandKind::Block), Dst::D0>{},
        PackTile<output(cb_x, OutputLifecycle::Bulk)>{});

    // Stage A: exp(x) IN PLACE on cb_x (read cb_x -> DEST -> exp -> pack cb_x). THE CASE UNDER TEST.
    if constexpr (life == 0) {
        // Front rotation: pop frees the front tile, reserve reuses it. Scalar reads the current front.
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<input(cb_x, InputLifecycle::BulkDrain), Dst::D0>{},
            Exp<>{},
            PackTile<output(cb_x)>{});
    } else if constexpr (life == 1) {
        // Chunk lockstep: pop/reserve K per chunk. Block index walks the K-tile front window.
        eltwise_chain(
            EltwiseShape::tiles(n, blk),
            CopyTile<input(cb_x, InputLifecycle::Chunked, OperandKind::Block), Dst::D0>{},
            Exp<>{},
            PackTile<output(cb_x, OutputLifecycle::Chunked)>{});
    } else {  // life == 2
        // Per-tile rotation: like life 0 but the wait is per-tile too.
        eltwise_chain(EltwiseShape::tiles(n), CopyTile<input(cb_x)>{}, Exp<>{}, PackTile<output(cb_x)>{});
    }

    // Stage B: copy cb_x -> cb_out (plain Bulk copy) so the DRAM writer drains cb_out, never cb_x.
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<input(cb_x, InputLifecycle::Bulk, OperandKind::Block), Dst::D0>{},
        PackTile<output(cb_out, OutputLifecycle::Bulk)>{});
}
