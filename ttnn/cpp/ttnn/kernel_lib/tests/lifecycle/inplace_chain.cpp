// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// In-place chain lifecycle suite (G1 output side, in-place variant). Run under --dev.
//
// "In-place" = the chain reads AND writes the SAME circular buffer (cb_x below): its input element
// reads cb_x and its PackTile writes cb_x. This is how an op overwrites a resident (e.g. sharded)
// buffer without a second CB. The hazard is a CB self-deadlock: the packer's cb_reserve_back cannot
// succeed while the reader's tiles still occupy the buffer.
//
// THE INVARIANT (why only some (InputLifecycle, OutputLifecycle) pairs are in-place-safe):
//   The chain's per-iter order is  wait -> read -> POP  (compute phase)  then  RESERVE -> pack -> push
//   (pack phase); see eltwise_chain.inl. The pop frees the slot BEFORE the reserve asks for one, so an
//   output that reserves INCREMENTALLY (per-tile / per-chunk) reuses the just-freed slot and the CB
//   stays at steady occupancy. An output that reserves UPFRONT (Bulk / ReserveAllPushPerTile /
//   ReserveAllPushPerChunk) issues cb_reserve_back(N) BEFORE the loop, while all N input tiles still
//   occupy cb_x -> 0 free slots -> DEADLOCK. Symmetrically the input must free space as it goes
//   (BulkDrain / Streaming pop-per-tile, or Chunked pop-per-chunk); an input that holds everything to
//   the end (HeldBulk) never frees a slot for the writer either.
//
//   => In-place needs an INCREMENTAL-RESERVE output paired with an INCREMENTAL-POP input.
//
// SHOWCASED (life selector) — every one PASSES (no-hang + correct values):
//   life  in-place chain on cb_x                     pattern            per-iter edges
//    0    BulkDrain(in) + Streaming(out), Scalar     "front rotation"   wait-all upfront, pop1/reserve1/push1 per tile
//    1    Chunked(in)   + Chunked(out),   Block      "chunk lockstep"   wait/pop K then reserve/push K per chunk
//    2    Streaming(in) + Streaming(out), Scalar     "per-tile rotation" wait1/pop1 then reserve1/push1 per tile
//
// NOT in-place-safe (would hang — deliberately NOT built as a passing case):
//    BulkDrain(in) + Bulk(out)  — Bulk reserves N upfront while cb_x holds N -> cb_reserve_back(N) hangs.
//    (Same for any *upfront-reserve* output: ReserveAllPushPerTile / ReserveAllPushPerChunk.)
//
// TOPOLOGY (why 3 stages, not just reader->cb_x->writer): a CB has ONE tiles_acked counter shared by
// every consumer. If the DRAM writer drained cb_x directly it would see N tiles available the instant
// the reader fills them and pop the PRE-transform data (a race), and a concurrent reader would be a
// second producer on cb_x. So cb_x is owned entirely by compute: stage 0 fills it from the reader's
// cb_src, stage A transforms it IN PLACE (the case under test), stage B copies it out to cb_out for
// the writer. Only stage A is the subject; stages 0/B are plain Bulk copies.
//
// CT args: [n, life, blk].

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

    // Stage 0: fill cb_x from the reader's cb_src (plain Bulk copy). After this, cb_x holds all n
    // input tiles and its only producer/consumer is the compute kernel itself.
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_src, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
        PackTile<cb_x, OutputLifecycle::Bulk, PackTileReconfig::Output>{});

    // Stage A: exp(x) IN PLACE on cb_x (read cb_x -> DEST -> exp -> pack cb_x). THE CASE UNDER TEST.
    if constexpr (life == 0) {
        // Front rotation: wait all n upfront, then each tile pop 1 (frees the front) then reserve 1
        // (reuses it) then push 1. Scalar index always reads the current front (the just-uncovered tile).
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_x, Dst::D0, InputLifecycle::BulkDrain, CopyTileReconfig::Input, OperandKind::Scalar>{},
            Exp<>{},
            PackTile<cb_x, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
    } else if constexpr (life == 1) {
        // Chunk lockstep: wait/pop K per chunk (frees K), then reserve/push K (reuses them). Block index
        // walks the just-waited K-tile front window.
        eltwise_chain(
            EltwiseShape::tiles(n, blk),
            CopyTile<cb_x, Dst::D0, InputLifecycle::Chunked, CopyTileReconfig::Input, OperandKind::Block>{},
            Exp<>{},
            PackTile<cb_x, OutputLifecycle::Chunked, PackTileReconfig::Output>{});
    } else {  // life == 2
        // Per-tile rotation: wait 1 / pop 1, then reserve 1 / push 1. Like life 0 but the wait is also
        // per tile instead of one upfront wait.
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_x, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input, OperandKind::Scalar>{},
            Exp<>{},
            PackTile<cb_x, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
    }

    // Stage B: copy cb_x -> cb_out (plain Bulk copy) so the DRAM writer drains cb_out, never cb_x.
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_x, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
        PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
}
