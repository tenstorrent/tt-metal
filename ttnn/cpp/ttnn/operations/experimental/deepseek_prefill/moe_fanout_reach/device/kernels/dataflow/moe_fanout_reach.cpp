// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This chip's row of the multicast reach table: for each direction round the ring and each hop h,
// how many of this chip's tokens still have a surviving destination at least h hops away.
//
// Reach is POST-DROP, which is what forces a token walk rather than a count. A pick is dropped when
// its per-expert allocator has already passed the destination buffer's capacity, and that allocator
// is seeded from the offsets table and advanced by every earlier pick of the same expert -- including
// the dropped ones. A token whose every surviving page is local must not hold a hop open: the origin
// would then send fewer pages than the relay downstream waits for, and the axis stops rather than
// producing wrong numbers.
//
// The walk is split over a grid in four phases:
//
//   1. Each core counts its own token range's picks per expert.
//   2. A Hillis-Steele scan over the cores turns those counts into each core's starting allocator
//      value per expert. The rank of a pick is the number of earlier picks of the same expert, and
//      the ranges are contiguous and in core order, so the scan reproduces the sequential allocator
//      exactly rather than approximating it.
//   3. Each core re-walks its tokens from that base, applies the drop rule, and reduces each token to
//      its farthest surviving hop per direction.
//   4. A binary tree sums the two rows across the cores; the root writes them out.
//
// The experts are compacted first: only those the dispatch table places on this axis can ever be
// counted, so the scan carries one word per PRESENT expert instead of one per routed expert. Every
// core derives the same compaction from the same replicated table, which is what lets the scan
// vectors be added word for word.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ckernel.h"

#include "moe_fanout_reach_kernel_interface.hpp"

void kernel_main() {
    Noc noc;

    const uint32_t indices_addr = get_arg_val<uint32_t>(mfr::RtArg::kIndicesAddr);
    const uint32_t table_addr = get_arg_val<uint32_t>(mfr::RtArg::kTableAddr);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(mfr::RtArg::kOffsetsAddr);
    const uint32_t out_dram_addr = get_arg_val<uint32_t>(mfr::RtArg::kOutAddr);
    const uint32_t tok_start = get_arg_val<uint32_t>(mfr::RtArg::kTokStart);
    const uint32_t tok_count = get_arg_val<uint32_t>(mfr::RtArg::kTokCount);

    constexpr uint32_t num_routed_experts = get_compile_time_arg_val(mfr::CtArg::kNumRoutedExperts);
    constexpr uint32_t topk = get_compile_time_arg_val(mfr::CtArg::kTopk);
    constexpr uint32_t extent = get_compile_time_arg_val(mfr::CtArg::kExtent);
    constexpr uint32_t my_row = get_compile_time_arg_val(mfr::CtArg::kMyRow);
    constexpr uint32_t capacity = get_compile_time_arg_val(mfr::CtArg::kCapacity);
    constexpr uint32_t hops = get_compile_time_arg_val(mfr::CtArg::kHops);
    constexpr uint32_t tokens_per_core = get_compile_time_arg_val(mfr::CtArg::kTokensPerCore);
    constexpr uint32_t rounds = get_compile_time_arg_val(mfr::CtArg::kRounds);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(mfr::CtArg::kOutPageBytes);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(mfr::CtArg::kGatherSemId);

    constexpr mfr::Geometry geom{
        .num_routed_experts = num_routed_experts,
        .topk = topk,
        .tokens_per_core = tokens_per_core,
        .rounds = rounds,
        .out_page_bytes = out_page_bytes};
    constexpr uint32_t rec_stride = mfr::indices_stride(topk);
    constexpr uint32_t scan_stride = mfr::scan_row_bytes(num_routed_experts);
    constexpr uint32_t out_words = out_page_bytes / 4u;
    // The farthest hop a token can have; the table's terminating zero sits one past it.
    constexpr uint32_t max_hop = hops - 2u;

    constexpr auto idx_args = TensorAccessorArgs<mfr::kAccessorBase>();
    const auto idx_acc = TensorAccessor(idx_args, indices_addr);
    constexpr uint32_t tbl_accessor_offset = idx_args.next_compile_time_args_offset();
    constexpr auto tbl_args = TensorAccessorArgs<tbl_accessor_offset>();
    const auto tbl_acc = TensorAccessor(tbl_args, table_addr);
    constexpr uint32_t off_accessor_offset = tbl_args.next_compile_time_args_offset();
    constexpr auto off_args = TensorAccessorArgs<off_accessor_offset>();
    const auto off_acc = TensorAccessor(off_args, offsets_addr);
    constexpr uint32_t out_accessor_offset = off_args.next_compile_time_args_offset();
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, out_dram_addr);

    CircularBuffer cb_carve(mfr::kCbCarve);
    // Rounded up so every block starts on a 64-byte line, which a DRAM read's L1 destination has to be
    // on Blackhole. Every core rounds the same circular-buffer base, so a peer's block addresses match.
    const uint32_t base = mfr::align64(cb_carve.get_write_ptr());
    const uint32_t l1_indices = base + mfr::block_offset(mfr::kIndices, geom);
    const uint32_t l1_table = base + mfr::block_offset(mfr::kTable, geom);
    const uint32_t l1_offsets = base + mfr::block_offset(mfr::kOffsets, geom);
    const uint32_t l1_info = base + mfr::block_offset(mfr::kInfo, geom);
    const uint32_t l1_coffs = base + mfr::block_offset(mfr::kCompactOffsets, geom);
    const uint32_t l1_scan = base + mfr::block_offset(mfr::kScan, geom);
    const uint32_t l1_alloc = base + mfr::block_offset(mfr::kAlloc, geom);
    const uint32_t l1_out = base + mfr::block_offset(mfr::kOut, geom);
    const uint32_t l1_gather = base + mfr::block_offset(mfr::kGather, geom);

    // ---- Phase 1a: pull the control tables and this core's token records out of DRAM. ----
    for (uint32_t t = 0; t < tok_count; t++) {
        noc.async_read(
            idx_acc, CoreLocalMem<uint32_t>(l1_indices + t * rec_stride), topk * 2u, {.page_id = tok_start + t}, {});
    }
    noc.async_read(tbl_acc, CoreLocalMem<uint32_t>(l1_table), num_routed_experts * 4u, {.page_id = 0}, {});
    noc.async_read(off_acc, CoreLocalMem<uint32_t>(l1_offsets), num_routed_experts * 4u, {.page_id = 0}, {});
    noc.async_read_barrier();

    // ---- Phase 1b: expert -> (destination row, compacted index), identically on every core. ----
    //
    // The row is bounded against the table rather than trusted: an entry naming a row off the axis
    // would put a hop in the table that no chip can serve, and a relay would then wait for pages the
    // origin never sends. Refusing the expert makes a malformed table produce no reach at all.
    volatile tt_l1_ptr int32_t* table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_table);
    volatile tt_l1_ptr uint32_t* offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_offsets);
    volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_info);
    volatile tt_l1_ptr uint32_t* coffs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_coffs);
    uint32_t n_compact = 0;
    for (uint32_t e = 0; e < num_routed_experts; e++) {
        const int32_t row = table[e];
        if (row < 0 || static_cast<uint32_t>(row) >= extent) {
            info[e] = mfr::kNotRouted;
            continue;
        }
        const uint32_t ci = n_compact++;
        info[e] = (static_cast<uint32_t>(row) << mfr::kRowShift) | ci;
        coffs[ci] = offsets[e];
    }

    // ---- Phase 1c: this core's picks per expert. ----
    volatile tt_l1_ptr uint32_t* hist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_scan);
    for (uint32_t i = 0; i < n_compact; i++) {
        hist[i] = 0;
    }
    {
        uint32_t rec = l1_indices;
        for (uint32_t t = 0; t < tok_count; t++, rec += rec_stride) {
            volatile tt_l1_ptr uint16_t* idx = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec);
            for (uint32_t k = 0; k < topk; k++) {
                const uint32_t e = idx[k];
                // A padded token's sentinel id lands here, as does anything past the table.
                if (e >= num_routed_experts) {
                    continue;
                }
                const uint32_t w = info[e];
                if (w == mfr::kNotRouted) {
                    continue;
                }
                hist[w & mfr::kCompactMask]++;
            }
        }
    }

    // ---- Phase 2: Hillis-Steele scan across the cores. ----
    //
    // Round r adds the vector core i - 2^r produced in round r - 1, so after `rounds` rounds every
    // core holds the inclusive total over cores 0..i. Each round writes a fresh vector rather than
    // overwriting the one a peer is about to read: a core may be a full round ahead of its consumer,
    // and a ping-pong pair would hand that consumer half of two different rounds.
    //
    // One semaphore per round, because a single counting semaphore cannot say WHICH producer arrived
    // and they do not arrive in round order.
    const auto scan_page = [&](uint32_t r) { return l1_scan + r * scan_stride; };
    const auto publish = [&](uint32_t r) {
        const uint32_t tx = get_arg_val<uint32_t>(mfr::kScanBase + r * mfr::kScanWordsPerRound + 2u);
        if (tx == mfr::kNoCore) {
            return;
        }
        // Drain the stores into round r's vector before the consumer is told to NoC-read it: a
        // baby-RISCV store can retire before its write lands, and the semaphore increment is an
        // MMIO/NoC store that can race ahead of an L1 write.
        //
        // That is an ORDERING requirement and the blocking load meets it on its own. Waiting for the
        // increment to be acknowledged as well is a round trip per round on the critical path and buys
        // nothing here: the consumer spins until it lands. The kernel still barriers once before it
        // returns, which is where an unacknowledged atomic would actually be a problem -- it could
        // otherwise arrive after the next program has taken the core.
        volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scan_page(r));
        (void)ckernel::load_blocking(page + (n_compact > 0 ? n_compact - 1u : 0u));
        Semaphore<> sem(r);
        sem.up(noc, tx, get_arg_val<uint32_t>(mfr::kScanBase + r * mfr::kScanWordsPerRound + 3u), 1);
    };

    publish(0);
    for (uint32_t r = 0; r < rounds; r++) {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scan_page(r));
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scan_page(r + 1u));
        const uint32_t sx = get_arg_val<uint32_t>(mfr::kScanBase + r * mfr::kScanWordsPerRound);
        if (sx == mfr::kNoCore) {
            for (uint32_t i = 0; i < n_compact; i++) {
                dst[i] = src[i];
            }
        } else {
            Semaphore<> sem(r);
            sem.wait_min(1);
            // The peer's round-r vector lands where this core's round-(r+1) vector goes, then the two
            // are added in place; the partner's L1 address is this core's because the carve matches.
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(scan_page(r + 1u)),
                scan_stride,
                {.noc_x = sx,
                 .noc_y = get_arg_val<uint32_t>(mfr::kScanBase + r * mfr::kScanWordsPerRound + 1u),
                 .addr = scan_page(r)},
                {});
            noc.async_read_barrier();
            for (uint32_t i = 0; i < n_compact; i++) {
                dst[i] += src[i];
            }
        }
        publish(r + 1u);
    }

    // The allocator starts where this chip's run starts, plus every pick an EARLIER core made -- which
    // is the predecessor's inclusive total, not this core's.
    volatile tt_l1_ptr uint32_t* alloc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_alloc);
    {
        const uint32_t sx = get_arg_val<uint32_t>(mfr::kScanBase + rounds * mfr::kScanWordsPerRound);
        if (sx == mfr::kNoCore) {
            for (uint32_t i = 0; i < n_compact; i++) {
                alloc[i] = coffs[i];
            }
        } else {
            Semaphore<> sem(rounds);
            sem.wait_min(1);
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(scan_page(rounds + 1u)),
                scan_stride,
                {.noc_x = sx,
                 .noc_y = get_arg_val<uint32_t>(mfr::kScanBase + rounds * mfr::kScanWordsPerRound + 1u),
                 .addr = scan_page(rounds)},
                {});
            noc.async_read_barrier();
            volatile tt_l1_ptr uint32_t* before =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scan_page(rounds + 1u));
            for (uint32_t i = 0; i < n_compact; i++) {
                alloc[i] = coffs[i] + before[i];
            }
        }
    }

    // ---- Phase 3: re-walk the tokens and reduce each to its farthest surviving hop per direction. ----
    //
    // Counting the farthest-hop CLASSES and turning them into reach at the end costs one pass over
    // hops rather than one per token, and the suffix sum is the definition of reach.
    uint32_t cls[2][hops];
    for (uint32_t d = 0; d < 2u; d++) {
        for (uint32_t h = 0; h < hops; h++) {
            cls[d][h] = 0;
        }
    }
    {
        uint32_t rec = l1_indices;
        for (uint32_t t = 0; t < tok_count; t++, rec += rec_stride) {
            volatile tt_l1_ptr uint16_t* idx = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec);
            uint32_t far_cw = 0;
            uint32_t far_ccw = 0;
            for (uint32_t k = 0; k < topk; k++) {
                const uint32_t e = idx[k];
                if (e >= num_routed_experts) {
                    continue;
                }
                const uint32_t w = info[e];
                if (w == mfr::kNotRouted) {
                    continue;
                }
                const uint32_t ci = w & mfr::kCompactMask;
                const uint32_t page = alloc[ci];
                // The counter advances whether or not the page fits, because every later pick of this
                // expert is placed relative to it.
                alloc[ci] = page + 1u;
                if (page >= capacity) {
                    continue;
                }
                const uint32_t row = w >> mfr::kRowShift;
                if (row == my_row) {
                    continue;  // delivered here; it crosses no cable
                }
                // Both rows are below extent, so the wrap is a conditional subtract rather than a
                // modulo, which on this RISC is a called division.
                uint32_t cw = row + extent - my_row;
                if (cw >= extent) {
                    cw -= extent;
                }
                const uint32_t ccw = extent - cw;
                // A tie at exactly half the ring goes clockwise, matching mc_dir_of in
                // dispatch_fabric2d's reader. Resolving it the other way here would make a chunk's
                // length disagree with what the origin sends, which strands the axis.
                if (cw <= ccw) {
                    if (cw > far_cw) {
                        far_cw = cw;
                    }
                } else {
                    if (ccw > far_ccw) {
                        far_ccw = ccw;
                    }
                }
            }
            // A token with no surviving destination this way is not in the list at all.
            if (far_cw > 0) {
                cls[0][far_cw]++;
            }
            if (far_ccw > 0) {
                cls[1][far_ccw]++;
            }
        }
    }

    for (uint32_t d = 0; d < 2u; d++) {
        volatile tt_l1_ptr uint32_t* row_p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_out + d * out_page_bytes);
        // Zeroing the whole page covers hop 0, the terminator at max_hop + 1 and the page's alignment
        // padding, all of which are summed and written out verbatim.
        for (uint32_t i = 0; i < out_words; i++) {
            row_p[i] = 0;
        }
        uint32_t run = 0;
        for (uint32_t h = max_hop; h >= 1u; h--) {
            run += cls[d][h];
            row_p[h] = run;
        }
    }

    // ---- Phase 4: sum the rows up the tree; the root writes them out. ----
    Semaphore<> gather_sem(gather_sem_id);
    const uint32_t num_children = get_arg_val<uint32_t>(mfr::RtArg::kNumChildren);
    // One counter cannot say which child arrived, so wait for all of them before reading any.
    gather_sem.wait_min(num_children);
    volatile tt_l1_ptr uint32_t* local_rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_out);
    // All the children are already final, so their reads go out together and cost one round trip
    // between them instead of one each. That matters at the root, which has a child per tree level and
    // sits at the end of the whole reduction's critical path.
    const uint32_t child_slot = 2u * out_page_bytes;
    for (uint32_t c = 0; c < num_children; c++) {
        noc.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(l1_gather + c * child_slot),
            child_slot,
            {.noc_x = get_arg_val<uint32_t>(mfr::RtArg::kChildrenBase + c * 2u),
             .noc_y = get_arg_val<uint32_t>(mfr::RtArg::kChildrenBase + c * 2u + 1u),
             .addr = l1_out},
            {});
    }
    if (num_children > 0) {
        noc.async_read_barrier();
        for (uint32_t c = 0; c < num_children; c++) {
            volatile tt_l1_ptr uint32_t* child_rows =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_gather + c * child_slot);
            for (uint32_t i = 0; i < 2u * out_words; i++) {
                local_rows[i] += child_rows[i];
            }
        }
    }

    const uint32_t parent_x = get_arg_val<uint32_t>(mfr::RtArg::kParentNocX);
    if (parent_x != mfr::kNoCore) {
        // Same ordering hazard as the scan's publish: drain the tree-add stores before the parent is
        // told it may read them.
        (void)ckernel::load_blocking(local_rows + (2u * out_words - 1u));
        gather_sem.up(noc, parent_x, get_arg_val<uint32_t>(mfr::RtArg::kParentNocY), 1);
    } else {
        for (uint32_t d = 0; d < 2u; d++) {
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_out + d * out_page_bytes), out_acc, out_page_bytes, {}, {.page_id = d});
        }
        noc.async_write_barrier();
    }
    // The scan's publishes and the signal above are posted without waiting for an acknowledgement. This
    // is where that has to be settled: an atomic still in flight when the core is handed to the next
    // program would land on whatever that program put at the same semaphore.
    noc.async_atomic_barrier();
}
