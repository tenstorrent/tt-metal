// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// trid_double_issue reader (NCRISC / NoC0) — the kernel under study.
//
// Streams this core's contiguous range of interleaved DRAM pages into cb_in in
// blocks of `block` pages. The ONLY thing that differs between the two variants
// is WHICH BARRIER retires a block; the read-issue path is character-for-character
// the same call (`noc_async_read`) in both.
//
//   num_trids == 0  -> FULL BARRIER (baseline). Issue a block, then
//                      noc_async_read_barrier(), which waits for EVERY outstanding
//                      read on this NoC. The moment it returns, zero requests are
//                      in flight — the next block's reads are not issued until
//                      after the wait, so the NoC drains once per block and the
//                      DRAM round trip is fully exposed, once per block.
//
//   num_trids >= 2  -> TRID DOUBLE-ISSUE. Tag each block with a transaction id
//                      (noc_async_read_set_trid) and retire it with
//                      noc_async_read_barrier_with_trid(previous_id), which waits
//                      ONLY for that id. Blocks are issued `num_trids` deep before
//                      the first one is awaited, so while the reader is blocked on
//                      block k, blocks k+1 .. k+num_trids-1 are already in flight.
//                      The NoC never drains.
//
// The tag/wait pair comes from the shared kernel helper library
// (dataflow_kernel_lib::set_read_trid / ::async_read_barrier_with_trid) rather
// than the raw NoC calls: same two primitives on this RISC-V's default NoC, plus
// the watcher's transaction-id sanitizer under --dev.
//
// WHY A PLAIN noc_async_read CAN CARRY A TRID: the transaction id lives in the
// read command buffer's NOC_PACKET_TAG register. noc_async_read_set_trid writes
// that register and nothing else; an ordinary noc_async_read writes the address,
// length and control registers and never touches the tag. So the tag SET ONCE
// applies to every subsequent read on that command buffer until it is changed.
// That is what keeps this example on ordinary INTERLEAVED DRAM: the tagged reads
// still carry a full 64-bit NoC address, so consecutive pages may live in
// different DRAM banks.
//
// THE RING NEEDS NO COUNTERS. Block k carries id `(k % num_trids) + 1`, and so did
// block k-num_trids — so the id we are about to tag with is exactly the one whose
// slot must be freed first. That makes one barrier do both jobs (retire the old
// block, free the id), and every quantity below falls out of the loop index k:
// no issue/wait cursors and no in-flight counter to keep in step.
//
// SLOT ARITHMETIC (the part that is easy to get wrong): cb_in holds `cb_blocks`
// block-sized slots and is the SAME size in both variants. A circular buffer's
// write pointer only advances on cb_push_back, so while `in_flight` blocks are
// issued-but-not-yet-pushed the CB pointer lags them: the landing address for the
// new block is `get_write_ptr() + in_flight` slots, wrapped by hand at the end of
// the CB region. Reserving `(in_flight + 1)` blocks is what makes that slot safe.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr uint32_t num_trids = get_compile_time_arg_val(3);  // 0 = full-barrier baseline
    constexpr uint32_t cb_blocks = get_compile_time_arg_val(4);  // CB depth in blocks (same for both variants)
    constexpr uint32_t ahead = get_compile_time_arg_val(5);      // baseline: blocks issued per barrier
    constexpr auto in_args = TensorAccessorArgs<6>();

    // Blocks kept outstanding. The two variants agree on this number and on all the
    // buffer machinery below; they differ ONLY in how a block is RETIRED, which is
    // what forces the two loop shapes:
    //   baseline  cannot name a block, so retiring means draining EVERYTHING. That is
    //             affordable only once per `depth` blocks, so it fills, drains to zero,
    //             and refills -- a batch loop.
    //   trid      names the oldest block by id, so it retires one block per iteration
    //             while the rest stay on the wire -- a sliding window.
    // That is the whole experiment, so the loops are deliberately not merged. Everything
    // they share -- ring addressing, issuing a block, the sub-block tail -- is factored
    // into the three lambdas below and written once.
    constexpr uint32_t depth = (num_trids == 0) ? ahead : num_trids;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);
    const Noc noc;  // this RISC-V's default NoC — the one plain noc_async_read issues on

    const uint32_t cb_bytes = cb_blocks * block * page_bytes;
    const uint32_t cb_base = get_write_ptr(cb_in);  // before any push -> start of the CB region
    const uint32_t cb_end = cb_base + cb_bytes;
    const uint32_t full_blocks = num_pages / block;
    const uint32_t tail = num_pages - full_blocks * block;

    // Ring slot `j` blocks past `from`, wrapped once at the end of the CB region. One
    // wrap always suffices because no caller looks more than cb_blocks-1 slots ahead.
    auto slot = [=](uint32_t from, uint32_t j) {
        const uint32_t a = from + j * block * page_bytes;
        return a >= cb_end ? a - cb_bytes : a;
    };
    // Issue `n` page reads of the run starting at page `first`, landing at `addr`.
    auto issue = [=](uint32_t first, uint32_t n, uint32_t addr) {
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(in_acc.get_noc_addr(start_page + first + i), addr + i * page_bytes, page_bytes);
        }
    };
    // The < block leftover, if any. Always the plain path: too small to pipeline, and it
    // leaves the CB pointer slot-aligned for the next kernel_iters pass.
    auto do_tail = [=]() {
        if (tail) {
            cb_reserve_back(cb_in, tail);
            issue(full_blocks * block, tail, get_write_ptr(cb_in));
            noc_async_read_barrier();
            cb_push_back(cb_in, tail);
        }
    };

    if constexpr (num_trids == 0) {
        // ---- baseline: fill `depth` blocks, ONE barrier, push them, repeat ----
        for (uint32_t it = 0; it < kernel_iters; ++it) {
            for (uint32_t k = 0; k < full_blocks;) {
                const uint32_t nb = (full_blocks - k) < depth ? (full_blocks - k) : depth;
                cb_reserve_back(cb_in, nb * block);
                const uint32_t base = get_write_ptr(cb_in);
                for (uint32_t j = 0; j < nb; ++j) {
                    issue((k + j) * block, block, slot(base, j));
                }
                noc_async_read_barrier();  // drains EVERY outstanding read -> NoC empty here
                for (uint32_t j = 0; j < nb; ++j) {
                    cb_push_back(cb_in, block);
                }
                k += nb;
            }
            do_tail();
        }
    } else {
        // ---- trid: sliding window; retire the oldest id, the rest stay in flight ----
        // No counters needed: block k and block k-depth share an id, so the id we are
        // about to tag with is exactly the one whose slot must be freed. One barrier
        // both retires the old block and makes the id reusable.
        for (uint32_t it = 0; it < kernel_iters; ++it) {
            for (uint32_t k = 0; k < full_blocks; ++k) {
                const uint32_t trid = (k % depth) + 1;
                if (k >= depth) {
                    dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);
                    cb_push_back(cb_in, block);
                }
                const uint32_t in_flight = (k < depth) ? k : (depth - 1);
                cb_reserve_back(cb_in, (in_flight + 1) * block);
                dataflow_kernel_lib::set_read_trid(noc, trid);  // tags every read below until changed
                issue(k * block, block, slot(get_write_ptr(cb_in), in_flight));
            }
            const uint32_t to_drain = (full_blocks < depth) ? full_blocks : depth;
            for (uint32_t d = 0; d < to_drain; ++d) {
                dataflow_kernel_lib::async_read_barrier_with_trid(noc, ((full_blocks - to_drain + d) % depth) + 1);
                cb_push_back(cb_in, block);
            }
            do_tail();
        }
        noc_async_read_barrier();                    // everything is already retired; belt and braces
        dataflow_kernel_lib::set_read_trid(noc, 0);  // restore untagged, as the firmware expects
    }
}
