// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader — NCRISC / NoC0.
//
// One block = one output tile-row x `w` output tile-columns. Blocks are
// linearized `b = wchunk * nt_h + r`; this core owns the contiguous range
// [b0, b0 + nb). Per block the reader issues `tile_h` NoC reads of
// `w * tile_row_bytes` bytes at L1 stride `w * tile_row_bytes` and ONE barrier,
// then pushes `w` tile-sized pages — exactly the contract
// `dataflow_kernel_lib::read_sticks_for_tilize<TILE>` implements, which is why
// the production path is that helper call and nothing else.
//
// `resident == 1` (Refinement 2, A3/C14) is the SECOND production path and the
// only one that is not a NoC read at all: the input shard already sits in this
// core's L1 and `cb_input_sticks` is aliased onto it, so the reader just arms
// the CB. That is what implementing sharding means — re-reading a local shard
// through the TensorAccessor above would re-fetch bytes the core already holds.
//
// The `if constexpr` arms below are LEVER COUNTERFACTUALS, not alternative
// production paths: `barrier_per_block == 0` is the master.md B7 off-arm (one
// barrier per transaction instead of per block) and `stub_read == 1` is the
// /perf-measure read ablation (keep the CB sync scaffolding, drop the NoC
// payload). At their defaults (1, 0) the compiler emits only the helper call, so
// they cannot perturb the measured path.
//
// `n_bands > 1` (Refinement 4, A3c) is the THIRD production path: the source is
// sharded narrower than a row, so its pages are BANDS of a row scattered across
// cores, and this core (which owns the OUTPUT block) pulls the bands it needs
// over L1->L1. See the arm's own comment for the topology.
//
// HELPER SUBSTITUTION, declared (Refinement 4): the gather arm cannot be
// `read_sticks_for_tilize` either. That helper reads ONE page per stick at a
// fixed `byte_offset_within_page` (`tilize_helpers_dataflow.inl`), which is the
// whole-row-page contract; a banded source needs the stick's bytes assembled
// from SEVERAL pages at different offsets, into one contiguous L1 row. There is
// no kernel_lib entry point for a segmented stick read, so `noc_async_read` +
// `TensorAccessor::get_noc_addr(page, offset)` is the only mechanism — and the
// helper remains the emitted code on the whole-row path (`n_bands == 1`), which
// is every pre-R4 shape, byte for byte.
//
// HELPER SUBSTITUTION, declared (Refinement 3): `stagger_reads == 1` issues the
// same `tile_h` reads to the same L1 destinations in a ROTATED order, and
// `read_sticks_for_tilize` cannot express that — it walks `start_page ..
// start_page + total_num_rows` sequentially into consecutive L1, so the only way
// to reorder the issue is to split it into two calls, which would write the
// sticks into the CB in rotated order and produce a row-permuted tile. The
// substitution is therefore the rotation itself, not a preference; it reuses the
// arm that already existed for B7, and the helper remains the emitted code
// whenever the lever is off (the `stagger_reads == 0` condition above).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t nt_h = get_compile_time_arg_val(1);             // tile-rows
    constexpr uint32_t n_wchunks = get_compile_time_arg_val(2);        // column-blocks per tile-row
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);           // sticks per tile-row
    constexpr uint32_t tile_row_bytes = get_compile_time_arg_val(4);   // 32 * elem
    constexpr uint32_t wt_block = get_compile_time_arg_val(5);         // the block-width knob
    constexpr uint32_t wt_tail = get_compile_time_arg_val(6);
    constexpr uint32_t barrier_per_block = get_compile_time_arg_val(7);  // lever B7 (1 = on)
    constexpr uint32_t stub_read = get_compile_time_arg_val(8);          // ablation (0 = off)
    constexpr uint32_t resident = get_compile_time_arg_val(9);           // A3/C14 zero-copy (1 = on)
    constexpr uint32_t stagger_reads = get_compile_time_arg_val(10);     // lever R3/A3 (1 = on)
    constexpr uint32_t n_bands = get_compile_time_arg_val(11);           // R4: source pages per row
    constexpr uint32_t band_bytes = get_compile_time_arg_val(12);        // R4: bytes per source page
    constexpr auto src_args = TensorAccessorArgs<13>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1);
    const uint32_t nb = get_arg_val<uint32_t>(2);

    // A3/C14 zero-copy: `cb_input_sticks` is ALIASED onto this core's own input
    // shard, so the block is already in L1 and there is nothing to fetch — the
    // reader exists only to arm the CB. NO NoC read, no TensorAccessor, and the
    // shard hands us the block width (`wt_block == Wt_shard`), so the whole
    // shard is `nb * wt_block` tile-sized pages.
    if constexpr (resident == 1) {
        const uint32_t pages = nb * wt_block;
        cb_reserve_back(cb_input_sticks, pages);
        cb_push_back(cb_input_sticks, pages);
        return;
    }

    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t i = 0; i < nb; ++i) {
        const uint32_t b = b0 + i;
        const uint32_t wchunk = b / nt_h;      // column-block index
        const uint32_t r = b - wchunk * nt_h;  // global tile-row index

        // The tail column-block is the last one; its width is WT_TAIL (== WT_BLOCK
        // when Wt divides evenly), so the reader's per-block page count matches
        // compute's `WT_BLOCK x n_full` then `WT_TAIL x n_tail` sequence exactly.
        const uint32_t w = (wchunk == n_wchunks - 1) ? wt_tail : wt_block;
        const uint32_t row_bytes = w * tile_row_bytes;
        const uint32_t byte_offset = wchunk * wt_block * tile_row_bytes;

        if constexpr (n_bands > 1) {
            // R4 (A3c) CROSS-SPEC RESHARD — the PULL gather.
            //
            // The source is sharded NARROWER than a row, so its pages are bands:
            // page `(row, band)` holds bytes [band*band_bytes, +band_bytes) of
            // that row and lives in whatever core's L1 owns that shard. This
            // core owns the OUTPUT block, so it pulls the bands its block needs
            // — the read is split at band boundaries and each segment is one
            // NoC read from whichever core holds it. Still ONE barrier per block
            // (B7), still no semaphore and no multicast: §1.1 makes the map a
            // bijection, so every source byte is read by exactly one core and
            // there is nothing to coordinate.
            //
            // `band_bytes` is a whole number of tile-columns (the host refuses
            // anything else), so every segment length is a multiple of
            // `tile_row_bytes` and the L1 cursor stays on the alignment grid.
            cb_reserve_back(cb_input_sticks, w);
            const uint32_t l1_base = get_write_ptr(cb_input_sticks);
            const uint32_t end = byte_offset + row_bytes;
            for (uint32_t s = 0; s < tile_h; ++s) {
                const uint32_t row_page0 = (r * tile_h + s) * n_bands;
                uint32_t l1 = l1_base + s * row_bytes;
                uint32_t off = byte_offset;
                while (off < end) {
                    const uint32_t band = off / band_bytes;
                    const uint32_t in_band = off - band * band_bytes;
                    uint32_t len = band_bytes - in_band;
                    if (len > end - off) {
                        len = end - off;
                    }
                    if constexpr (stub_read == 0) {
                        noc_async_read(src.get_noc_addr(row_page0 + band, in_band), l1, len);
                        if constexpr (barrier_per_block == 0) {
                            noc_async_read_barrier();  // B7 off: one barrier per transaction
                        }
                    }
                    l1 += len;
                    off += len;
                }
            }
            if constexpr (barrier_per_block == 1) {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_sticks, w);
        } else if constexpr (barrier_per_block == 1 && stub_read == 0 && stagger_reads == 0) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                src,
                /*total_num_rows*/ tile_h,
                /*row_bytes*/ row_bytes,
                /*start_page*/ r * tile_h,
                /*byte_offset_within_page*/ byte_offset);
        } else {
            // Counterfactual / ablation arm, and the `stagger_reads` production arm:
            // identical CB accounting (reserve w, read tile_h sticks at L1 stride
            // row_bytes, push w) — only the ISSUE ORDER of the reads can differ.
            cb_reserve_back(cb_input_sticks, w);
            const uint32_t l1_base = get_write_ptr(cb_input_sticks);
            // R3 read stagger. On a wide-short tensor (`nt_h == 1`) EVERY core reads
            // the same `tile_h` source pages, differing only in byte offset, and the
            // helper issues them in page order — so at any instant the whole fleet
            // is requesting one page, i.e. ONE DRAM bank, and the banks are used
            // one at a time. Rotating the issue order by this core's own block index
            // spreads the in-flight requests across the banks. Byte-for-byte the
            // same transfers to the same L1 destinations: `s` indexes both the
            // source page and its L1 slot, so the block lands stick-ordered no
            // matter which stick is issued first, and the single barrier below still
            // covers all of them.
            const uint32_t s0 = (stagger_reads == 1) ? (b % tile_h) : 0;
            for (uint32_t i = 0; i < tile_h; ++i) {
                uint32_t s = s0 + i;
                if (s >= tile_h) {
                    s -= tile_h;
                }
                if constexpr (stub_read == 0) {
                    noc_async_read(src.get_noc_addr(r * tile_h + s, byte_offset), l1_base + s * row_bytes, row_bytes);
                    if constexpr (barrier_per_block == 0) {
                        noc_async_read_barrier();  // B7 off: one barrier per transaction
                    }
                }
            }
            if constexpr (barrier_per_block == 1) {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_sticks, w);
        }
    }
}
