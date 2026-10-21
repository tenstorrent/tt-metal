// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// N-way ROW_MAJOR width-concat reader.
//
// Each output stick is assembled directly from all original input sticks.  No
// pairwise intermediate is materialized, which avoids both extra dispatches and
// the shape-dependent corruption seen when an unaligned intermediate width is
// fed into another two-input concat on Blackhole.
//
// All inputs share one TensorAccessor ABI (same interleaved memory placement),
// but carry independent logical and aligned page sizes.
//
// Fast path (mirrors the proven two-input width reader): when an input's
// logical stick fills its whole physical page AND its packed destination
// offset is a legal NOC endpoint for the shared source transport, read the
// stick straight into the output CB at that offset.  Those reads are issued
// back to back and drained by ONE barrier per batch.  The nightly 4-input
// NHWC dim=-1 class is exactly this case; staging every stick through scratch
// with a barrier per read plus a RISC halfword copy of the full payload made
// it up to 20x slower than TTNN.  Unaligned inputs keep the scratch path.
//
// CT args:
//   cb_out, cb_scratch, N_INPUTS, OUT_PAGE_SIZE, BATCH, SRC_NOC_ALIGNMENT,
//   TensorAccessorArgs(shared input placement)
// RT args:
//   num_pages, start_page,
//   bases[N_INPUTS], stick_sizes[N_INPUTS], page_sizes[N_INPUTS]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ckernel.h"

// Copy `size` bytes from `src_addr` to `dst_addr` using 16-bit stores only, for
// a segment whose packed destination offset or size is odd (a 1-byte dtype with
// an odd width).  Every halfword the segment touches is written as a whole; the
// two edge halfwords are read-modify-write so the neighbouring byte survives.
// Returns the address of the final halfword store -- the exact address the drain
// must load, because ckernel::load_blocking has no 8-bit form and Wormhole
// matches a load to a store by start address alone.
//
// The neighbouring byte is never written concurrently, so the RMW cannot lose
// an update:
//   * Direct NOC reads start at a multiple of SRC_NOC_ALIGNMENT and span a whole
//     physical page (a multiple of the L1 alignment), so every direct segment
//     begins and ends on an even address.  An edge halfword therefore never
//     straddles a direct segment's boundary.
//   * Every other neighbour is a staged segment (or page padding) written by this
//     RISC in program order, and the next page's direct read starts on an
//     OUT_PAGE_SIZE boundary, past this page's padding.
// The head RMW of one segment loads the halfword its predecessor's tail RMW
// just stored, at the identical start address, so Wormhole's start-address
// conflict check orders it after that store.
FORCE_INLINE uint32_t copy_bytes_halfword_rmw(uint32_t src_addr, uint32_t dst_addr, uint32_t size) {
    CoreLocalMem<volatile uint8_t> src(src_addr);
    const uint32_t end = dst_addr + size;
    const uint32_t last = (end - 1) & ~1U;
    for (uint32_t half = dst_addr & ~1U; half <= last; half += 2) {
        CoreLocalMem<volatile uint16_t> target(half);
        const bool lo_in = half >= dst_addr;
        const bool hi_in = half + 1 < end;
        uint16_t value = 0;
        if (!(lo_in && hi_in)) {
            value = target[0];
        }
        if (lo_in) {
            value = (value & 0xFF00U) | src[half - dst_addr];
        }
        if (hi_in) {
            value = (value & 0x00FFU) | (static_cast<uint16_t>(src[half + 1 - dst_addr]) << 8);
        }
        target[0] = value;
    }
    return last;
}

void kernel_main() {
    uint32_t num_pages = get_arg_val<uint32_t>(0);
    uint32_t src_page = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t N_INPUTS = get_compile_time_arg_val(2);
    constexpr uint32_t OUT_PAGE_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t BATCH = get_compile_time_arg_val(4);
    constexpr uint32_t SRC_NOC_ALIGNMENT = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr bool WORD_COPY = get_named_compile_time_arg_val("word_copy") != 0;
    constexpr bool DIRECT_READ = get_named_compile_time_arg_val("direct_read") != 0;

    constexpr uint32_t base_rt = 2;
    constexpr uint32_t stick_rt = base_rt + N_INPUTS;
    constexpr uint32_t page_rt = stick_rt + N_INPUTS;

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer scratch_cb(cb_scratch);

    scratch_cb.reserve_back(1);
    const uint32_t scratch_addr = scratch_cb.get_write_ptr();

    while (num_pages > 0) {
        const uint32_t batch = num_pages < BATCH ? num_pages : BATCH;
        out_cb.reserve_back(batch);
        const uint32_t out_base = out_cb.get_write_ptr();

        // Address and width of the batch's final RISC store, 0 if it made none.
        uint32_t drain_addr = 0;
        bool drain_wide = false;

        for (uint32_t page = 0; page < batch; ++page) {
            uint32_t dst = out_base + page * OUT_PAGE_SIZE;
            uint32_t dst_offset = 0;

            for (uint32_t input = 0; input < N_INPUTS; ++input) {
                const uint32_t base = get_arg_val<uint32_t>(base_rt + input);
                const uint32_t stick_size = get_arg_val<uint32_t>(stick_rt + input);
                const uint32_t page_size = get_arg_val<uint32_t>(page_rt + input);
                const auto accessor = TensorAccessor(src_args, base, page_size);

                // Direct path: the stick is its whole physical page and its
                // packed destination offset (relative to the CB write pointer,
                // whose pages the writer already addresses at transport
                // alignment) is a legal endpoint for the source transport.
                // Issue the read into the output CB and defer completion to
                // the single per-batch barrier below.
                if constexpr (DIRECT_READ) {
                    const uint32_t out_cb_offset = page * OUT_PAGE_SIZE + dst_offset;
                    if (stick_size == page_size && (out_cb_offset % SRC_NOC_ALIGNMENT) == 0) {
                        noc.async_read(
                            accessor,
                            out_cb,
                            stick_size,
                            {.page_id = src_page + page},
                            {.offset_bytes = out_cb_offset});
                        dst_offset += stick_size;
                        continue;
                    }
                }

                // Physical pages are always NOC-aligned.  Read one complete
                // source page, then copy only its logical payload into the
                // packed output stick.  2- and 4-byte dtypes give even sticks
                // at even offsets and take the word/halfword copies; a 1-byte
                // dtype can make a size or an offset odd, and that segment goes
                // through copy_bytes_halfword_rmw.  Either way the final store
                // is a 16- or 32-bit one, which the drain below can name.
                noc.async_read(accessor, scratch_cb, page_size, {.page_id = src_page + page}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if (stick_size == 0) {
                    continue;
                }
                if (((dst_offset | stick_size) & 1U) != 0) {
                    drain_addr = copy_bytes_halfword_rmw(scratch_addr, dst + dst_offset, stick_size);
                    drain_wide = false;
                    dst_offset += stick_size;
                    continue;
                }
                bool word_store = false;
                if constexpr (WORD_COPY) {
                    word_store = (dst_offset & 3U) == 0;
                }
                if (word_store) {
                    CoreLocalMem<volatile uint32_t> src(scratch_addr);
                    CoreLocalMem<volatile uint32_t> target(dst + dst_offset);
                    const uint32_t words = stick_size / 4;
                    for (uint32_t word = 0; word < words; ++word) {
                        target[word] = src[word];
                    }
                    if ((stick_size & 2U) != 0) {
                        CoreLocalMem<volatile uint16_t> src_tail(scratch_addr);
                        CoreLocalMem<volatile uint16_t> target_tail(dst + dst_offset);
                        target_tail[words * 2] = src_tail[words * 2];
                    }
                } else {
                    CoreLocalMem<volatile uint16_t> src(scratch_addr);
                    CoreLocalMem<volatile uint16_t> target(dst + dst_offset);
                    for (uint32_t word = 0; word < stick_size / 2; ++word) {
                        target[word] = src[word];
                    }
                }
                // The last store of this copy: a halfword at the end of the
                // stick unless a word store ended it exactly.
                const bool ends_on_word = word_store && (stick_size & 2U) == 0;
                drain_wide = ends_on_word;
                drain_addr = dst + dst_offset + stick_size - (ends_on_word ? sizeof(uint32_t) : sizeof(uint16_t));
                dst_offset += stick_size;
            }
        }

        // Drain every deferred direct read for this batch.  Scratch-staged
        // inputs already completed at their own per-read barriers; for an
        // all-scratch batch this is a cheap counter check.
        if constexpr (DIRECT_READ) {
            noc.async_read_barrier();
        }
        // A RISC store can retire before its write-request reaches L1, and the
        // NoC and the CB consumer are separate L1 clients with no program order
        // against this core.  push_back only bumps a stream register -- a
        // different memory region -- so neither it nor a fence orders the copy
        // against it.  Read the last store back instead: the read response
        // proves L1 processed that write, and the store queue is FIFO into one
        // region, so every earlier store landed with it.  The load must start at
        // exactly the last store's address -- Wormhole detects store/load
        // conflicts on the starting byte address alone, so a load of the
        // containing word does not order a halfword stored partway into it.
        if (drain_addr != 0) {
            if (drain_wide) {
                (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drain_addr));
            } else {
                (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(drain_addr));
            }
        }
        out_cb.push_back(batch);
        src_page += batch;
        num_pages -= batch;
    }
}
