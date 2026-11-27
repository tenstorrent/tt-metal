// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_zeros_like writer: the op's only kernel (BRISC / NoC1). It never touches the input tensor.
//
// Each core streams zeros to the output pages it owns, then drains the writes. What a core owns
// depends on the host regime (CT arg bank_run_stride):
//   0: a contiguous page-id range [start_page, start_page + num_pages); consecutive ids sit in different
//      banks, so each page is its own write.
//   B: an in-bank run start_page + k * B, k < num_pages, B = the bank count; those pages are contiguous
//      bytes in one bank, written in zero_scratch_bytes chunks from the same scratch.
//
// Two zero sources (CT arg head_pages picks the split):
//   - the firmware MEM_ZEROS region (MEM_ZEROS_SIZE bytes, zero in every core's L1): the first head_pages
//     pages of the range are written straight from it, one MEM_ZEROS_SIZE write per chunk. Nothing has to
//     be prepared, so the core's first DRAM write leaves right after its setup and every bank starts
//     draining at kernel start instead of after a fill.
//   - the L1 scratch cb_zero_page (zero_scratch_bytes, at least one page), zero-filled by NoC loopback
//     from MEM_ZEROS: the rest of the range, one page (or chunk) per write. The fill runs only when the
//     head does not cover the range. With head_pages = 0 it is issued first and the runtime-arg reads
//     and the accessor setup run in the shadow of its loopback reads; otherwise it is issued after the
//     head writes, whose DRAM drain hides it.
//
// store_zero_block uses the raw Noc::async_write_zeros DRAM overload (one page per call), store_zero_bank_run
// and store_zero_head raw Noc::async_write(_with_state): no kernel_lib helper writes a constant L1 source
// to interleaved DRAM, and the zeros overload refuses a write that crosses a page.
//
// cb_zero_page is a private scratch: no credit ever moves on it, so its read pointer stays equal to
// the write pointer the fill targets, which is where the DRAM overload reads its source.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

struct PageRange {
    uint32_t start_page;
    uint32_t num_pages;
};

template <typename AccessorArgs>
FORCE_INLINE auto setup(const AccessorArgs& output_args, uint32_t page_size, PageRange& range) {
    range.start_page = get_arg_val<uint32_t>(0);
    range.num_pages = get_arg_val<uint32_t>(1);
    return TensorAccessor(output_args, get_common_arg_val<uint32_t>(0), page_size);
}

// zero_page_fill, with the core's setup issued between the loopback reads and their barrier so it runs
// in their shadow. One zone: the setup belongs to the fill phase.
template <typename AccessorArgs>
FORCE_INLINE auto zero_page_fill_and_setup(
    const Noc& noc,
    const CircularBuffer& cb_zero_page,
    const AccessorArgs& output_args,
    uint32_t page_size,
    uint32_t zero_scratch_bytes,
    PageRange& range) {
    DeviceZoneScopedN("g6zl-zero-page");
    noc.async_write_zeros(cb_zero_page, zero_scratch_bytes);
    const auto output = setup(output_args, page_size, range);
    noc.write_zeros_l1_barrier();
    return output;
}

FORCE_INLINE void zero_page_fill(const Noc& noc, const CircularBuffer& cb_zero_page, uint32_t zero_scratch_bytes) {
    DeviceZoneScopedN("g6zl-zero-page");
    noc.async_write_zeros(cb_zero_page, zero_scratch_bytes);
    noc.write_zeros_l1_barrier();
}

// [dst, dst + bytes) from MEM_ZEROS: full chunks share one write state (only the addresses change per
// chunk), a ragged remainder is one plain write.
FORCE_INLINE void write_zeros_from_mem_zeros(const Noc& noc, uint64_t dst, uint32_t bytes) {
    const CoreLocalMem<uint32_t> mem_zeros(static_cast<uintptr_t>(MEM_ZEROS_BASE));
    const PrecomposedUnicastEndpoint dram;
    constexpr uint32_t chunk = MEM_ZEROS_SIZE;
    const uint32_t full_bytes = bytes - bytes % chunk;
    if (full_bytes != 0) {
        noc.set_async_write_state<NocOptions::DEFAULT, chunk>(dram, chunk, {.noc_addr = dst});
        for (uint32_t offset = 0; offset < full_bytes; offset += chunk) {
            noc.async_write_with_state<NocOptions::DEFAULT, chunk>(
                mem_zeros, dram, chunk, {}, {.noc_addr = dst + offset});
        }
    }
    if (full_bytes != bytes) {
        noc.async_write(mem_zeros, dram, bytes - full_bytes, {}, {.noc_addr = dst + full_bytes});
    }
}

// The first num_head_pages pages of the range. A bank run's pages are one contiguous byte range.
template <uint32_t bank_run_stride, typename Accessor>
FORCE_INLINE void store_zero_head(
    const Noc& noc, const Accessor& output, uint32_t page_size, const PageRange& range, uint32_t num_head_pages) {
    DeviceZoneScopedN("g6zl-zero-head");
    if constexpr (bank_run_stride != 0) {
        write_zeros_from_mem_zeros(
            noc, output.get_noc_addr(range.start_page, 0, noc.get_noc_id()), num_head_pages * page_size);
    } else {
        for (uint32_t k = 0; k < num_head_pages; ++k) {
            write_zeros_from_mem_zeros(noc, output.get_noc_addr(range.start_page + k, 0, noc.get_noc_id()), page_size);
        }
    }
}

// Pages [range.start_page + first_page, range end), one write each.
template <typename Accessor>
FORCE_INLINE void store_zero_block(
    const Noc& noc,
    const Accessor& output,
    const CircularBuffer& cb_zero_page,
    uint32_t page_size,
    const PageRange& range,
    uint32_t first_page) {
    DeviceZoneScopedN("g6zl-issue-writes");
    const uint32_t end_page = range.start_page + range.num_pages;
    for (uint32_t page = range.start_page + first_page; page < end_page; ++page) {
        noc.async_write_zeros(output, page_size, {.page_id = page}, cb_zero_page);
    }
}

// The run's pages are contiguous bytes from start_page's address on (interleaved DRAM: every
// bank_run_stride-th page id of a bank follows the previous one), so the run from its first_page-th page
// on is one byte range written in chunks of at most the scratch size, each sourced from the scratch's start.
template <uint32_t zero_scratch_bytes, uint32_t bank_run_stride, typename Accessor>
FORCE_INLINE void store_zero_bank_run(
    const Noc& noc,
    const Accessor& output,
    const CircularBuffer& cb_zero_page,
    uint32_t page_size,
    const PageRange& range,
    uint32_t first_page) {
    DeviceZoneScopedN("g6zl-issue-writes");
    // The host plan's bank count must be the accessor's in-bank page stride.
    ASSERT(output.contiguous_page_stride() == bank_run_stride);
    ASSERT(
        output.num_contiguous_pages(range.start_page, range.start_page + (range.num_pages - 1) * bank_run_stride + 1) ==
        range.num_pages);
    const uint32_t run_bytes = range.num_pages * page_size;
    for (uint32_t offset = first_page * page_size; offset < run_bytes; offset += zero_scratch_bytes) {
        const uint32_t remaining = run_bytes - offset;
        const uint32_t chunk = remaining < zero_scratch_bytes ? remaining : zero_scratch_bytes;
        noc.async_write(cb_zero_page, output, chunk, {}, {.page_id = range.start_page, .offset_bytes = offset});
    }
}

FORCE_INLINE void drain_writes(const Noc& noc) {
    DeviceZoneScopedN("g6zl-write-barrier");
    noc.write_zeros_dram_barrier();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_zero_page_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t zero_scratch_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t bank_run_stride = get_compile_time_arg_val(3);
    constexpr uint32_t head_pages = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();
    // The DRAM zero overload streams min(page_size, NOC_MAX_BURST_SIZE) bytes from the scratch per burst.
    static_assert(zero_scratch_bytes >= page_size);
    // A bank-run chunk ends on a page boundary, so a ragged last chunk is still whole pages.
    static_assert(zero_scratch_bytes % page_size == 0);

    Noc noc;
    CircularBuffer cb_zero_page(cb_zero_page_id);
    ASSERT(cb_zero_page.get_tile_size() == page_size);

    PageRange range;
    uint32_t first_scratch_page = 0;
    const auto output = [&] {
        if constexpr (head_pages == 0) {
            return zero_page_fill_and_setup(noc, cb_zero_page, output_args, page_size, zero_scratch_bytes, range);
        } else {
            const auto accessor = setup(output_args, page_size, range);
            first_scratch_page = range.num_pages < head_pages ? range.num_pages : head_pages;
            store_zero_head<bank_run_stride>(noc, accessor, page_size, range, first_scratch_page);
            if (first_scratch_page < range.num_pages) {
                zero_page_fill(noc, cb_zero_page, zero_scratch_bytes);
            }
            return accessor;
        }
    }();

    // The block is the core's whole page range or bank run: the zero source is constant in its extent.
    if constexpr (bank_run_stride != 0) {
        store_zero_bank_run<zero_scratch_bytes, bank_run_stride>(
            noc, output, cb_zero_page, page_size, range, first_scratch_page);
    } else {
        store_zero_block(noc, output, cb_zero_page, page_size, range, first_scratch_page);
    }
    drain_writes(noc);
}
