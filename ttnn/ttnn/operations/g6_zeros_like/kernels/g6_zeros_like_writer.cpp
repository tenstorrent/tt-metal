// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_zeros_like writer: the op's only kernel (BRISC / NoC1). It never touches the input tensor.
//
// Each core zero-fills its L1 scratch (zero_scratch_bytes, at least one page) from the firmware
// MEM_ZEROS region, then streams it to every output page of its contiguous range
// [start_page, start_page + num_pages_this_core), then drains the writes.
//
// The fill is issued and awaited separately (the same two calls dataflow_kernel_lib::zero_tile makes)
// so the runtime-arg reads and the output accessor setup run in the shadow of the loopback reads,
// which sit on every core's critical path before its first page write.
//
// store_zero_block uses the raw Noc::async_write_zeros DRAM overload: no kernel_lib helper writes a
// constant L1 page to a range of interleaved DRAM pages.
//
// cb_zero_page is a private scratch: no credit ever moves on it, so its read pointer stays equal to
// the write pointer the fill targets, which is where the DRAM overload reads its source.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

struct PageRange {
    uint32_t start_page;
    uint32_t num_pages;
};

// zero_page_fill, with the core's setup (runtime args, output accessor) issued between the loopback
// reads and their barrier so it runs in their shadow. One zone: the setup belongs to the fill phase.
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
    range.start_page = get_arg_val<uint32_t>(0);
    range.num_pages = get_arg_val<uint32_t>(1);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0), page_size);
    noc.write_zeros_l1_barrier();
    return output;
}

template <typename Accessor>
FORCE_INLINE void store_zero_block(
    const Noc& noc,
    const Accessor& output,
    const CircularBuffer& cb_zero_page,
    uint32_t page_size,
    const PageRange& range) {
    DeviceZoneScopedN("g6zl-issue-writes");
    const uint32_t end_page = range.start_page + range.num_pages;
    for (uint32_t page = range.start_page; page < end_page; ++page) {
        noc.async_write_zeros(output, page_size, {.page_id = page}, cb_zero_page);
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
    constexpr auto output_args = TensorAccessorArgs<3>();
    // The DRAM zero overload streams min(page_size, NOC_MAX_BURST_SIZE) bytes from the scratch per burst.
    static_assert(zero_scratch_bytes >= page_size);

    Noc noc;
    CircularBuffer cb_zero_page(cb_zero_page_id);
    ASSERT(cb_zero_page.get_tile_size() == page_size);

    PageRange range;
    const auto output = zero_page_fill_and_setup(noc, cb_zero_page, output_args, page_size, zero_scratch_bytes, range);

    // The block is the core's whole page range: the zero source is constant in its extent.
    store_zero_block(noc, output, cb_zero_page, page_size, range);
    drain_writes(noc);
}
