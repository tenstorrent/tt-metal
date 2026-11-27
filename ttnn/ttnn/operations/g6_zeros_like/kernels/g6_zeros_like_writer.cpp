// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_zeros_like writer: the op's only kernel (BRISC / NoC1). It never touches the input tensor.
//
// Each core zero-fills one page of L1 scratch from the firmware MEM_ZEROS region, then streams that
// page to every output page of its contiguous range [start_page, start_page + num_pages_this_core),
// then drains the writes.
//
// store_zero_block uses the raw Noc::async_write_zeros DRAM overload: no kernel_lib helper writes a
// constant L1 page to a range of interleaved DRAM pages.
//
// cb_zero_page is a private scratch: no credit ever moves on it, so its read pointer stays equal to
// the write pointer zero_tile fills, which is where the DRAM overload reads its source.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {

FORCE_INLINE void zero_page_fill(const CircularBuffer& cb_zero_page) {
    DeviceZoneScopedN("g6zl-zero-page");
    dataflow_kernel_lib::zero_tile(cb_zero_page);
}

template <typename Accessor>
FORCE_INLINE void store_zero_block(
    const Noc& noc,
    const Accessor& output,
    const CircularBuffer& cb_zero_page,
    uint32_t page_size,
    uint32_t first_page,
    uint32_t num_pages) {
    DeviceZoneScopedN("g6zl-issue-writes");
    const uint32_t end_page = first_page + num_pages;
    for (uint32_t page = first_page; page < end_page; ++page) {
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
    constexpr auto output_args = TensorAccessorArgs<2>();

    CircularBuffer cb_zero_page(cb_zero_page_id);
    ASSERT(cb_zero_page.get_tile_size() == page_size);
    zero_page_fill(cb_zero_page);

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);

    Noc noc;
    const auto output = TensorAccessor(output_args, output_addr, page_size);

    // The block is the core's whole page range: the zero source is constant in its extent.
    store_zero_block(noc, output, cb_zero_page, page_size, start_page, num_pages_this_core);
    drain_writes(noc);
}
