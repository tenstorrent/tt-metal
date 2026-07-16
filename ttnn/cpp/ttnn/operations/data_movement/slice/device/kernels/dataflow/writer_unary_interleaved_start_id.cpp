// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Slice-specific writer using named compile-time args for DFB index.
// Based on eltwise/unary writer but uses get_named_compile_time_arg_val("dfb_id_out")
// so the DFB index can be remapped by the fusion infrastructure.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t dfb_id_out = get_named_compile_time_arg_val("dfb_id_out");
    constexpr auto dst_args = TensorAccessorArgs<0>();

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_out(dfb_id_out);

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb_out.get_entry_size();
    Noc noc;

#ifdef OUT_SHARDED
    dfb_out.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_out.wait_front(onepage);
        noc.async_write(dfb_out, s, page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_out.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
