// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp. Identical dataflow logic; the output CB index
// becomes a dfb:: binding, the destination TensorAccessor a tensor:: binding (the dst_addr runtime arg is
// gone), the remaining runtime args become named, and the page size is read off the DataflowBuffer
// object (get_entry_size()) instead of get_local_cb_interface(cb_id).fifo_page_size. The OUT_SHARDED /
// BACKWARDS build variants are preserved so the fork is a drop-in for any consumer. The legacy copy is
// retained for the not-yet-ported consumers (~22 op families).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_pages = get_arg(args::num_pages);
    const auto start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb_out.get_entry_size();

#ifdef OUT_SHARDED
    dfb_out.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::output);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_out.wait_front(onepage);
        noc.async_write(dfb_out, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_out.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
