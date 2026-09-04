// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Slice's own copy of the eltwise/unary interleaved writer, kept separate because the two took
// their buffer index by different means: this copy read a named compile-time arg, so the index
// could be remapped by the fusion infrastructure, while the eltwise copy read positional arg 0.
// Under Metal 2.0 neither reads an index at all -- the buffer arrives as the dfb::out binding, and
// remapping is what a binding is -- so the reason the copies diverged no longer applies. They are
// candidates for consolidation; see issue #52228, which tracks the same question for the other
// copies of this kernel.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    // Create objects for Device 2.0 API
    // dfb_out holds the pages the reader staged; the host binds this kernel as its consumer.
    DataflowBuffer dfb_out(dfb::out);

    // Get page size from the DFB entry size (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb_out.get_entry_size();
    Noc noc;

#ifdef OUT_SHARDED
    dfb_out.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::dst);

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
