// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp. Identical dataflow logic; the CB and
// output tensor are sourced from Metal 2.0 named bindings (dfb::out / tensor::output) and named args
// instead of a CB-index CTA, a buffer-address RTA, and TensorAccessorArgs plumbing. The legacy
// writer_unary_interleaved_start_id.cpp is retained for the not-yet-ported multi_core_default
// factory; delete this fork's twin once both factories are on Metal 2.0.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer cb(dfb::out);
    // QSR: read page size from the DFB object (get_local_cb_interface().fifo_page_size is stale for Metal-2.0 DFBs);
    // must be after cb construction
    const uint32_t page_bytes = cb.get_entry_size();

#ifdef OUT_SHARDED
    cb.wait_front(num_pages);
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
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
