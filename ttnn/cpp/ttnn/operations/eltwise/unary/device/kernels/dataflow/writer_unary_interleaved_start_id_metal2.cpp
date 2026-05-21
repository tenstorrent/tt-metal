// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp.
// Kept side-by-side with the legacy copy during the bulk-port window.
// The only sanctioned changes from the legacy copy are: arg-retrieval calls,
// DFB-wrapper construction (CB id → dfb::name), and the TensorAccessor token
// (TensorAccessorArgs<N>() → TensorAccessor(ta::output)). The OUT_SHARDED and
// BACKWARDS #ifdef branches are preserved as legacy control flow.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_pages = get_arg(args::num_pages);
    auto start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer cb(dfb::out_dfb);

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts).
    // For tile-format DFBs in the reduction op family, this is equal to the
    // configured per-tile byte size.
    const uint32_t page_bytes = cb.get_tile_size();

#ifdef OUT_SHARDED
    cb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::output);

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
