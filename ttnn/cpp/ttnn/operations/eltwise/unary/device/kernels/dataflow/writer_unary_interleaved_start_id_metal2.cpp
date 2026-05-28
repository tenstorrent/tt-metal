// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp.
//
// Bindings (named, from host KernelSpec):
//   dfb::out                  — DFB endpoint (CONSUMER) — page-aligned (TILE or ROW_MAJOR)
//   ta::out                   — TensorAccessor (output, interleaved)
//   args::num_pages           — RTA
//   args::start_id            — RTA
//
// OUT_SHARDED / BACKWARDS defines preserved as build-time toggles (driven by
// KernelSpec::defines).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_pages = get_arg(args::num_pages);
    auto start_id = get_arg(args::start_id);

    // Get page size from DFB (works for both TILE and ROW_MAJOR layouts).
    const uint32_t page_bytes = get_tile_size(dfb::out);

    Noc noc;
    DataflowBuffer cb(dfb::out);

#ifdef OUT_SHARDED
    cb.wait_front(num_pages);
#else
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::out);

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
