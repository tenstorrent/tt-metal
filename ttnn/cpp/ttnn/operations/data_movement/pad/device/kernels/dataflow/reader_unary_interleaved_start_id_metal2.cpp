// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary's reader_unary_interleaved_start_id.cpp, owned by pad's
// single-core tile factory. The device-side NoC + TensorAccessor logic is unchanged; only the
// resource bindings are migrated to the Metal 2.0 namespaces (dfb::/ta::/args::). The legacy
// shared reader is left untouched for its many co-borrowers; sunset this fork when the shared
// reader itself is Metal-2.0-prepped.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    const auto s = TensorAccessor(ta::src);

    Noc noc;
    DataflowBuffer cb(dfb::cb_in0);

    // Page size from the DFB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = cb.get_tile_size();

    // ublocks size defined in pages (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

// read a ublock of pages from src to the DFB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb.reserve_back(onepage);
        noc.async_read(s, cb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(onepage);
    }
}
