// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_interleaved_start_id.cpp.
//
// Bindings:
//   dfb::input              — DFB endpoint (PRODUCER)
//   ta::input               — TensorAccessor (input, interleaved)
//   args::num_pages         — RTA
//   args::start_id          — RTA
//
// BACKWARDS define preserved as a build-time toggle.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_pages = get_arg(args::num_pages);
    auto start_id = get_arg(args::start_id);

    const uint32_t page_bytes = get_tile_size(dfb::input);
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::input);

    Noc noc;
    DataflowBuffer cb(dfb::input);

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
