// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the interleaved unary reader (op-private copy). The shared
// eltwise/unary reader_unary_interleaved_start_id.cpp is still consumed positionally by many legacy ops
// and must not be touched, so typecast's migrated interleaved / subgrid factories carry their own copy
// here. Only the binding mechanism changed: the input address comes from the TensorAccessor binding
// (ta::), the CB id from the DFB token (dfb::), and num_pages/start_id from named runtime args (args::).
// The forward, single-page read loop is preserved. (The shared kernel's BACKWARDS branch is dropped:
// this op's interleaved path never sets that define.)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t cb_id_in0 = dfb::input_cb;

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    // ublocks size defined in pages (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::src_args);

    Noc noc;
    CircularBuffer cb(cb_id_in0);

    // read a ublock of pages from src to CB, and then push the ublock to unpacker
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.reserve_back(onepage);
        noc.async_read(s, cb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(onepage);
    }
}
