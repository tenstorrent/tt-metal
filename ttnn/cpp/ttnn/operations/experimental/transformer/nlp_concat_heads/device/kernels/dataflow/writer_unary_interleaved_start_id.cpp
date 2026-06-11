// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the interleaved unary writer (op-private copy). The shared
// eltwise/unary writer_unary_interleaved_start_id.cpp is still consumed positionally by ~10 legacy ops
// and must not be touched, so nlp_concat_heads' interleaved path carries its own copy here. Only the
// binding mechanism changed: the CB id comes from the DFB token (dfb::), the output address from the
// TensorAccessor binding (ta::), and num_pages/start_id from named runtime args (args::). The forward,
// single-page write loop is preserved. (The shared kernel's OUT_SHARDED / BACKWARDS branches are dropped:
// this op's interleaved path never sets those defines.)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t cb_id_out = dfb::cb_id_out;

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    Noc noc;
    CircularBuffer cb(cb_id_out);

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::dst_args);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
}
