// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/.../writer_unary_interleaved_start_id.cpp, specialized for
// nlp_concat_heads' interleaved path (forward, non-sharded output). The legacy shared writer
// reads a positional CTA + TensorAccessorArgs + a buffer-address RTA, which a Metal 2.0 factory
// cannot emit; this fork uses named bindings (dfb::in / ta::output) instead. The legacy copy
// stays in place for its ~31 co-borrowers. See METAL2_PORT_REPORT.md (cross-op kernel fork).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    // Page size from the CB interface (works for both TILE and ROW_MAJOR layouts).
    const uint32_t page_bytes = get_local_cb_interface(dfb::in).fifo_page_size;

    Noc noc;
    DataflowBuffer cb(dfb::in);

    constexpr uint32_t onepage = 1;
    const auto s = TensorAccessor(ta::output);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
}
