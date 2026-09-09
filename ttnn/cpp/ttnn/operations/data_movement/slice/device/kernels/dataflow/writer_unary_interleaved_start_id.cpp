// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Slice's private copy of the eltwise/unary writer of the same name. It is functionally identical to
// the Metal 2.0 fork beside that original, writer_unary_interleaved_start_id_metal2.cpp: same
// dfb::out and tensor::dst bindings, same named arguments, same OUT_SHARDED / BACKWARDS branches.
// Nothing distinguishes the two, so this copy is a candidate for retirement in favour of binding
// that fork, as the tensor-args slice factory already does.
//
// TODO(#52228): the issue tracks the consolidation of this kernel's duplicates.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_out(dfb::out);

    // Get page size from the dataflow buffer (works for both TILE and ROW_MAJOR layouts)
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
