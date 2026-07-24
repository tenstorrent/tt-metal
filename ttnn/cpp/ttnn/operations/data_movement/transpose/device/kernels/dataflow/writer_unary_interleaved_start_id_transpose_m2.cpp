// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp,
// forked into the transpose op directory because the orchestration constraint forbids editing
// the broadly-shared donor in place. Used by the HC-Tiled and WH (tiled) transpose factories.
// Sunset when the shared donor and all its consumers migrate to Metal 2.0.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb(dfb::out);

    // Get page size from the DFB (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb.get_entry_size();

#ifdef OUT_SHARDED
    dfb.wait_front(num_pages);
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
        dfb.wait_front(onepage);
        noc.async_write(dfb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
