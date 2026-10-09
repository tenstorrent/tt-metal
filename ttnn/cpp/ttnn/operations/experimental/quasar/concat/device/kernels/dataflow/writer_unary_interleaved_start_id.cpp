// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar concat's copy of the shared Metal 2.0 writer
// ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp,
// kept here so the experimental/quasar op does not change when the shared kernel does. The binding
// names (dfb::out, tensor::dst) and the named argument set are that kernel's interface.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    // dfb holds the tiles/sticks to drain to the destination tensor; the host binds this kernel as
    // its consumer.
    DataflowBuffer dfb(dfb::out);

    // Get page size from the DFB entry size (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb.get_entry_size();

#ifdef OUT_SHARDED
    // Output is sharded in place; the wait is only a readiness handshake. Pop to
    // leave the DFB balanced.
    dfb.wait_front(num_pages);
    dfb.pop_front(num_pages);
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
