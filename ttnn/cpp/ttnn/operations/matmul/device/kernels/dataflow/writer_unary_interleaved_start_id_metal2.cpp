// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp. The legacy writer is still bound by the
// generic-op gtest (tests/ttnn/unit_tests/gtests/test_generic_op.cpp), which builds a
// ProgramDescriptor with positional args, so the Metal 2.0 multi-core matmul factory binds a forked
// copy with named args, a DFB handle, and a typed tensor binding. (The non-matmul OUT_SHARDED /
// BACKWARDS paths are dropped — the matmul multi-core factory exercises only the interleaved,
// forward path.)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr auto cb_id_out = dfb::cb_out;

    Noc noc;
    DataflowBuffer cb_out(cb_id_out);

    // Get page size from the DFB (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = cb_out.get_entry_size();

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::dst);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_out.wait_front(onepage);
        noc.async_write(cb_out, s, page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        cb_out.pop_front(onepage);
    }
    noc.async_write_barrier();
}
