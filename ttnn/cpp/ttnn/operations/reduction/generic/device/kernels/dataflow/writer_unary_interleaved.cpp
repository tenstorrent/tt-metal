// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 writer for the multi-core / single-core reduction primitive (also
// used by Welford W and H).
//
// Migration notes:
//   - Runtime arguments (args::num_pages, args::start_id) are bound by name
//     and threaded per-node via the host's ProgramRunParams.
//   - The output dataflow buffer is bound by name (dfb::output).
//   - The output tensor is bound by name (ta::output_tensor); the host supplies
//     a MeshTensor via ProgramRunParams::TensorArg so address generation no
//     longer needs is_dram or aligned_page_size as kernel arguments.
//   - Buffer sync (wait/pop) goes through experimental::DataflowBuffer.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    // Per-node runtime arguments.
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    experimental::DataflowBuffer dfb_output(dfb::output);

    TensorAccessor output_accessor(ta::output_tensor);
    const uint32_t tile_bytes = dfb_output.get_tile_size();

    experimental::Noc noc;

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        dfb_output.wait_front(onetile);
        noc.async_write(dfb_output, output_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_output.pop_front(onetile);
    }
    noc.async_write_barrier();
}
