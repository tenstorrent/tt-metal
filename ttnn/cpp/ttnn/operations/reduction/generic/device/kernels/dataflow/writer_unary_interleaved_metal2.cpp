// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 writer for the multi-core reduction primitive.
//
// Migration notes:
//   - All compile-time arguments are bound by name (`args::*`).
//   - Runtime arguments (`args::dst_addr`, `args::num_pages`, `args::start_id`) are
//     bound by name and threaded per-node via the host's ProgramRunParams.
//   - The output dataflow buffer is bound by name (`dfb::output`).
//   - Address generation uses `InterleavedAddrGenFast` for the same reason as the
//     reader: Metal 2.0 ProgramSpec does not currently support the positional
//     compile-time arguments that `TensorAccessorArgs<N>()` requires. Sharded outputs
//     are therefore not supported by this Metal 2.0 writer yet.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/dataflow_buffer.h"

void kernel_main() {
    // Per-node runtime arguments.
    const uint32_t dst_addr = get_arg(args::dst_addr);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    // Compile-time arguments.
    constexpr uint32_t aligned_page_size = get_arg(args::aligned_page_size);
    constexpr bool is_dram = get_arg(args::is_dram) != 0;

    experimental::DataflowBuffer cb_output(dfb::output);

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = aligned_page_size,
        .data_format = get_dataformat(cb_output.get_id()),
    };

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        cb_output.wait_front(onetile);
        noc_async_write_tile(i, s, cb_output.get_read_ptr());
        noc_async_writes_flushed();
        cb_output.pop_front(onetile);
    }
    noc_async_write_barrier();
}
