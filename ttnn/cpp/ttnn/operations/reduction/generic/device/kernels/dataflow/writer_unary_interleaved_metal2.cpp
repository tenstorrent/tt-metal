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
//   - Buffer sync (wait/pop) goes through `experimental::DataflowBuffer`, which is
//     arch-agnostic.
//
// Arch coverage caveat: see the header of `reader_unary_reduce_universal_start_id_metal2.cpp`.
// The address generator and `noc_async_write_tile` are Gen1-only; Quasar support for the
// data path is blocked on the same TensorAccessor / Metal 2.0 CTA framework issue.

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

    experimental::DataflowBuffer output_buf(dfb::output);

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = aligned_page_size,
        .data_format = get_dataformat(output_buf.get_id()),
    };

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        output_buf.wait_front(onetile);
        noc_async_write_tile(i, s, output_buf.get_read_ptr());
        noc_async_writes_flushed();
        output_buf.pop_front(onetile);
    }
    noc_async_write_barrier();
}
