// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// graph_kernel writer: drains the compute kernel's `out` dataflow buffer into pages
// [start_id, start_id + pages_per_core) of the output tensor.
void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t pages_per_core = get_arg(args::pages_per_core);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer out_pages(dfb::out);
    Noc noc;
    const auto out = TensorAccessor(tensor::out);

    for (uint32_t i = start_id; i < start_id + pages_per_core; ++i) {
        out_pages.wait_front(1);
        noc.async_write(out_pages, out, page_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        out_pages.pop_front(1);
    }
}
