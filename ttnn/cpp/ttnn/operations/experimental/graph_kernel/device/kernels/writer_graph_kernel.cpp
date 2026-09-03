// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// graph_kernel basis writer: drains the `pages` dataflow buffer into pages
// [start_id, start_id + num_pages) of the output tensor.
void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer pages(dfb::pages);
    Noc noc;
    const auto out = TensorAccessor(tensor::out);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        pages.wait_front(1);
        noc.async_write(pages, out, page_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        pages.pop_front(1);
    }
}
