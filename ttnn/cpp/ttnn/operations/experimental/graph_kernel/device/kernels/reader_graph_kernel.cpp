// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// graph_kernel basis reader: streams pages [start_id, start_id + num_pages) of input 0
// into the `pages` dataflow buffer. Inputs 1..num_inputs-1 are bound as tensor::in1,
// tensor::in2, ... and are available for the graph to consume later.
void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer pages(dfb::pages);
    Noc noc;
    const auto in0 = TensorAccessor(tensor::in0);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        pages.reserve_back(1);
        noc.async_read(in0, pages, page_size, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        pages.push_back(1);
    }
}
