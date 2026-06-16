// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Simple kernel that copies [start_page, end_page) pages from dst to dst.
void kernel_main() {
    auto accessor_dst = TensorAccessor(ta::output);

    constexpr uint32_t page_size = get_arg(args::page_size);

    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t end_page = get_arg(args::end_page);

    Noc noc;
    DataflowBuffer dfb_in(dfb::cb_in0);

    constexpr uint32_t one_tile = 1;
    auto pages = accessor_dst.pages(start_page, end_page);
    for (const auto& page : pages) {
        dfb_in.wait_front(one_tile);
        noc.async_write(
            dfb_in, accessor_dst, page_size, {.offset_bytes = 0}, {.page_id = page.page_id(), .offset_bytes = 0});
        noc.async_write_barrier();
        dfb_in.pop_front(one_tile);
    }
}
