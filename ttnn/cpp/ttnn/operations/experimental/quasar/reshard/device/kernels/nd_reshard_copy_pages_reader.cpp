// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Simple kernel that copies [start_page, end_page) pages from src to dst.
void kernel_main() {
    constexpr auto page_size = get_arg(args::page_size);

    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t end_page = get_arg(args::end_page);

    auto accessor_src = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb(dfb::cb);

    constexpr uint32_t one_tile = 1;
    auto pages = accessor_src.pages(start_page, end_page);
    for (const auto& page : pages) {
        dfb.reserve_back(one_tile);
        noc.async_read(
            accessor_src, dfb, page_size, {.page_id = page.page_id(), .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(one_tile);
    }
}
