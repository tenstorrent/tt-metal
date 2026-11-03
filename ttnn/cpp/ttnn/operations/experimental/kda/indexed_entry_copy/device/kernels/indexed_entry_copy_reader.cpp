// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads this core's page range of one entry from the source into the staging DFB, a batch per push.

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "indexed_entry.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t page_stride = get_arg(args::page_stride);
    constexpr uint32_t batch_pages = get_arg(args::batch_pages);
    constexpr bool indexed_is_source = get_arg(args::indexed_is_source);
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    const Noc noc;
    uint32_t entry_base;
    if (!resolve_entry_base(noc, first_page, entry_base)) {
        return;
    }
    const uint32_t source_base = indexed_is_source ? entry_base : first_page;

    const auto source = TensorAccessor(tensor::source);
    DataflowBuffer staging(dfb::staging);
    for (uint32_t done = 0; done < num_pages; done += batch_pages) {
        const uint32_t count = num_pages - done < batch_pages ? num_pages - done : batch_pages;
        staging.reserve_back(count);
        const uint32_t batch_addr = staging.get_write_ptr();
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(
                source,
                CoreLocalMem<uint32_t>(batch_addr + i * page_stride),
                page_size,
                {.page_id = source_base + done + i},
                {});
        }
        noc.async_read_barrier();
        staging.push_back(count);
    }
}
