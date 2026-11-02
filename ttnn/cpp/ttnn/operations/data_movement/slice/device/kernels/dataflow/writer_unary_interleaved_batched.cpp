// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Page writer with one write barrier per batch of pages; each batch is popped as a unit.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t tiles_per_batch = get_arg(args::tiles_per_batch);

    DataflowBuffer dfb_out(dfb::out);
    const uint32_t page_bytes = dfb_out.get_entry_size();
    Noc noc;

    const auto s = TensorAccessor(tensor::dst);

    uint32_t page_id = start_id;
    for (uint32_t done = 0; done < num_pages;) {
        const uint32_t remaining = num_pages - done;
        const uint32_t n = remaining < tiles_per_batch ? remaining : tiles_per_batch;
        dfb_out.wait_front(n);
        uint32_t l1_read_addr = dfb_out.get_read_ptr();
        for (uint32_t t = 0; t < n; ++t) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(src, s, page_bytes, {.offset_bytes = 0}, {.page_id = page_id + t});
            l1_read_addr += page_bytes;
        }
        noc.async_write_barrier();
        dfb_out.pop_front(n);
        page_id += n;
        done += n;
    }
}
