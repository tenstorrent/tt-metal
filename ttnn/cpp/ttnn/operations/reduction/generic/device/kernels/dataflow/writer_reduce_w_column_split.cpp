// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reduce W column split. Every core of a tile row sends its partial tile into its own slot of the
// gather buffer on the row's first core; that core waits for all of them, and its compute kernel
// combines them into the tile this kernel writes out.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_parts = get_arg(args::num_parts);
    const uint32_t part = get_arg(args::part);
    const uint32_t first_x = get_arg(args::first_x);
    const uint32_t first_y = get_arg(args::first_y);
    const uint32_t out_page = get_arg(args::out_page);

    Noc noc;
    DataflowBuffer dfb_partial(dfb::partial);
    DataflowBuffer dfb_gather(dfb::gather);
    Semaphore gathered(sem::gathered);

    const bool is_first = part == 0;
    if (is_first) {
        dfb_gather.reserve_back(num_parts);
    }
    // Nothing is ever pushed to gather on the other cores, so its write pointer is the slot base
    // on every core of the row.
    const uint32_t entry_bytes = dfb_gather.get_entry_size();
    dfb_partial.wait_front(1);
    noc.async_write(
        dfb_partial,
        UnicastEndpoint{},
        entry_bytes,
        {},
        {.noc_x = first_x, .noc_y = first_y, .addr = dfb_gather.get_write_ptr() + (part * entry_bytes)});
    noc.async_write_barrier();
    dfb_partial.pop_front(1);
    gathered.up(noc, first_x, first_y, 1);
    if (!is_first) {
        noc.async_atomic_barrier();
        return;
    }

    gathered.wait_min(num_parts);
    dfb_gather.push_back(num_parts);

    DataflowBuffer dfb_out(dfb::out);
    const auto dst = TensorAccessor(tensor::dst);
    dfb_out.wait_front(1);
    noc.async_write(dfb_out, dst, dfb_out.get_entry_size(), {}, {.page_id = out_page});
    noc.async_write_barrier();
    dfb_out.pop_front(1);
}
