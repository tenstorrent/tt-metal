// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One core of an op reads pages every core needs and multicasts them into the same buffer of every other core
// (common/device/page_multicast.hpp builds the rectangles), rather than every core reading the same pages.

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"

namespace nomic_embed_text_v2_moe {

// The sender reads pages [0, pages) of source into destination, which sits at the same address on every core of
// the op, multicasts them there and sets ready there; the others wait for ready. The caller reserves and pushes
// destination, and the sender waits for its writes before it exits. The rectangles are runtime arguments from
// rectangle_args on, as multicast_rectangle_args lays them out.
template <typename Accessor>
void multicast_pages(
    const Noc& noc,
    const Accessor& source,
    DataflowBuffer& destination,
    uint32_t pages,
    uint32_t page_bytes,
    Semaphore<>& ready,
    bool sender,
    uint32_t rectangle_args) {
    if (!sender) {
        ready.wait(1);
        return;
    }
    for (uint32_t page = 0; page < pages; ++page) {
        noc.async_read(source, destination, page_bytes, {.page_id = page}, {.offset_bytes = page * page_bytes});
    }
    noc.async_read_barrier();
    const uint32_t address = destination.get_write_ptr();
    ready.set(1);
    const uint32_t rectangles = get_arg_val<uint32_t>(rectangle_args);
    for (uint32_t rectangle = 0; rectangle < rectangles; ++rectangle) {
        const uint32_t arg = rectangle_args + 1 + 5 * rectangle;
        const uint32_t start_x = get_arg_val<uint32_t>(arg);
        const uint32_t start_y = get_arg_val<uint32_t>(arg + 1);
        const uint32_t end_x = get_arg_val<uint32_t>(arg + 2);
        const uint32_t end_y = get_arg_val<uint32_t>(arg + 3);
        const uint32_t count = get_arg_val<uint32_t>(arg + 4);
        // Linked, so the flag reaches each core only after the pages.
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(address),
            MulticastEndpoint{},
            pages * page_bytes,
            count,
            {},
            {.noc_x_start = start_x, .noc_y_start = start_y, .noc_x_end = end_x, .noc_y_end = end_y, .addr = address},
            true);
        ready.set_multicast(noc, start_x, start_y, end_x, end_y, count);
    }
}

}  // namespace nomic_embed_text_v2_moe
