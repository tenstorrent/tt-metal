// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Data-movement helpers shared by the prepare_chunk_recurrence reader and writer.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"

namespace kda_prep {

// Write `items` work items' tiles, `tiles` each and published together, to their pages.
template <typename DFB, typename Accessor>
inline void drain_items(
    Noc& noc, DFB& buffer, const Accessor& accessor, uint32_t tiles, const uint32_t* work_items, uint32_t items) {
    buffer.wait_front(items * tiles);
    for (uint32_t item = 0; item < items; ++item) {
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            noc.async_write(
                buffer,
                accessor,
                buffer.get_entry_size(),
                {.offset_bytes = (item * tiles + tile) * buffer.get_entry_size()},
                {.page_id = work_items[item] * tiles + tile});
        }
    }
    noc.async_write_barrier();
    buffer.pop_front(items * tiles);
}

}  // namespace kda_prep
