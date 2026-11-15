// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"

// Writes this core's 2 * STATS_NTILES stats tiles (per output tile column: the sum row, then the sum of squares row)
// to the epilogue_stats tensor. Common runtime args: the tensor's address, the slice's first page.
template <uint32_t stats_cta>
FORCE_INLINE void write_epilogue_stats(Noc& noc, uint32_t core_first_page) {
    constexpr uint32_t num_pages = 2 * STATS_NTILES;
    const uint32_t page_size = get_tile_size(STATS_CB);
    const uint32_t first_page = get_common_arg_val<uint32_t>(1) + core_first_page;
    const auto stats = TensorAccessor(TensorAccessorArgs<stats_cta>(), get_common_arg_val<uint32_t>(0), page_size);
    DataflowBuffer stats_dfb(STATS_CB);
    stats_dfb.wait_front(num_pages);
    for (uint32_t t = 0; t < num_pages; ++t) {
        noc.async_write(stats_dfb, stats, page_size, {.offset_bytes = t * page_size}, {.page_id = first_page + t});
    }
    noc.async_write_barrier();
    stats_dfb.pop_front(num_pages);
}
