// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Writes this core's 2 * STATS_NTILES stats tiles (per output tile column: the sum row, then the sum of squares row)
// to the epilogue_stats tensor. Common runtime args: the tensor's address, the slice's first page.
template <uint32_t stats_cta>
FORCE_INLINE void write_epilogue_stats(uint32_t core_first_page) {
    constexpr uint32_t num_pages = 2 * STATS_NTILES;
    const uint32_t page_size = get_tile_size(STATS_CB);
    const uint32_t first_page = get_common_arg_val<uint32_t>(1) + core_first_page;
    const auto stats = TensorAccessor(TensorAccessorArgs<stats_cta>(), get_common_arg_val<uint32_t>(0), page_size);
    cb_wait_front(STATS_CB, num_pages);
    const uint32_t l1_addr = get_read_ptr(STATS_CB);
    for (uint32_t t = 0; t < num_pages; ++t) {
        noc_async_write_page(first_page + t, stats, l1_addr + t * page_size);
    }
    noc_async_write_barrier();
    cb_pop_front(STATS_CB, num_pages);
}
