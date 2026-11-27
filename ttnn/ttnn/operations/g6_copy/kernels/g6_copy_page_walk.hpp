// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_copy {

// Moves the pages [block_first, block_first + block_len) between DRAM and the L1 region at l1_base,
// one NoC transfer per bank run (pages p, p + num_banks, ... sit contiguously in one bank).
// A page's L1 slot follows walk order, not page order, so the reader and the writer must both walk
// through this function with the same arguments.
template <bool IsRead, typename Accessor>
FORCE_INLINE void transfer_block(
    const Accessor& accessor, uint32_t block_first, uint32_t block_len, uint32_t walk_rotation, uint32_t l1_base) {
    // The bank stride of the accessor's default page size; the CB page size matches it on the host.
    const uint32_t page_bytes = accessor.get_aligned_page_size();
    // A run must stop at this block's end: bank contiguity continues into other blocks and other cores.
    const uint32_t block_end = block_first + block_len;
    const uint32_t num_banks = accessor.contiguous_page_stride();
    const uint32_t num_runs = block_len < num_banks ? block_len : num_banks;

    uint32_t k = walk_rotation % num_runs;
    uint32_t l1_addr = l1_base;
    for (uint32_t i = 0; i < num_runs; ++i) {
        const uint32_t page = block_first + k;
        const uint32_t run_bytes = accessor.num_contiguous_pages(page, block_end) * page_bytes;
        if constexpr (IsRead) {
            noc_async_read(accessor.get_noc_addr(page), l1_addr, run_bytes);
        } else {
            noc_async_write(l1_addr, accessor.get_noc_addr(page), run_bytes);
        }
        l1_addr += run_bytes;
        if (++k == num_runs) {
            k = 0;
        }
    }
}

}  // namespace g6_copy
