// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_copy {

enum class Direction : uint8_t { Read, Write };

// Moves one block of `block_len` pages starting at page `block_first` between DRAM and the
// staging slot at `l1_base`, one NoC call per bank run. Pages p, p + num_banks, ... of an
// interleaved tensor are contiguous in one bank, so each run is a single sequential span.
//
// The L1 layout of the block follows walk order, not page order. The reader and the writer
// must call this with identical (block_first, block_len, walk_rotation) so page p's bytes sit
// at the same L1 offset on both sides. Runs are bounded by the block's own end, never the
// tensor's: a longer run would spill into another block's slot or another core's pages.
template <Direction dir, typename Accessor>
FORCE_INLINE void transfer_block(
    const Accessor& accessor,
    uint32_t block_first,
    uint32_t block_len,
    uint32_t walk_rotation,
    uint32_t l1_base,
    uint32_t page_bytes) {
    const uint32_t block_end = block_first + block_len;
    const uint32_t num_banks = accessor.contiguous_page_stride();
    const uint32_t nk = block_len < num_banks ? block_len : num_banks;
    uint32_t k = walk_rotation % nk;
    uint32_t l1 = l1_base;
    for (uint32_t i = 0; i < nk; ++i) {
        const uint32_t p = block_first + k;
        const uint32_t run_bytes = accessor.num_contiguous_pages(p, block_end) * page_bytes;
        if constexpr (dir == Direction::Read) {
            noc_async_read(accessor.get_noc_addr(p), l1, run_bytes);
        } else {
            noc_async_write(l1, accessor.get_noc_addr(p), run_bytes);
        }
        l1 += run_bytes;
        k = (k + 1 == nk) ? 0 : k + 1;
    }
}

}  // namespace g6_copy
