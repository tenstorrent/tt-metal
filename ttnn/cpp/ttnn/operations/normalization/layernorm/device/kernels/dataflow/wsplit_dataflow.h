// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Width-split interleaved RMSNorm: drain blocks [b_begin, b_end) of this core's Wt-tile row segment from a dataflow
// buffer to an interleaved TILE tensor (tile ids tile_offset + block.start() + i), flushing (not barriering) per
// block; the caller issues the final write barrier. Blocks as generic::blocks(Wt, blk).
#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"

namespace norm::layernorm::wsplit {

// number of blocks of the row the reader drains (the first ones; the writer drains the rest)
inline uint32_t reader_out_blocks(uint32_t Wt, uint32_t blk) { return ((Wt + blk - 1) / blk + 1) / 2; }

template <typename Accessor>
inline void drain_blocks(
    const Noc& noc,
    DataflowBuffer& dfb,
    const Accessor& dst,
    uint32_t tile_offset,
    uint32_t Wt,
    uint32_t blk,
    uint32_t b_begin,
    uint32_t b_end) {
    const uint32_t tile_bytes = dfb.get_tile_size();
    uint32_t b = 0;
    for (auto block : norm::kernel_util::generic::blocks(Wt, blk)) {
        if (b >= b_begin && b < b_end) {
            dfb.wait_front(static_cast<uint16_t>(block.full_block_size()));
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_write(
                    dfb,
                    dst,
                    tile_bytes,
                    {.offset_bytes = idx * tile_bytes},
                    {.page_id = tile_offset + block.start() + i});
                idx++;
            }
            noc.async_writes_flushed();
            dfb.pop_front(static_cast<uint16_t>(block.full_block_size()));
        }
        b++;
    }
}

}  // namespace norm::layernorm::wsplit
