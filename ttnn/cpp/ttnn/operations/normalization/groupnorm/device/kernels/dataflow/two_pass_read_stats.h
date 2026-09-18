// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"

template <std::uint32_t TilesPerRow, std::uint32_t TileBytes, bool HasAlias, typename Accessor>
inline void read_two_pass_stats_block(
    const Noc& noc,
    const Accessor& src,
    DataflowBuffer& input,
    DataflowBuffer& alias,
    std::uint32_t ring_end,
    std::uint32_t start_tile,
    std::uint32_t row_stride,
    std::uint32_t rows) {
    const std::uint32_t tiles = rows * TilesPerRow;
    for (std::uint32_t tile = 0; tile < tiles;) {
        const std::uint32_t address = input.get_write_ptr();
        // A reservation must never cross the CB's physical wrap boundary.
        const std::uint32_t count =
            std::min<std::uint32_t>(8, std::min(tiles - tile, (ring_end - address) / TileBytes));
        input.reserve_back(count);
        if constexpr (HasAlias) {
            alias.reserve_back(count);
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t offset = tile + i;
            noc.async_read(
                src,
                CoreLocalMem<std::uint32_t>(address + i * TileBytes),
                TileBytes,
                {.page_id = start_tile + (offset / TilesPerRow) * row_stride + offset % TilesPerRow},
                {});
        }
        noc.async_read_barrier();
        input.push_back(count);
        if constexpr (HasAlias) {
            alias.push_back(count);
        }
        tile += count;
    }
}
