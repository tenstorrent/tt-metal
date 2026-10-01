// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace reduce_scatter_common {

/* Compute the ring-directional parity for a chunk of tiles such that it is independent of worker distribution or
 * total number of chunks. This avoids some lower dim dependent non-determinism ultimately caused by floating point
 * accumulation non-associativity.
 *
 */

template <uint32_t tile_granularity>
std::tuple<bool, uint32_t> chunk_ring_parity(uint32_t tiles_read, uint32_t total_tiles_to_read) {
    const bool is_even_chunk = ((tiles_read / tile_granularity) % 2) == 0;
    const uint32_t next_boundary = ((tiles_read / tile_granularity) + 1) * tile_granularity;
    const uint32_t tiles_to_read = std::min(next_boundary, total_tiles_to_read) - tiles_read;

    return std::make_tuple(is_even_chunk, tiles_to_read);
}
}  // namespace reduce_scatter_common
