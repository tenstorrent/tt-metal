// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute side of the DRAM height-sharded work queue (WORK_QUEUE=1, see
// ../dataflow/unary_work_queue.hpp): the reader announces each chunk's tile count on CB 4.

#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"

namespace unary_wq {

// Tile count of the next chunk; 0 means there is no more work. Unpack, math and pack all receive
// the same value (read_tile_value distributes it from the unpack thread).
ALWI uint32_t next_chunk_tiles() {
    constexpr uint32_t cb_count = tt::CBIndex::c_4;
    cb_wait_front(cb_count, 1);
    const uint32_t n = read_tile_value(cb_count, 0, 0);
    cb_pop_front(cb_count, 1);
    return n;
}

}  // namespace unary_wq
