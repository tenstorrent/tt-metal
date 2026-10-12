// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute helpers for the DRAM-sharded unary work queue.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_sharded_common.hpp"

namespace dram_shard {

#if WORK_QUEUE
// Tile count of the next chunk from the reader; 0 means done.
ALWI uint32_t next_chunk_tiles() {
    DataflowBuffer counts(kCbComputeCount);
    counts.wait_front(1);
    const uint32_t count = counts.read_tile_value(0, 0);
    counts.pop_front(1);
    return count;
}
#endif

// Calls process(num_tiles) once, or with WORK_QUEUE process(count) once per chunk.
template <typename Process>
ALWI void for_each_chunk(uint32_t num_tiles, Process process) {
#if WORK_QUEUE
    for (uint32_t count = next_chunk_tiles(); count != 0; count = next_chunk_tiles()) {
        process(count);
    }
#else
    process(num_tiles);
#endif
}

// Per-tile loop counter: num_tiles iterations, or with WORK_QUEUE one per tile of every chunk.
class Tiles {
public:
#if WORK_QUEUE
    explicit Tiles(uint32_t /*num_tiles*/) {}
#else
    explicit Tiles(uint32_t num_tiles) : left_(num_tiles) {}
#endif

    ALWI bool next() {
#if WORK_QUEUE
        if (left_ == 0) {
            left_ = next_chunk_tiles();
        }
#endif
        if (left_ == 0) {
            return false;
        }
        --left_;
        return true;
    }

private:
    uint32_t left_ = 0;
};

}  // namespace dram_shard
