// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-sharded unary, compute side: work queue (WORK_QUEUE).

#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_sharded_common.hpp"

namespace dram_shard {

// Calls process(num_tiles), or with WORK_QUEUE process(count) for each chunk the reader announces, until count 0.
// read_tile_value gives unpack, math and pack the same count.
template <typename Process>
ALWI void for_each_chunk(uint32_t num_tiles, Process process) {
#if WORK_QUEUE
    while (true) {
        cb_wait_front(kCbComputeCount, 1);
        const uint32_t count = read_tile_value(kCbComputeCount, 0, 0);
        cb_pop_front(kCbComputeCount, 1);
        if (count == 0) {
            break;
        }
        process(count);
    }
#else
    process(num_tiles);
#endif
}

}  // namespace dram_shard
