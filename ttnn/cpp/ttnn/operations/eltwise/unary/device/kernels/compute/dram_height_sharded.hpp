// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute side of the DRAM height-sharded work queue (../dataflow/dram_height_sharded.hpp).

#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_height_sharded_common.hpp"

namespace dram_hs {

// Runs process(n) over this core's tiles: once with num_tiles, or with WORK_QUEUE once per chunk the reader
// announces, until a count of 0. read_tile_value gives unpack, math and pack the same count.
template <typename Process>
ALWI void for_each_chunk(uint32_t num_tiles, Process process) {
#if WORK_QUEUE
    while (true) {
        cb_wait_front(kCbComputeCount, 1);
        const uint32_t n = read_tile_value(kCbComputeCount, 0, 0);
        cb_pop_front(kCbComputeCount, 1);
        if (n == 0) {
            return;
        }
        process(n);
    }
#else
    process(num_tiles);
#endif
}

}  // namespace dram_hs
