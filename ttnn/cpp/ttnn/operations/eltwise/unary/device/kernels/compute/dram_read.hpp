// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute helpers for the unary DRAM read work queue: with WORK_QUEUE the reader hands over tiles chunk by chunk
// instead of a fixed count per core.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace dram_read {

// Calls process(num_tiles) once; with WORK_QUEUE, process(count) for every chunk the reader hands over, until it sends
// a count of 0.
template <typename Process>
ALWI void for_each_chunk(uint32_t num_tiles, Process process) {
#if WORK_QUEUE
    DataflowBuffer counts(dfb::chunk_count);
    while (true) {
        counts.wait_front(1);
        const uint32_t count = counts.read_tile_value(0, 0);
        counts.pop_front(1);
        if (count == 0) {
            return;
        }
        process(count);
    }
#else
    process(num_tiles);
#endif
}

}  // namespace dram_read
