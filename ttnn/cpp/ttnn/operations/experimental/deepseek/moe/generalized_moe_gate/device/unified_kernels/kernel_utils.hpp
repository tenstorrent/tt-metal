// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#endif

namespace unified_kernels {

// ============================================================================
// Sharded persistent buffer setup utilities
// ============================================================================

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)

// Setup a sharded persistent buffer by reserving and pushing tiles
// This makes the buffer available for compute to read from
// Note: Can be called from either NCRISC or BRISC, whichever runs first
FORCE_INLINE void setup_sharded_buffer(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(num_tiles);
    cb.push_back(num_tiles);
}

#endif

}  // namespace unified_kernels
