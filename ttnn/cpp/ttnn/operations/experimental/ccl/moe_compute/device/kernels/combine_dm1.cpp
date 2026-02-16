// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tt-metalium/constants.hpp"
#include "moe_ring_common.h"

//
// Combine destination core dm1 kernel.
//
// Each combine core waits for all source (ring) cores in its x-column to signal
// that they have finished writing all expert data.
//
// There are RING_CORES_PER_COMBINE_COL (12/4 = 3) source cores per x-column.
// Each source core increments this core's semaphore once after all experts are done.
//

void kernel_main() {
    // Run-time arguments
    uint32_t argidx = 0;
    const auto semaphore_id = get_arg_val<uint32_t>(argidx++);

    // Semaphore setup
    uint32_t semaphore_addr = get_semaphore(semaphore_id);
    volatile tt_l1_ptr uint32_t* semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    *semaphore_ptr = 0;

    // Number of source cores that signal this combine core
    constexpr uint32_t num_sources = moe_ring::NUM_CORES / 4;  // 12/4 = 3

    // Wait until all source cores have signaled completion
    noc_semaphore_wait(semaphore_ptr, num_sources);
}
