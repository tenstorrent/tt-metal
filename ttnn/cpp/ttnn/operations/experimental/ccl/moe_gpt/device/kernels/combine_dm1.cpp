// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "moe_gpt_ring_common.h"

//
// Combine destination core dm1 kernel.
//
// Each combine core waits for all source (ring) cores in its width column to signal
// that they have finished writing all expert data.
//
// There are RING_CORES_PER_COMBINE_COL (12/3 = 4) source cores per width column.
// Each source core increments this core's semaphore once after all experts are done.
//

void kernel_main() {
    uint32_t argidx = 0;
    const auto semaphore_id = get_arg_val<uint32_t>(argidx++);

    Semaphore<> sem(semaphore_id);
    sem.set(0);

    constexpr uint32_t num_sources = moe_gpt_ring::RING_CORES_PER_COMBINE_COL;  // 4

    sem.wait_min(num_sources);
}
