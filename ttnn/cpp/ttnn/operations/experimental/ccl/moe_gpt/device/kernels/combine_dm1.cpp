// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

//
// Combine destination core dm1 kernel.
//
// Each combine core waits for all source (ring) cores in its width column to signal
// that they have finished writing all expert data.
//
// There are num_cores / width_shard_dim source cores per width column (12/3 = 4 on Wormhole,
// e.g. 8/2 = 4 on Blackhole). Both values are passed in as named compile-time args by the
// program factory rather than read from the kernel header's fixed ring-size-12 assumption, so
// this stays correct for any ring size.
// Each source core increments this core's semaphore once after all experts are done.
//

void kernel_main() {
    uint32_t argidx = 0;
    const auto semaphore_id = get_arg_val<uint32_t>(argidx++);

    Semaphore<> sem(semaphore_id);
    sem.set(0);

    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");
    constexpr uint32_t num_sources = num_cores / width_shard_dim;

    sem.wait_min(num_sources);
}
