// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tt-metalium/constants.hpp"
#include "moe_ring_common.h"

#include "api/debug/dprint_pages.h"

//
// Combine destination core dm1 kernel.
//
// Each combine core waits for all source (ring) cores in its x-column to signal
// that they have finished writing all expert data.
//
// There are RING_CORES_PER_COMBINE_COL = num_cores / width_shard_dim source cores per x-column
// (DeepSeek WH: 12/4=3, BH: 8/4=2; GPT WH: 12/3=4, BH: 8/2=4).
// Each source core increments this core's semaphore once after all experts are done.
//
// NOTE: this kernel is currently not loaded by program_factory.cpp (no CreateKernel
// reference). Kept compilable + arch-aware for a future PR that wires up a combine-side
// dm1 path.
//

void kernel_main() {
    // Compile-time arguments (mirror dm1.cpp's CT-arg pattern so a future wire-up has
    // a value-equivalent reference for num_cores / width_shard_dim).
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto semaphore_id = get_arg_val<uint32_t>(argidx++);

    // Semaphore setup
    uint32_t semaphore_addr = get_semaphore(semaphore_id);
    volatile tt_l1_ptr uint32_t* semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    *semaphore_ptr = 0;

    // Number of source (ring) cores per x-column that signal this combine core.
    constexpr uint32_t num_sources = num_cores / width_shard_dim;

    // Wait until all source cores have signaled completion
    noc_semaphore_wait(semaphore_ptr, num_sources);
    const uint32_t output_base_l1_addr = get_write_ptr(tt::CBIndex::c_0);
}
