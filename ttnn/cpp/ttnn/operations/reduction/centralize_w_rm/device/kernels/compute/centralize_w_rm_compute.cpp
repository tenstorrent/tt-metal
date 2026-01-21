// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // CB IDs - for stub, just use input and output
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;    // Input RM sticks
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;  // Output RM sticks

    // Initialize compute kernel
    copy_tile_init(cb_in_rm);

    //  STUB KERNEL: Just copy data from input CB to output CB
    // Real implementation (Stage 7) will do:
    //   tilize -> reduce -> bcast_sub -> untilize
    // For now, just pass data through to verify:
    //   1. Kernels compile at runtime
    //   2. CB synchronization doesn't deadlock
    //   3. Output tensor has correct shape

    const uint32_t num_tiles = Ht * Wt;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Wait for 1 tile from reader
        cb_wait_front(cb_in_rm, 1);

        // Reserve output space
        cb_reserve_back(cb_out_rm, 1);

        // Copy tile through
        copy_tile(cb_in_rm, 0, 0);

        // Release
        cb_push_back(cb_out_rm, 1);
        cb_pop_front(cb_in_rm, 1);
    }
}

}  // namespace NAMESPACE
