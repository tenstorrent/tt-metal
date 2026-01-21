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

    // CB IDs
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;

    // Initialize copy tile
    copy_tile_init();

    // STUB: Simple passthrough - consume input, produce output
    // This verifies CB sync and kernel compilation, not correctness
    for (uint32_t block = 0; block < Ht; ++block) {
        // Wait for input from reader
        cb_wait_front(cb_in_rm, Wt);

        // Reserve output space
        cb_reserve_back(cb_out_rm, 1);

        // Copy first input page to output (garbage data - this is expected for stubs)
        copy_tile(cb_in_rm, 0, 0);

        // Push output to writer
        cb_push_back(cb_out_rm, 1);

        // Pop consumed input
        cb_pop_front(cb_in_rm, Wt);
    }
}

}  // namespace NAMESPACE
