// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);

    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);
    tilize_init(cb_id_in0, tiles_per_chunk, cb_id_out0);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        // Process the block in chunks to fit within L1 memory limits
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_wait_front(cb_id_in0, tiles_per_chunk);
            cb_reserve_back(cb_id_out0, tiles_per_chunk);

            tilize_block(cb_id_in0, tiles_per_chunk, cb_id_out0);

            cb_push_back(cb_id_out0, tiles_per_chunk);
            cb_pop_front(cb_id_in0, tiles_per_chunk);
        }
    }
}
}  // namespace NAMESPACE
