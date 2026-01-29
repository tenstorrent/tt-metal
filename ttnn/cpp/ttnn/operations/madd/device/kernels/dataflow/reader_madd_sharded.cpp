// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

/*
 * Sharded reader for madd operation.
 * Data is already in L1 (sharded), so we just need to:
 * 1. Signal that input tiles are available in the CBs
 * 2. Generate a zero tile for the compute kernel
 */
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_srcA_index = get_compile_time_arg_val(0);
    constexpr uint32_t cb_srcB_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_srcC_index = get_compile_time_arg_val(2);
    constexpr uint32_t cb_zero_index = get_compile_time_arg_val(3);

    // Signal that all input tiles are available (data already in L1 from sharding)
    cb_reserve_back(cb_srcA_index, num_tiles);
    cb_push_back(cb_srcA_index, num_tiles);

    cb_reserve_back(cb_srcB_index, num_tiles);
    cb_push_back(cb_srcB_index, num_tiles);

    cb_reserve_back(cb_srcC_index, num_tiles);
    cb_push_back(cb_srcC_index, num_tiles);

    // Generate zero tile for compute kernel
    cb_reserve_back(cb_zero_index, 1);
    const uint32_t zero_tile_addr = get_write_ptr(cb_zero_index);
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_zero_index);

    // Zero out the tile
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(zero_tile_addr);
    for (uint32_t i = 0; i < tile_size_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = 0;  // How fast is this? will it work?
    }
    cb_push_back(cb_zero_index, 1);
}
