// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block_id = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);

    // Initialize interleaved address generator for DRAM access
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // Process each block of data
    uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t i = start_block_id; i < end_block_id; ++i) {
        // Reserve space in the circular buffer for a row of tiles
        cb_reserve_back(cb_id_in0, ntiles_per_row);
        uint64_t l1_write_addr = get_write_ptr(cb_id_in0);

        // Read each tile in the current row
        for (uint32_t j = 0; j < ntiles_per_row; ++j) {
            // Calculate tile index and read from DRAM to L1
            noc_async_read_tile(ntiles_per_row * i + j, s, l1_write_addr);
            l1_write_addr += tile_bytes;
        }

        // Ensure all async reads are complete before proceeding
        noc_async_read_barrier();

        // Push the completed row of tiles to the circular buffer
        cb_push_back(cb_id_in0, ntiles_per_row);
    }
}
