// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in0 reader kernel
// For batched matmul: [1, B, M, N] x [1, B, N, K] = [1, B, M, K]
// Each worker handles B/num_workers batches independently
// Input A is L1 sharded by batch - each core has B/num_cores complete [M, N] matrices

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    // COMPILE TIME ARGS
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);   // tiles per block (M * in0_block_w)
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);  // bytes per block
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);            // N / in0_block_w (K blocks in inner loop)
    constexpr uint32_t num_batches_per_core = get_compile_time_arg_val(3);  // B / num_cores
    constexpr uint32_t in0_tensor_stride_batch_bytes = get_compile_time_arg_val(4);  // bytes per batch in in0

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg_val<uint32_t>(0);
    if (worker_core_type == 0) {
        return;  // idle core
    }

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;  // Sharded CB for in0

    // Read from sharded L1 buffer and push to compute CB
    uint32_t l1_read_addr = get_read_ptr(cb_id_in2);

    // Process each batch
    DPRINT << "num_batches_per_core: " << num_batches_per_core << ENDL();
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        uint32_t batch_read_addr = l1_read_addr + batch * in0_tensor_stride_batch_bytes;

        // Process K blocks within each batch
        // DPRINT << "num_blocks: " << num_blocks << ENDL();
        // DPRINT << "in0_block_num_tiles: " << in0_block_num_tiles << ENDL();
        // DPRINT << "in0_block_size_bytes: " << in0_block_size_bytes << ENDL();

        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            // Copy block from sharded buffer to compute CB
            uint64_t src_noc_addr = get_noc_addr(batch_read_addr);
            noc_async_read(src_noc_addr, l1_write_addr, in0_block_size_bytes);
            noc_async_read_barrier();

            cb_push_back(cb_id_in0, in0_block_num_tiles);
            // DPRINT << "Reader BMM Tile Layout In0 Sender Dram Sharded Height" << ENDL();
            // print_full_tile(cb_id_in0, 0);
            // print_full_tile(cb_id_in0, 1);
            // This looks normal - identity matrix and then zeros
            batch_read_addr += in0_block_size_bytes;
        }
    }
}
