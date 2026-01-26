// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in0 reader kernel
// For batched matmul: [1, B, M, N] x [1, B, N, K] = [1, B, M, K]
// Each worker handles B/num_workers batches independently
// Input A is L1 sharded by batch on INPUT STORAGE CORES
// Workers are on OPTIMAL DRAM READER CORES (different from storage cores)
// Workers NOC read their in0 shard from their corresponding input storage core

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
    constexpr uint32_t in0_shard_size_bytes = get_compile_time_arg_val(5);           // full shard size in bytes

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg_val<uint32_t>(0);
    if (worker_core_type == 0) {
        return;  // idle core
    }

    // Get the input storage core coordinates and L1 address (where in0 shard is located)
    const uint32_t input_storage_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t input_storage_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t input_shard_l1_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;

    // Build NOC address for the remote input storage core
    uint64_t remote_shard_base_noc_addr = get_noc_addr(input_storage_noc_x, input_storage_noc_y, input_shard_l1_addr);

    // Process each batch
    DPRINT << "in0 reader: NOC reading from storage core (" << input_storage_noc_x << ", " << input_storage_noc_y << ")"
           << ENDL();
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        uint32_t batch_offset = batch * in0_tensor_stride_batch_bytes;

        // Process K blocks within each batch
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

            // NOC read block from REMOTE input storage core to local CB
            uint32_t read_offset = batch_offset + block * in0_block_size_bytes;
            uint64_t src_noc_addr = remote_shard_base_noc_addr + read_offset;
            noc_async_read(src_noc_addr, l1_write_addr, in0_block_size_bytes);
            noc_async_read_barrier();

            cb_push_back(cb_id_in0, in0_block_num_tiles);
        }
    }
}
