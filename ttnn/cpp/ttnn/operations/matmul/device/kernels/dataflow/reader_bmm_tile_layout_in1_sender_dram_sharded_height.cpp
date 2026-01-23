// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in1 reader and output writer kernel
// For batched matmul: [1, B, M, N] x [1, B, N, K] = [1, B, M, K]
// Each worker handles B/num_workers batches independently
// Input B (weights) is DRAM sharded by batch - each bank has B/12 complete [N, K] matrices
// Output is written back to L1 sharded by batch

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // RUNTIME ARGS
    const bool is_worker_core = get_arg_val<uint32_t>(0) == 1;
    if (not is_worker_core) {
        return;
    }

    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(1);
#ifdef FUSE_BIAS
    const uint32_t in3_tensor_addr = get_arg_val<uint32_t>(2);
#endif
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);

    // COMPILE TIME ARGS
    constexpr uint32_t in1_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(2);                    // K tiles per block
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(3);            // in0_block_w * K
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);                     // N / in0_block_w
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(5);            // M * K
    constexpr uint32_t num_batches_per_core = get_compile_time_arg_val(6);           // B / num_cores
    constexpr uint32_t in1_tensor_stride_batch_bytes = get_compile_time_arg_val(7);  // bytes per batch in in1
    constexpr uint32_t out_tensor_stride_batch_bytes = get_compile_time_arg_val(8);  // bytes per batch in output

#ifdef FUSE_BIAS
    constexpr uint32_t in3_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t in3_num_pages = get_compile_time_arg_val(10);
    constexpr uint32_t in3_block_tiles = get_compile_time_arg_val(11);  // K tiles for bias
    constexpr uint32_t cb_id_in3 = 3;
#endif

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_4;
    constexpr uint32_t cb_id_out_reshard = tt::CBIndex::c_6;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr uint32_t out_single_tile_size_bytes = get_tile_size(cb_id_out);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    // DRAM read setup
    uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_tensor_addr);

    // Process each batch
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        uint32_t in1_batch_offset = batch * in1_tensor_stride_batch_bytes;
        uint32_t l1_read_addr_in1 = 0;

        // Read all N blocks of weights for this batch
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);
            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            // Read weight block from DRAM
            uint64_t dram_read_addr = in1_base_addr + in1_batch_offset + l1_read_addr_in1;

            uint32_t remaining_bytes = in1_block_size_bytes;
            uint32_t curr_l1_addr = l1_write_addr_in1;
            uint32_t curr_dram_offset = l1_read_addr_in1;

            while (remaining_bytes > 0) {
                uint32_t read_size = (remaining_bytes > in1_page_size) ? in1_page_size : remaining_bytes;
                noc_async_read(in1_base_addr + in1_batch_offset + curr_dram_offset, curr_l1_addr, read_size);
                curr_l1_addr += read_size;
                curr_dram_offset += read_size;
                remaining_bytes -= read_size;
            }

            noc_async_read_barrier();
            cb_push_back(cb_id_in1, in1_block_num_tiles);
            l1_read_addr_in1 += in1_block_size_bytes;
        }

#ifdef FUSE_BIAS
        // Read bias for this batch (if fused)
        cb_reserve_back(cb_id_in3, in3_block_tiles);
        uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3);
        uint64_t in3_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in3_tensor_addr);
        // Bias might be shared across batches or per-batch - read accordingly
        noc_async_read(in3_addr, l1_write_addr_in3, in3_block_tiles * get_tile_size(cb_id_in3));
        noc_async_read_barrier();
        cb_push_back(cb_id_in3, in3_block_tiles);
#endif

        // Wait for compute to finish this batch
        cb_wait_front(cb_id_out, out_block_num_tiles);

#ifndef SKIP_WRITE_BACK
        // Write output to sharded L1 buffer
        uint32_t l1_read_addr_out = get_read_ptr(cb_id_out);
        uint32_t l1_write_addr_out = get_write_ptr(cb_id_out_reshard) + batch * out_tensor_stride_batch_bytes;

        // Copy output to reshard buffer (local write for batch-sharded - no cross-core transfer)
        noc_async_write(
            l1_read_addr_out, get_noc_addr(l1_write_addr_out), out_block_num_tiles * out_single_tile_size_bytes);
        noc_async_write_barrier();
#endif

        cb_pop_front(cb_id_out, out_block_num_tiles);
    }
}
