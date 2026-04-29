// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in1 reader and output writer kernel
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Each worker handles B/num_workers batches independently
// Input B (weights) is DRAM sharded by batch - each bank has B/12 complete [N, K] matrices
// Output is NOC written to OUTPUT STORAGE CORES (different from worker cores)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

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

    // Output storage core coordinates and L1 address (where to NOC write output)
    const uint32_t output_storage_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t output_storage_noc_y = get_arg_val<uint32_t>(6);
    const uint32_t output_shard_l1_addr = get_arg_val<uint32_t>(7);

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
    constexpr uint32_t out_shard_size_bytes = get_compile_time_arg_val(9);           // full output shard size

#ifdef FUSE_BIAS
    constexpr uint32_t in3_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t in3_num_pages = get_compile_time_arg_val(11);
    constexpr uint32_t in3_block_tiles = get_compile_time_arg_val(12);  // K tiles for bias
    constexpr uint32_t cb_id_in3 = get_named_compile_time_arg_val("cb_bias");
#endif

    constexpr uint32_t cb_id_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_id_out = get_named_compile_time_arg_val("cb_out");  // Local output CB (compute writes here)
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr uint32_t out_single_tile_size_bytes = get_tile_size(cb_id_out);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
    constexpr uint32_t out_block_size_bytes = out_block_num_tiles * out_single_tile_size_bytes;

    experimental::Noc noc;
    experimental::CircularBuffer cb_in1(cb_id_in1);
    experimental::CircularBuffer cb_out(cb_id_out);
    // DRAM read setup
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_bank;
    // Output reshard setup - build NOC address for remote output storage core
    experimental::UnicastEndpoint remote;
#ifdef FUSE_BIAS
    experimental::CircularBuffer cb_in3(cb_id_in3);
#endif

    // Process each batch
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        uint32_t in1_batch_offset = batch * in1_tensor_stride_batch_bytes;
        uint32_t l1_read_addr_in1 = 0;

        // Read all N blocks of weights for this batch
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_in1.reserve_back(in1_block_num_tiles);

            // Read weight block from DRAM
            uint32_t remaining_bytes = in1_block_size_bytes;
            uint32_t cb_write_offset = 0;
            uint32_t curr_dram_offset = l1_read_addr_in1;

            while (remaining_bytes > 0) {
                uint32_t read_size = (remaining_bytes > in1_page_size) ? in1_page_size : remaining_bytes;
                noc.async_read(
                    dram_bank,
                    cb_in1,
                    read_size,
                    {.bank_id = dram_bank_id, .addr = in1_tensor_addr + in1_batch_offset + curr_dram_offset},
                    {.offset_bytes = cb_write_offset});
                cb_write_offset += read_size;
                curr_dram_offset += read_size;
                remaining_bytes -= read_size;
            }

            noc.async_read_barrier();
            cb_in1.push_back(in1_block_num_tiles);
            l1_read_addr_in1 += in1_block_size_bytes;
        }

#ifdef FUSE_BIAS
        // Read bias for this batch (if fused)
        cb_in3.reserve_back(in3_block_tiles);
        noc.async_read(
            dram_bank,
            cb_in3,
            in3_block_tiles * get_tile_size(cb_id_in3),
            {.bank_id = dram_bank_id, .addr = in3_tensor_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in3.push_back(in3_block_tiles);
#endif

        // Wait for compute to finish this batch
        cb_out.wait_front(out_block_num_tiles);

#ifdef OUT_SHARDED
        // NOC write output to remote output storage core (CB6)
        uint32_t out_batch_offset = batch * out_tensor_stride_batch_bytes;
        noc.async_write(
            experimental::use<experimental::CircularBuffer::AddrSelector::READ_PTR>(cb_out),
            remote,
            out_block_size_bytes,
            {.offset_bytes = 0},
            {.noc_x = output_storage_noc_x,
             .noc_y = output_storage_noc_y,
             .addr = output_shard_l1_addr + out_batch_offset});
        noc.async_write_barrier();
#endif

        cb_out.pop_front(out_block_num_tiles);
    }
}
