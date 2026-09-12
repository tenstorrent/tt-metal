// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in0 reader kernel
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Each worker handles B/num_workers batches independently
// Input A is L1 sharded by batch on INPUT STORAGE CORES
// Workers are on OPTIMAL DRAM READER CORES (different from storage cores)
// Workers NOC read their in0 shard from their corresponding input storage core

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // COMPILE TIME ARGS
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);    // tiles per block (M * in0_block_w)
    constexpr auto in0_block_size_bytes = get_arg(args::in0_block_size_bytes);  // bytes per block
    constexpr auto num_blocks = get_arg(args::num_blocks);  // K / in0_block_w (K blocks in inner loop)
    constexpr auto num_batches_per_core = get_arg(args::num_batches_per_core);  // B / num_cores
    constexpr auto in0_tensor_stride_batch_bytes =
        get_arg(args::in0_tensor_stride_batch_bytes);                           // bytes per batch in in0
    constexpr auto in0_shard_size_bytes = get_arg(args::in0_shard_size_bytes);  // full shard size in bytes

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg(args::worker_core_type);
    if (worker_core_type == 0) {
        return;  // idle core
    }

    // Get the input storage core coordinates and L1 address (where in0 shard is located)
    const uint32_t input_storage_noc_x = get_arg(args::input_storage_noc_x);
    const uint32_t input_storage_noc_y = get_arg(args::input_storage_noc_y);
    // in0's shard base address. The storage core and this worker are the same core (the factory
    // asserts the two orderings match element-wise), and an L1 shard sits at the same offset on
    // every core holding one, so the local base is also the remote core's base.
    const uint32_t input_shard_l1_addr = TensorAccessor(tensor::in0).get_bank_base_address();

    // Build NOC address for the remote input storage core
    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    const UnicastEndpoint src_core;

    // Process each batch
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        const uint32_t batch_offset = batch * in0_tensor_stride_batch_bytes;

        // Process K blocks within each batch
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in0.reserve_back(in0_block_num_tiles);

            // NOC read block from REMOTE input storage core to the local dataflow buffer
            const uint32_t read_offset = batch_offset + (block * in0_block_size_bytes);
            noc.async_read(
                src_core,
                dfb_in0,
                in0_block_size_bytes,
                {.noc_x = input_storage_noc_x, .noc_y = input_storage_noc_y, .addr = input_shard_l1_addr + read_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();

            dfb_in0.push_back(in0_block_num_tiles);
        }
    }
}
