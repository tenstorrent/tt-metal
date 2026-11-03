// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in0 reader kernel
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Each worker handles B/num_workers batches independently
// Input A is L1 sharded by batch on the input storage cores
// Workers are the optimal DRAM reader cores, which the factory requires to be exactly those same
// storage cores, so a worker's NOC read of its in0 shard targets its own L1

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // COMPILE TIME ARGS
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);    // tiles per block (M * in0_block_w)
    constexpr auto in0_block_size_bytes = get_arg(args::in0_block_size_bytes);  // bytes per block
    constexpr auto num_blocks = get_arg(args::num_blocks);  // K / in0_block_w (K blocks in inner loop)
    constexpr auto num_batches_per_core = get_arg(args::num_batches_per_core);  // B / num_cores
    constexpr auto in0_tensor_stride_batch_bytes =
        get_arg(args::in0_tensor_stride_batch_bytes);  // bytes per batch in in0

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg(args::worker_core_type);
    if (worker_core_type == 0) {
        return;  // idle core
    }

    // in0's L1 shard. The factory requires the input storage core to be this worker core, so the
    // shard is this core's own L1 and is read as a loopback NoC source (input_storage_noc_x/y name
    // this core and are not needed).
    const LocalTensorAccessor<uint32_t> in0_shard(tensor::in0);

    const Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);

    // Process each batch
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        const uint32_t batch_offset = batch * in0_tensor_stride_batch_bytes;

        // Process K blocks within each batch
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in0.reserve_back(in0_block_num_tiles);

            // NOC read block from the input storage core (this core) to the local dataflow buffer
            const uint32_t read_offset = batch_offset + (block * in0_block_size_bytes);
            noc.async_read(
                in0_shard, dfb_in0, in0_block_size_bytes, {.offset_bytes = read_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();

            dfb_in0.push_back(in0_block_num_tiles);
        }
    }
}
