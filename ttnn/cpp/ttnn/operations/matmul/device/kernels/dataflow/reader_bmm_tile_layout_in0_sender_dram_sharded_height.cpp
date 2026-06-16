// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in0 reader kernel
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Each worker handles B/num_workers batches independently
// Input A is L1 sharded by batch on INPUT STORAGE CORES
// Workers are on OPTIMAL DRAM READER CORES (different from storage cores)
// Workers NOC read their in0 shard from their corresponding input storage core
//
// Metal 2.0: only the access mechanism changes. The in0 CB id -> dfb::cb_in0; positional
// CT/RT args -> get_arg(args::...). The in0 shard's L1 base address used to arrive as a raw
// RTA (the resolved buffer address); it is now a Case-2 binding (this is a data-movement
// kernel performing explicit remote-NoC address arithmetic) — `a` is bound as a tensor and
// the base is pulled via TensorAccessor::get_bank_base_address(), keeping the raw NoC walk
// to the remote input storage core unchanged.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // COMPILE TIME ARGS
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);    // tiles per block (M * in0_block_w)
    constexpr uint32_t in0_block_size_bytes = get_arg(args::in0_block_size_bytes);  // bytes per block
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);  // K / in0_block_w (K blocks in inner loop)
    constexpr uint32_t num_batches_per_core = get_arg(args::num_batches_per_core);  // B / num_cores
    constexpr uint32_t in0_tensor_stride_batch_bytes =
        get_arg(args::in0_tensor_stride_batch_bytes);                               // bytes per batch in in0
    constexpr uint32_t in0_shard_size_bytes = get_arg(args::in0_shard_size_bytes);  // full shard size in bytes

    // RUNTIME ARGS
    const uint32_t worker_core_type = get_arg(args::worker_core_type);
    if (worker_core_type == 0) {
        return;  // idle core
    }

    // Get the input storage core coordinates (where in0 shard is located)
    const uint32_t input_storage_noc_x = get_arg(args::input_storage_noc_x);
    const uint32_t input_storage_noc_y = get_arg(args::input_storage_noc_y);

    constexpr uint32_t cb_id_in0 = dfb::cb_in0;

    // The in0 shard L1 address now comes from the typed tensor binding (Case-2 bridge), not a
    // raw RTA. For an L1-sharded tensor this returns the local L1 base of the shard region,
    // which is the same address the legacy RTA carried.
    const auto in0 = TensorAccessor(ta::a);
    const uint32_t input_shard_l1_addr = in0.get_bank_base_address();

    // Build NOC address for the remote input storage core
    Noc noc;
    DataflowBuffer cb_in0(cb_id_in0);
    UnicastEndpoint src_core;

    // Process each batch
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        uint32_t batch_offset = batch * in0_tensor_stride_batch_bytes;

        // Process K blocks within each batch
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_in0.reserve_back(in0_block_num_tiles);

            // NOC read block from REMOTE input storage core to local CB
            uint32_t read_offset = batch_offset + block * in0_block_size_bytes;
            noc.async_read(
                src_core,
                cb_in0,
                in0_block_size_bytes,
                {.noc_x = input_storage_noc_x, .noc_y = input_storage_noc_y, .addr = input_shard_l1_addr + read_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();

            cb_in0.push_back(in0_block_num_tiles);
        }
    }
}
