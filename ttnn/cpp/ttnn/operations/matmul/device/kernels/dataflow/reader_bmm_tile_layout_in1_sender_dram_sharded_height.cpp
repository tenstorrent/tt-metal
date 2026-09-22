// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // RUNTIME ARGS
    const bool is_worker_core = get_arg(args::is_worker_core) == 1;
    if (not is_worker_core) {
        return;
    }

    const uint32_t in1_tensor_addr = TensorAccessor(tensor::in1).get_bank_base_address();
#ifdef FUSE_BIAS
    const uint32_t in3_tensor_addr = TensorAccessor(tensor::bias).get_bank_base_address();
#endif
    const uint32_t dram_bank_id = get_arg(args::dram_bank_id);
    const uint32_t vc = get_arg(args::vc);

    // Output storage core coordinates and L1 address (where to NOC write output)
    const uint32_t output_storage_noc_x = get_arg(args::output_storage_noc_x);
    const uint32_t output_storage_noc_y = get_arg(args::output_storage_noc_y);
    const uint32_t output_shard_l1_addr = TensorAccessor(tensor::output).get_bank_base_address();

    // COMPILE TIME ARGS
    constexpr auto in1_page_size = get_arg(args::in1_page_size);
    constexpr auto in1_num_pages = get_arg(args::in1_num_pages);
    constexpr auto in1_block_w = get_arg(args::in1_block_w);                    // K tiles per block
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);    // in0_block_w * K
    constexpr auto num_blocks = get_arg(args::num_blocks);                      // N / in0_block_w
    constexpr auto out_block_num_tiles = get_arg(args::out_block_num_tiles);    // M * K
    constexpr auto num_batches_per_core = get_arg(args::num_batches_per_core);  // B / num_cores
    constexpr auto in1_tensor_stride_batch_bytes =
        get_arg(args::in1_tensor_stride_batch_bytes);  // bytes per batch in in1
    constexpr auto out_tensor_stride_batch_bytes =
        get_arg(args::out_tensor_stride_batch_bytes);                           // bytes per batch in output
    constexpr auto out_shard_size_bytes = get_arg(args::out_shard_size_bytes);  // full output shard size

#ifdef FUSE_BIAS
    constexpr auto in3_page_size = get_arg(args::in3_page_size);
    constexpr auto in3_num_pages = get_arg(args::in3_num_pages);
    constexpr auto in3_block_tiles = get_arg(args::in3_block_tiles);  // K tiles for bias
#endif

    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(dfb::in1);
    constexpr uint32_t out_single_tile_size_bytes = get_tile_size(dfb::out);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
    constexpr uint32_t out_block_size_bytes = out_block_num_tiles * out_single_tile_size_bytes;

    const Noc noc;
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);  // Local output buffer (compute writes here)
    // DRAM read setup
    const AllocatorBank<AllocatorBankType::DRAM> dram_bank;
    // Output reshard setup - build NOC address for remote output storage core
    const UnicastEndpoint remote;
#ifdef FUSE_BIAS
    DataflowBuffer dfb_in3(dfb::bias);
#endif

    // Process each batch
    for (uint32_t batch = 0; batch < num_batches_per_core; ++batch) {
        const uint32_t in1_batch_offset = batch * in1_tensor_stride_batch_bytes;
        uint32_t l1_read_addr_in1 = 0;

        // Read all N blocks of weights for this batch
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in1.reserve_back(in1_block_num_tiles);

            // Read weight block from DRAM
            uint32_t remaining_bytes = in1_block_size_bytes;
            uint32_t dfb_write_offset = 0;
            uint32_t curr_dram_offset = l1_read_addr_in1;

            while (remaining_bytes > 0) {
                const uint32_t read_size = (remaining_bytes > in1_page_size) ? in1_page_size : remaining_bytes;
                noc.async_read(
                    dram_bank,
                    dfb_in1,
                    read_size,
                    {.bank_id = dram_bank_id, .addr = in1_tensor_addr + in1_batch_offset + curr_dram_offset},
                    {.offset_bytes = dfb_write_offset});
                dfb_write_offset += read_size;
                curr_dram_offset += read_size;
                remaining_bytes -= read_size;
            }

            noc.async_read_barrier();
            dfb_in1.push_back(in1_block_num_tiles);
            l1_read_addr_in1 += in1_block_size_bytes;
        }

#ifdef FUSE_BIAS
        // Read bias for this batch (if fused)
        dfb_in3.reserve_back(in3_block_tiles);
        noc.async_read(
            dram_bank,
            dfb_in3,
            in3_block_tiles * dfb_in3.get_tile_size(),
            {.bank_id = dram_bank_id, .addr = in3_tensor_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in3.push_back(in3_block_tiles);
#endif

        // Wait for compute to finish this batch
        dfb_out.wait_front(out_block_num_tiles);

#ifdef OUT_SHARDED
        // NOC write output into the output tensor's own L1 shard on the remote output storage core
        const uint32_t out_batch_offset = batch * out_tensor_stride_batch_bytes;
        noc.async_write(
            dfb_out,
            remote,
            out_block_size_bytes,
            {.offset_bytes = 0},
            {.noc_x = output_storage_noc_x,
             .noc_y = output_storage_noc_y,
             .addr = output_shard_l1_addr + out_batch_offset});
        noc.async_write_barrier();
#endif

        dfb_out.pop_front(out_block_num_tiles);
    }
}
