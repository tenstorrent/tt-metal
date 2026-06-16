// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Batch-sharded DRAM matmul - in1 reader and output writer kernel
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Each worker handles B/num_workers batches independently
// Input B (weights) is DRAM sharded by batch - each bank has B/12 complete [N, K] matrices
// Output is NOC written to OUTPUT STORAGE CORES (different from worker cores)
//
// Metal 2.0: only the access mechanism changes. CB ids -> dfb::cb_in1 / dfb::cb_out /
// dfb::cb_bias; positional CT/RT args -> get_arg(args::...). The in1 (DRAM bank) base, the
// bias (DRAM bank) base, and the output shard's L1 base all used to arrive as raw RTAs (the
// resolved buffer addresses); they are now Case-2 bindings (this is a data-movement kernel
// performing explicit DRAM-bank and remote-NoC address arithmetic) — `b`, `bias`, and `out`
// are bound as tensors and their bases are pulled via TensorAccessor::get_bank_base_address(),
// keeping the raw DRAM-bank reads and remote-NoC output write unchanged.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // RUNTIME ARGS
    const bool is_worker_core = get_arg(args::is_worker_core) == 1;
    if (not is_worker_core) {
        return;
    }

    const uint32_t dram_bank_id = get_arg(args::dram_bank_id);
    const uint32_t vc = get_arg(args::vc);

    // Output storage core coordinates (where to NOC write output)
    const uint32_t output_storage_noc_x = get_arg(args::output_storage_noc_x);
    const uint32_t output_storage_noc_y = get_arg(args::output_storage_noc_y);

    // COMPILE TIME ARGS
    constexpr uint32_t in1_page_size = get_arg(args::in1_page_size);
    constexpr uint32_t in1_num_pages = get_arg(args::in1_num_pages);
    constexpr uint32_t in1_block_w = get_arg(args::in1_block_w);                    // K tiles per block
    constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);    // in0_block_w * K
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);                      // N / in0_block_w
    constexpr uint32_t out_block_num_tiles = get_arg(args::out_block_num_tiles);    // M * K
    constexpr uint32_t num_batches_per_core = get_arg(args::num_batches_per_core);  // B / num_cores
    constexpr uint32_t in1_tensor_stride_batch_bytes =
        get_arg(args::in1_tensor_stride_batch_bytes);  // bytes per batch in in1
    constexpr uint32_t out_tensor_stride_batch_bytes =
        get_arg(args::out_tensor_stride_batch_bytes);                               // bytes per batch in output
    constexpr uint32_t out_shard_size_bytes = get_arg(args::out_shard_size_bytes);  // full output shard size

#ifdef FUSE_BIAS
    constexpr uint32_t in3_page_size = get_arg(args::in3_page_size);
    constexpr uint32_t in3_num_pages = get_arg(args::in3_num_pages);
    constexpr uint32_t in3_block_tiles = get_arg(args::in3_block_tiles);  // K tiles for bias
    constexpr uint32_t cb_id_in3 = dfb::cb_bias;
#endif

    constexpr uint32_t cb_id_in1 = dfb::cb_in1;
    constexpr uint32_t cb_id_out = dfb::cb_out;  // Local output CB (compute writes here)
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr uint32_t out_single_tile_size_bytes = get_tile_size(cb_id_out);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
    constexpr uint32_t out_block_size_bytes = out_block_num_tiles * out_single_tile_size_bytes;

    // Tensor base addresses now arrive via the typed tensor bindings (Case-2 bridge), not raw
    // RTAs. For DRAM-sharded `b`/`bias` this is the DRAM bank base; for the L1-sharded output
    // it is the remote output shard's local L1 base — the same addresses the legacy RTAs carried.
    const auto in1 = TensorAccessor(ta::b);
    const uint32_t in1_tensor_addr = in1.get_bank_base_address();
#ifdef FUSE_BIAS
    const auto in3 = TensorAccessor(ta::bias);
    const uint32_t in3_tensor_addr = in3.get_bank_base_address();
#endif
    const auto out = TensorAccessor(ta::out);
    const uint32_t output_shard_l1_addr = out.get_bank_base_address();

    Noc noc;
    DataflowBuffer cb_in1(cb_id_in1);
    DataflowBuffer cb_out(cb_id_out);
    // DRAM read setup
    AllocatorBank<AllocatorBankType::DRAM> dram_bank;
    // Output reshard setup - build NOC address for remote output storage core
    UnicastEndpoint remote;
#ifdef FUSE_BIAS
    DataflowBuffer cb_in3(cb_id_in3);
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
            cb_out,
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
