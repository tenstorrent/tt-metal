// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * TopK Multicore Reader Kernel - Local Core with Pre-existing Index Tensor
 *
 * This kernel variant runs on local processing cores when an index tensor is already
 * provided as input (rather than generating indices on-demand). It handles:
 * 1. Reading the assigned width chunk of input tensor values from DRAM
 * 2. Reading the corresponding pre-existing index tensor from DRAM
 * 3. Streaming both to the local compute kernel for bitonic sorting
 *
 * Use Case:
 * - When input tensors already have associated index information
 * - Supports scenarios where element tracking is more complex than simple position indices
 * - Enables chaining of TopK operations while preserving original index mappings
 */
void kernel_main() {
    // Runtime arguments - core-specific work assignment and tensor addresses
    uint32_t src_addr = get_arg_val<uint32_t>(0);          // DRAM address of input values tensor
    uint32_t src_indices_addr = get_arg_val<uint32_t>(1);  // DRAM address of input indices tensor
    uint32_t start_ht = get_arg_val<uint32_t>(2);          // Starting height tile index
    uint32_t start_wt = get_arg_val<uint32_t>(3);          // Starting width tile index for this core

    // Compile-time circular buffer configuration
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);  // Input values circular buffer
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);  // Input indices circular buffer

    // Tensor dimension configuration
    constexpr uint32_t Ht = get_compile_time_arg_val(2);        // Total height tiles in tensor
    constexpr uint32_t Wt_local = get_compile_time_arg_val(3);  // Width tiles assigned to this core
    constexpr uint32_t Wt = get_compile_time_arg_val(4);        // Total width tiles in tensor

    // DRAM tensor accessor configuration for both input tensors
    constexpr auto s_input_args = TensorAccessorArgs<5>();
    constexpr auto s_indices_args = TensorAccessorArgs<s_input_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t tile_bytes_indices = get_tile_size(cb_id_in1);

    const auto s_input = TensorAccessor(s_input_args, src_addr, tile_bytes);
    const auto s_indices = TensorAccessor(s_indices_args, src_indices_addr, tile_bytes_indices);

    // MAIN DATA STREAMING LOOP
    // Stream both input values and pre-existing indices from DRAM to local circular buffers.
    // Unlike the index generation variant, this kernel reads both tensors from DRAM,
    // which requires careful coordination to maintain synchronization between values and indices.
    //
    // Memory Access Pattern:
    // - Parallel reads of values and indices at the same (i,j) positions
    // - Double-buffered to hide DRAM latency during compute operations
    // - Linear access pattern within height rows for optimal memory bandwidth
    for (uint32_t i = start_ht; i < Ht; ++i) {                       // For each height row
        for (uint32_t j = start_wt; j < start_wt + Wt_local; ++j) {  // For each width tile in chunk
            // Simultaneously stream both input values and indices from DRAM
            cb_reserve_back(cb_id_in0, onetile);  // Reserve space for values
            cb_reserve_back(cb_id_in1, onetile);  // Reserve space for indices

            uint32_t l1_write_addr_input = get_write_ptr(cb_id_in0);
            uint32_t l1_write_addr_indices = get_write_ptr(cb_id_in1);

            // Issue parallel DRAM reads for values and indices at position (i,j)
            noc_async_read_tile(i * Wt + j, s_input, l1_write_addr_input);
            noc_async_read_tile(i * Wt + j, s_indices, l1_write_addr_indices);
            noc_async_read_barrier();  // Ensure both reads complete before proceeding

            // Make data available to compute kernel
            cb_push_back(cb_id_in0, onetile);
            cb_push_back(cb_id_in1, onetile);
        }
    }
}
