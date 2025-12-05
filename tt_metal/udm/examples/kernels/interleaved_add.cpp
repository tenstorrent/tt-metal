// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Kernel: Interleaved Add
 *
 * Each worker:
 * 1. Receives: buffer_addr, block_page_start, num_tiles, device_id
 * 2. Creates a BlockTensorAccessor
 * 3. For each tile:
 *    - Use BlockTensorAccessor to get NOC address from block page ID
 *    - Read tile
 *    - Add 1 to each element
 *    - Write tile back
 */

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/hw/inc/accessor/block_tensor_accessor.h"
#include "tt_metal/hw/inc/accessor/tensor_accessor.h"

void kernel_main() {
    // Get runtime args
    uint32_t arg_idx = 0;
    const uint32_t buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t block_page_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);

    // Create TensorAccessor from compile-time args
    // TODO: Properly construct from TensorAccessorArgs
    // For now, using simplified interleaved accessor
    constexpr uint32_t page_size = 32 * 32 * 2;  // 32×32 tile, UINT16

    using DSpec = InterleavedAddrGenFast<true>;  // DRAM, interleaved
    // Create BlockTensorAccessor wrapping the TensorAccessor
    BlockTensorAccessor<DSpec> block_accessor(buffer_addr, page_size);

    // Process each tile using BlockTensorAccessor
    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t page_id = page_start + i;

        // Get NOC address using BlockTensorAccessor
        // This internally converts block_page_id → local_page_id
        uint64_t noc_addr = block_accessor.get_noc_addr(page_id);
        block_accessor.async_read_tile(page_id, cb_id_in);

        // push to compute for plus one operation
    }
}
