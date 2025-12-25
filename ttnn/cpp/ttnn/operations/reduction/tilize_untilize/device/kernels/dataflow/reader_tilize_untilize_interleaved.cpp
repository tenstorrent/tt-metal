// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * Reader kernel for tilize_untilize operation.
 * Reads row-major sticks (32 rows at a time) from DRAM into CB_in (c_0).
 *
 * Operation selection via compile-time OpType argument and if constexpr:
 * - OpType::IDENTITY: No auxiliary CB setup needed
 * - Future: REDUCE_W_* operations will generate scaler tile in CB_scaler (c_2)
 *
 * Compile-time args:
 *   [0] stick_size - Size of each row in bytes
 *   [1] op_type - Operation type enum value
 *   [2] packed_scaler - Packed scaler value (used by reduction operations)
 *   [3+] TensorAccessorArgs for source tensor
 *
 * Runtime args:
 *   [0] src_addr - Source buffer address
 *   [1] num_sticks - Number of rows to read
 *   [2] start_stick_id - Starting row index
 */

// Operation type enum - must match host-side and compute kernel definition
enum class OpType : uint32_t {
    IDENTITY = 0,
    // Future:
    // REDUCE_W_SUM = 1,
    // REDUCE_W_MAX = 2,
    // REDUCE_W_AVG = 3,
    // RELU = 4,
};

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr OpType op_type = static_cast<OpType>(get_compile_time_arg_val(1));
    constexpr uint32_t packed_scaler = get_compile_time_arg_val(2);  // Used by reduction operations
    constexpr auto src_tensor_args = TensorAccessorArgs<3>();        // Starts after op_type and packed_scaler

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // CB indices
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;  // For reduction operations
    constexpr uint32_t tile_height = 32;

    // Calculate num_tiles_per_row from stick_size
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t element_size = 2;  // BF16
    constexpr uint32_t num_tiles_per_row = stick_size / (tile_width * element_size);

    // ========== Auxiliary CB Setup (operation-specific) ==========
    if constexpr (op_type == OpType::IDENTITY) {
        // No auxiliary CB setup needed for identity operation
    }
    // Future: reduction operations would generate scaler here
    // else if constexpr (op_type == OpType::REDUCE_W_SUM || ...) {
    //     generate_reduce_scaler(cb_scaler, packed_scaler);
    // }

    // ========== Core Data Reading (shared by all operations) ==========
    const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

    uint32_t stick_id = start_stick_id;
    uint32_t num_blocks = num_sticks / tile_height;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Reserve space in CB for num_tiles_per_row tiles
        cb_reserve_back(cb_id_in0, num_tiles_per_row);

        // Get write pointer
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        // Read 32 rows (full stick width each)
        for (uint32_t row = 0; row < tile_height; row++) {
            noc_async_read(get_noc_addr(stick_id + row, s), l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }

        // Wait for all reads to complete
        noc_async_read_barrier();

        // Push num_tiles_per_row tiles to compute
        cb_push_back(cb_id_in0, num_tiles_per_row);

        // Advance to next block of 32 sticks
        stick_id += tile_height;
    }
}
