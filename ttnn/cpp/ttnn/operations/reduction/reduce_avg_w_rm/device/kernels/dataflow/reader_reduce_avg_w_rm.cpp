// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for reduce_avg_w_rm operation
// Reads row-major input sticks and generates the scaler tile (1/W) for averaging.
//
// Per Kernel Design Document:
// - Phase 1: Generate scaler tile (once per kernel) using generate_reduce_scaler()
// - Phase 2: For each tile row, read 32 consecutive sticks into CB_rm_in

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // Compile-time args (from spec)
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);           // Size of one input row in bytes
    constexpr uint32_t packed_scaler_value = get_compile_time_arg_val(1);  // Two bfloat16 scaler values packed (1/W)
    constexpr auto src_tensor_args = TensorAccessorArgs<2>();              // TensorAccessor args start at index 2

    // Runtime args (from spec)
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);

    // CB indices
    constexpr uint32_t cb_rm_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;

    // Calculate Wt from stick_size and tile dimensions
    // Each tile is 32 elements wide. We need to know how many tiles span the input width.
    // tile_bytes = tile_size from the CB, but we work with sticks here.
    // stick_size = W * element_size
    // For row-major data in CB_rm_in, each tile-worth of data is 32 sticks of stick_size bytes.
    // The CB page is sized as a tile, but we're reading sticks.

    // Wt = stick_size / (32 * element_size) but we don't have element_size directly.
    // Actually, for the CB operations, the tile size in CB determines page size.
    // We know: input has W elements per stick, and W = Wt * 32
    // The CB has page_size = tile_size (for tiles)
    //
    // Key insight: The factory sets CB_rm_in with tile_size pages.
    // We read 32 sticks (each stick_size bytes) per tile row.
    // The tilize operation expects Wt tiles worth of row-major data.
    //
    // Total bytes per tile row = 32 sticks * stick_size = 32 * W * element_size
    // This equals Wt tiles * tile_size where tile_size = 32*32*element_size
    // So: 32 * W * element_size = Wt * 32 * 32 * element_size
    // Therefore: W = Wt * 32, confirming Wt = W / 32
    //
    // For CB operations: we reserve/push Wt tiles worth of space.
    // But we need Wt - we can compute it from tile_size vs stick_size.
    const uint32_t tile_bytes = get_tile_size(cb_rm_in);
    // Wt = (32 sticks * stick_size) / tile_bytes = 32 * stick_size / tile_bytes
    // But we need to be careful: tile_bytes = 32*32*element_size (for standard tiles)
    // stick_size = W * element_size = Wt * 32 * element_size
    // So: Wt = stick_size / (32 * element_size) = stick_size * 32 / tile_bytes
    // Wait, let me reconsider.
    //
    // tile_bytes = 32 * 32 * element_size (assuming standard tile)
    // stick_size = W * element_size = Wt * 32 * element_size
    // So: Wt = stick_size / (32 * element_size) = stick_size / (tile_bytes / 32)
    //       = stick_size * 32 / tile_bytes
    const uint32_t Wt = (stick_size * 32) / tile_bytes;

    // Initialize TensorAccessor for stick-based reads
    // Each page is one stick (stick_size bytes)
    const auto src_accessor = TensorAccessor(src_tensor_args, src_addr, stick_size);

    // =========================================================================
    // Phase 1: Generate scaler tile (once per kernel invocation)
    // The scaler contains 1/W to convert SUM to AVG
    // generate_reduce_scaler handles cb_reserve_back and cb_push_back internally
    // =========================================================================
    generate_reduce_scaler(cb_scaler, packed_scaler_value);

    // =========================================================================
    // Phase 2: Read sticks for each tile row
    // For each tile row (32 consecutive sticks), read all sticks into CB_rm_in
    // =========================================================================
    constexpr uint32_t TILE_HEIGHT = 32;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // Calculate the base stick index for this tile row
        uint32_t stick_base = (start_tile_row + tile_row) * TILE_HEIGHT;

        // Reserve space for Wt tiles in CB_rm_in
        cb_reserve_back(cb_rm_in, Wt);

        // Get the L1 write address
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);

        // Read 32 sticks (one tile height) from DRAM
        for (uint32_t stick = 0; stick < TILE_HEIGHT; ++stick) {
            uint32_t global_stick_id = stick_base + stick;
            uint64_t noc_addr = src_accessor.get_noc_addr(global_stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }

        // Wait for all reads to complete
        noc_async_read_barrier();

        // Push Wt tiles worth of data to CB_rm_in
        cb_push_back(cb_rm_in, Wt);
    }
}
