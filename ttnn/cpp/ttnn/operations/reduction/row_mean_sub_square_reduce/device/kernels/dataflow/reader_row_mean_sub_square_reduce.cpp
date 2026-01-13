// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

/**
 * Reader kernel for row_mean_sub_square_reduce operation
 *
 * Per the design document:
 * - Generate scaler (1/W) once at start for mean computation
 * - Read 32 sticks (one tile-row worth) from DRAM per tile-row
 * - Push Wt pages to signal compute (32 sticks = Wt tiles worth of data)
 *
 * CB Flow:
 * - CB c_0 (rm_in): Reserve Wt pages, read 32 sticks, push Wt pages per tile-row
 * - CB c_2 (scaler): Generate once, persists for entire program
 */
void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);           // Input row width in bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(1);                   // Tiles per row (padded_W / 32)
    constexpr uint32_t packed_scaler_value = get_compile_time_arg_val(2);  // 1/W packed as bfloat16
    constexpr auto src_tensor_args = TensorAccessorArgs<3>();              // TensorAccessor args start at index 3

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);   // Tile-rows to process
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);  // First tile-row for this core

    constexpr uint32_t cb_rm_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;

    // Setup TensorAccessor (page_size = stick_size for row-major input)
    const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

    // Phase 1: Generate scaler (1/W) once at start
    // The scaler CB persists for entire program duration
    generate_reduce_scaler(cb_scaler, packed_scaler_value);

    // Phase 2: Read row-major sticks per tile-row
    // Each tile-row has 32 sticks (TILE_HEIGHT = 32)
    constexpr uint32_t TILE_HEIGHT = 32;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // Reserve Wt pages in c_0 (each page = tile_size bytes)
        // 32 sticks * stick_size = Wt tiles worth of data
        cb_reserve_back(cb_rm_in, Wt);

        // Get write pointer to CB
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);

        // Calculate starting stick ID for this tile-row
        uint32_t base_stick_id = (start_tile_row + tile_row) * TILE_HEIGHT;

        // Read 32 sticks contiguously into CB
        for (uint32_t j = 0; j < TILE_HEIGHT; ++j) {
            uint32_t stick_id = base_stick_id + j;
            uint64_t noc_addr = s.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }

        // Barrier after all 32 sticks are read
        noc_async_read_barrier();

        // Push Wt pages to signal compute
        cb_push_back(cb_rm_in, Wt);
    }
}
