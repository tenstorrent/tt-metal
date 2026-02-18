// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Reader Kernel
//
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Startup (once):
//   1. Fill CB c_8 (cb_scaler) with reduce scaler tile (1/W as packed bf16)
//   2. Fill CB c_7 (cb_eps) with epsilon scalar tile (eps as packed bf16)
//
// Per tile-row (Ht_total iterations):
//   3. Reserve CB c_0 for Wt tiles worth of space
//   4. Read 32 sticks (each stick_size bytes) from DRAM
//   5. Barrier and push to CB c_0
//
// Compile-time args:
//   [0] stick_size        - W * 2 (bytes per RM stick)
//   [1] cb_rm_in          - CB c_0 ID
//   [2] cb_scaler         - CB c_8 ID
//   [3] cb_eps            - CB c_7 ID
//   [4+] TensorAccessorArgs(src)
//
// Runtime args:
//   [0] src_addr          - Input buffer DRAM base address
//   [1] num_sticks        - Total sticks to read (Ht_total * 32)
//   [2] Wt               - Tiles per tile-row
//   [3] start_stick_id    - First stick index (0 for single-core)
//   [4] packed_reduce_scaler - 1/W as (bf16 << 16 | bf16)
//   [5] packed_eps        - epsilon as (bf16 << 16 | bf16)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr auto src_tensor_args = TensorAccessorArgs<4>();

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t packed_reduce_scaler = get_arg_val<uint32_t>(4);
    const uint32_t packed_eps = get_arg_val<uint32_t>(5);

    // ========== TensorAccessor for input ==========
    const auto src_accessor = TensorAccessor(src_tensor_args, src_addr, stick_size);

    // ========== One-time setup: generate scaler and epsilon tiles ==========
    // Generate reduce scaler tile in cb_scaler (1/W)
    dataflow_kernel_lib::generate_reduce_scaler(cb_scaler, packed_reduce_scaler);

    // Generate epsilon scalar tile in cb_eps
    dataflow_kernel_lib::generate_bcast_scalar_bfloat16(cb_eps, packed_eps);

    // ========== Per tile-row: read 32 RM sticks into cb_rm_in ==========
    constexpr uint32_t TILE_H = 32;
    uint32_t num_tile_rows = num_sticks / TILE_H;
    uint32_t stick_id = start_stick_id;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // Reserve Wt pages (tile-sized) in cb_rm_in
        cb_reserve_back(cb_rm_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);

        // Read 32 sticks (one tile-row)
        for (uint32_t s = 0; s < TILE_H; ++s) {
            uint64_t noc_addr = src_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_rm_in, Wt);
    }
}
