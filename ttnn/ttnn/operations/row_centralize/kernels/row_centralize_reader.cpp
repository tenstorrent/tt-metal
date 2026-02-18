// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Reader Kernel
//
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Startup (once):
//   1. Fill CB c_8 (cb_scaler) with reduce scaler tile (1/W as packed bf16)
//   2. Fill CB c_7 (cb_eps) with epsilon scalar tile (eps as packed bf16)
//   3. If has_affine: read gamma stick into c_0, push Wt tiles (compute tilizes -> c_9)
//   4. If has_affine: read beta stick into c_0, push Wt tiles (compute tilizes -> c_10)
//
// Per tile-row (Ht_total iterations):
//   5. Reserve CB c_0 for Wt tiles worth of space
//   6. Read 32 sticks (each stick_size bytes) from DRAM
//   7. Barrier and push to CB c_0
//
// Compile-time args:
//   [0] stick_size        - W * 2 (bytes per RM stick)
//   [1] cb_rm_in          - CB c_0 ID
//   [2] cb_scaler         - CB c_8 ID
//   [3] cb_eps            - CB c_7 ID
//   [4] has_affine        - 1 if gamma/beta provided, 0 otherwise
//   [5+] TensorAccessorArgs(src)
//   [next+] TensorAccessorArgs(gamma) — always present (dummy when no affine)
//   [next+] TensorAccessorArgs(beta)  — always present (dummy when no affine)
//
// Runtime args:
//   [0] src_addr          - Input buffer DRAM base address
//   [1] num_sticks        - Total sticks to read (Ht_total * 32)
//   [2] Wt               - Tiles per tile-row
//   [3] start_stick_id    - First stick index (0 for single-core)
//   [4] packed_reduce_scaler - 1/W as (bf16 << 16 | bf16)
//   [5] packed_eps        - epsilon as (bf16 << 16 | bf16)
//   [6] gamma_addr        - Gamma buffer DRAM base address (0 when no affine)
//   [7] beta_addr         - Beta buffer DRAM base address (0 when no affine)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t has_affine = get_compile_time_arg_val(4);
    constexpr auto src_tensor_args = TensorAccessorArgs<5>();
    // Gamma/beta TensorAccessorArgs always present (dummy values from host when no affine)
    constexpr auto gamma_tensor_args = TensorAccessorArgs<src_tensor_args.next_compile_time_args_offset()>();
    constexpr auto beta_tensor_args = TensorAccessorArgs<gamma_tensor_args.next_compile_time_args_offset()>();

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t packed_reduce_scaler = get_arg_val<uint32_t>(4);
    const uint32_t packed_eps = get_arg_val<uint32_t>(5);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);

    // ========== TensorAccessors ==========
    const auto src_accessor = TensorAccessor(src_tensor_args, src_addr, stick_size);

    // ========== One-time setup: generate scaler and epsilon tiles ==========
    dataflow_kernel_lib::generate_reduce_scaler(cb_scaler, packed_reduce_scaler);
    dataflow_kernel_lib::generate_bcast_scalar_bfloat16(cb_eps, packed_eps);

    // ========== Affine startup: read gamma and beta sticks into c_0 ==========
    if constexpr (has_affine) {
        const auto gamma_accessor = TensorAccessor(gamma_tensor_args, gamma_addr, stick_size);
        const auto beta_accessor = TensorAccessor(beta_tensor_args, beta_addr, stick_size);

        // Read 1 gamma stick (W*2 bytes = stick_size) into c_0, push Wt tiles
        // Gamma is shape (1,...,1,W), so only 1 stick (row 0).
        // Only row 0 of the tile-row matters; rows 1-31 are L1 garbage (irrelevant for ROW broadcast).
        cb_reserve_back(cb_rm_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);
        uint64_t gamma_noc_addr = gamma_accessor.get_noc_addr(0);
        noc_async_read(gamma_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_rm_in, Wt);

        // Read 1 beta stick into c_0 (after compute tilizes gamma and pops c_0)
        cb_reserve_back(cb_rm_in, Wt);
        l1_write_addr = get_write_ptr(cb_rm_in);
        uint64_t beta_noc_addr = beta_accessor.get_noc_addr(0);
        noc_async_read(beta_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_rm_in, Wt);
    }

    // ========== Per tile-row: read 32 RM sticks into cb_rm_in ==========
    constexpr uint32_t TILE_H = 32;
    uint32_t num_tile_rows = num_sticks / TILE_H;
    uint32_t stick_id = start_stick_id;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        cb_reserve_back(cb_rm_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);

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
