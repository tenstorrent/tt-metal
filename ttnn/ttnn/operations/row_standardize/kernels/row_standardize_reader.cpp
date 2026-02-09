// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Standardize - Reader Kernel
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0
//
// Responsibilities:
// 1. Generate reduce scaler tile (1/W) once at start -> cb_scaler (c_1)
// 2. Generate epsilon scalar tile once at start -> cb_eps (c_2)
// 3. For each block: read 32 RM sticks from DRAM into cb_rm_in (c_0)
//
// Compile-time args:
//   0: stick_size_bytes - Size of one RM stick (W * datum_size)
//   1: is_float32 - 1 if float32, 0 if bfloat16
//   2+: TensorAccessorArgs (src)
//
// Runtime args:
//   0: src_addr - Source buffer base address in DRAM
//   1: num_sticks - Total number of sticks to read (nblocks * 32)
//   2: start_stick_id - First stick ID for this core (0 for single-core)
//   3: Wt - Tiles per row
//   4: scaler - Reduce scaler (1/W) as packed uint32
//   5: epsilon - Epsilon as packed uint32

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t is_float32 = get_compile_time_arg_val(1);
    constexpr auto src_tensor_args = TensorAccessorArgs<2>();

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(4);
    const uint32_t epsilon_packed = get_arg_val<uint32_t>(5);

    // ========== CB indices ==========
    constexpr uint32_t cb_rm_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_eps = tt::CBIndex::c_2;

    // ========== Constants ==========
    constexpr uint32_t tile_height = 32;

    // ========== TensorAccessor for input ==========
    const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size_bytes);

    // ========== One-time setup: Generate scaler tile (1/W) ==========
    // generate_reduce_scaler expects packed bf16: (bf16 << 16 | bf16) for bfloat16
    // or float bits reinterpreted as uint32 for float32
    dataflow_kernel_lib::generate_reduce_scaler(cb_scaler, scaler_packed);

    // ========== One-time setup: Generate epsilon scalar tile ==========
    // For SCALAR broadcast: value at position [0][0] of face 0 only
    if constexpr (is_float32) {
        dataflow_kernel_lib::generate_bcast_scalar(cb_eps, epsilon_packed);
    } else {
        dataflow_kernel_lib::generate_bcast_scalar_bfloat16(cb_eps, epsilon_packed);
    }

    // ========== Per-block loop: Read 32 RM sticks per block ==========
    uint32_t stick_id = start_stick_id;
    const uint32_t nblocks = num_sticks / tile_height;

    // Pre-compute NOC addresses for 32 sticks per block
    uint64_t base_src_noc_addr[tile_height];

    for (uint32_t block = 0; block < nblocks; ++block) {
        // Pre-compute NOC addresses for all 32 sticks in this block
        for (uint32_t j = 0; j < tile_height; ++j) {
            base_src_noc_addr[j] = get_noc_addr(stick_id + j, s);
        }

        // Reserve Wt pages in cb_rm_in (32 sticks = Wt tiles worth of data)
        cb_reserve_back(cb_rm_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_in);

        // Read 32 sticks sequentially
        for (uint32_t j = 0; j < tile_height; ++j) {
            noc_async_read(base_src_noc_addr[j], l1_write_addr, stick_size_bytes);
            l1_write_addr += stick_size_bytes;
        }

        noc_async_read_barrier();
        cb_push_back(cb_rm_in, Wt);

        stick_id += tile_height;
    }
}
