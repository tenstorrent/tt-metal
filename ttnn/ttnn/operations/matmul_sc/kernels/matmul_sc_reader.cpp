// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Reader Kernel
// Reads A and B tiles from DRAM into cb_in0 and cb_in1 using
// the read_matmul_tiles helper (WaitPerTile order).
//
// Compile-time args (positional):
//   TensorAccessorArgs for A starting at index 0
//   TensorAccessorArgs for B chained after A
//
// Runtime args:
//   [0] in0_addr  -- DRAM base address of A
//   [1] in1_addr  -- DRAM base address of B
//   [2] Mt        -- tile rows of A/C
//   [3] Kt        -- inner dimension tiles
//   [4] Nt        -- tile columns of B/C
//   [5] batch     -- always 1 for rank-2

#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"

constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_in1 = 1;

void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(5);

    dataflow_kernel_lib::read_matmul_tiles<cb_in0, cb_in1>(in0_addr, in1_addr, Mt, Nt, Kt, batch);
}
