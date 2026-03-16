// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Writer Kernel
// Writes output tiles from cb_out to DRAM using the write_matmul_tiles helper.
//
// Compile-time args (positional):
//   TensorAccessorArgs for C starting at index 0
//
// Runtime args:
//   [0] out_addr -- DRAM base address of C
//   [1] Mt       -- tile rows of C
//   [2] Nt       -- tile columns of C
//   [3] batch    -- always 1 for rank-2

#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_writer_helpers.hpp"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);

    dataflow_kernel_lib::write_matmul_tiles<cb_out>(out_addr, Mt, Nt, batch);
}
