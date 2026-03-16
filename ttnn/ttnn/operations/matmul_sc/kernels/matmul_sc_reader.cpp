// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Reader Kernel
// Runs on BRISC (RISCV_0), reads A and B tiles from DRAM into CBs via NoC0.
//
// Compile-time args (named):
//   arg0: cb_in0 (= 0)   -- CB index for matrix A tiles
//   arg1: cb_in1 (= 1)   -- CB index for matrix B tiles
//
// Compile-time args (positional, after named):
//   TensorAccessorArgs for A (at offset 0)
//   TensorAccessorArgs for B (chained after A)
//
// Runtime args:
//   [0] in0_addr  -- DRAM base address of A
//   [1] in1_addr  -- DRAM base address of B
//   [2] Mt        -- tile rows of A/C
//   [3] Kt        -- inner dimension tiles
//   [4] Nt        -- tile columns of B/C
//   [5] batch     -- always 1 for rank-2

#include "api/dataflow/dataflow_api.h"

// NOTE: matmul_1d_dataflow_helpers.hpp included here for future use:
// #include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"

void kernel_main() {
    // Stub: real implementation will call:
    //   dataflow_kernel_lib::read_matmul_tiles<cb_in0, cb_in1>(in0_addr, in1_addr, Mt, Nt, Kt, batch)
    // TODO: Implement in Stage 1 (data_pipeline) and Stage 2 (matmul_compute)
}
