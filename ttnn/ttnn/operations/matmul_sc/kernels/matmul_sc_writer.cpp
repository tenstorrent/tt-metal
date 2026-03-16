// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Writer Kernel
// Runs on NCRISC (RISCV_1), writes C tiles from CB to DRAM via NoC1.
//
// Compile-time args (named):
//   arg0: cb_out (= 16) -- CB index for matrix C tiles
//
// Compile-time args (positional, after named):
//   TensorAccessorArgs for C (at offset 0)
//
// Runtime args:
//   [0] out_addr -- DRAM base address of C
//   [1] Mt       -- tile rows of C
//   [2] Nt       -- tile columns of C
//   [3] batch    -- always 1 for rank-2

#include "api/dataflow/dataflow_api.h"

// NOTE: matmul_1d_dataflow_helpers.hpp included here for future use:
// #include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"

void kernel_main() {
    // Stub: real implementation will call:
    //   dataflow_kernel_lib::write_matmul_tiles<cb_out>(out_addr, Mt, Nt, batch)
    // TODO: Implement in Stage 1 (data_pipeline) and Stage 2 (matmul_compute)
}
