// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Compute Kernel
// Runs on math RISC-V core, performs tiled matrix multiplication via FPU.
//
// Compile-time args (named):
//   arg0: cb_in0 (= 0)   -- CB index for matrix A tiles
//   arg1: cb_in1 (= 1)   -- CB index for matrix B tiles
//   arg2: cb_out  (= 16) -- CB index for output C tiles
//
// Runtime args:
//   [0] Mt    -- tile rows of C
//   [1] Kt    -- inner dimension tiles
//   [2] Nt    -- tile columns of C
//   [3] batch -- always 1 for rank-2

#include "api/compute/compute_kernel_hw_startup.h"

// NOTE: matmul_1d_helpers.hpp included here for future use:
// #include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"

void kernel_main() {
    // Stub: real implementation will call:
    //   compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);  // 3-arg form required
    //   compute_kernel_lib::matmul_1d<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt, batch);
    // TODO: Implement in Stage 1 (data_pipeline) and Stage 2 (matmul_compute)
}
