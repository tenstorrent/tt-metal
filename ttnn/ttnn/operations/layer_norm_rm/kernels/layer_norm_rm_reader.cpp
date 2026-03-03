// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel (stub)
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Compile-time args:
//   [0]      stick_size           - bytes per RM input stick (W * sizeof(bfloat16))
//   [1..]    TensorAccessorArgs(input)
//   [N+0]    gamma_stick_size     - bytes per gamma stick
//   [N+1..]  TensorAccessorArgs(gamma)
//   [M+0]    beta_stick_size      - bytes per beta stick
//   [M+1..]  TensorAccessorArgs(beta)
//
// Runtime args:
//   [0] src_addr        - input buffer DRAM address
//   [1] gamma_addr      - gamma buffer DRAM address
//   [2] beta_addr       - beta buffer DRAM address
//   [3] num_rows        - total tile-rows to process
//   [4] Wt              - tiles per row (W / 32)
//   [5] start_stick_id  - first stick id for this core
//   [6] scaler_value    - 1/W as packed bfloat16 uint32
//   [7] eps_value       - epsilon as packed bfloat16 uint32
//
// CB usage:
//   c_0  (cb_input_rm)  - push Wt pages per tile-row (32 RM sticks per row)
//   c_1  (cb_gamma_rm)  - push Wt pages once at start (32 gamma sticks)
//   c_2  (cb_beta_rm)   - push Wt pages once at start (32 beta sticks)
//   c_8  (cb_scaler)    - push 1 tile (1/W scaler, never popped)
//   c_9  (cb_eps)       - push 1 tile (epsilon scaler, never popped)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Stub: no-op
    // Real implementation reads gamma, beta, scaler, eps once;
    // then for each tile-row reads 32 input sticks into c_0.
}
