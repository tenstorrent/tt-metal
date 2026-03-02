// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Reader Kernel (STUB)
// Runs on RISCV_0 (BRISC), reads input/gamma/beta from DRAM via NOC0,
// fills scaler and epsilon CBs.
//
// Compile-time args:
//   [0] gamma_has_value  : uint32_t - 1 if gamma is provided
//   [1] beta_has_value   : uint32_t - 1 if beta is provided
//   [2+] input_accessor_args : TensorAccessorArgs for input
//   [N+] gamma_accessor_args : TensorAccessorArgs for gamma (if present)
//   [M+] beta_accessor_args  : TensorAccessorArgs for beta (if present)
//
// Runtime args:
//   [0] input_addr    : uint32_t - Input tensor base address
//   [1] gamma_addr    : uint32_t - Gamma base address (0 if absent)
//   [2] beta_addr     : uint32_t - Beta base address (0 if absent)
//   [3] num_rows      : uint32_t - Number of tile-rows (N*C*Ht)
//   [4] Wt            : uint32_t - Width in tiles
//   [5] start_tile_id : uint32_t - Starting tile index
//   [6] eps_u32       : uint32_t - Epsilon packed as IEEE 754 float bits

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stub: reader kernel for layer_norm
    // Real implementation will:
    // 1. Fill cb_scaler with 1/Wt (reduce scaler for REDUCE_ROW AVG)
    // 2. Fill cb_eps with epsilon value as a tile
    // 3. For each row: read Wt input tiles from DRAM into cb_input
    // 4. If gamma: read Wt gamma tiles from DRAM into cb_gamma per row
    // 5. If beta: read Wt beta tiles from DRAM into cb_beta per row
}
