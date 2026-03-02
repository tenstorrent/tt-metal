// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Writer Kernel (STUB)
// Runs on RISCV_1 (NCRISC), writes normalized output tiles to DRAM via NOC1.
//
// Compile-time args:
//   [0+] output_accessor_args : TensorAccessorArgs for output tensor
//
// Runtime args:
//   [0] output_addr   : uint32_t - Output tensor base address
//   [1] num_rows      : uint32_t - Number of tile-rows
//   [2] Wt            : uint32_t - Width in tiles
//   [3] start_tile_id : uint32_t - Starting output tile index

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stub: writer kernel for layer_norm
    // Real implementation will:
    // For each row:
    //   1. cb_wait_front(cb_out, Wt) - wait for Wt output tiles
    //   2. Write Wt tiles to DRAM output buffer via TensorAccessor
    //   3. cb_pop_front(cb_out, Wt)
}
