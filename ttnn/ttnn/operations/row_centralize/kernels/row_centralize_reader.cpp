// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Reader Kernel
//
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Startup (once):
//   1. Fill CB c_8 (cb_scaler) with reduce scaler tile (1/W as packed bf16)
//   2. Fill CB c_7 (cb_eps) with epsilon scalar tile (eps as packed bf16)
//
// Per tile-row (Ht_total iterations):
//   3. Reserve CB c_0 for Wt tiles worth of space
//   4. Read 32 sticks (each stick_size bytes) from DRAM
//   5. Barrier and push to CB c_0
//
// Compile-time args:
//   [0] stick_size        - W * 2 (bytes per RM stick)
//   [1] cb_rm_in          - CB c_0 ID
//   [2] cb_scaler         - CB c_8 ID
//   [3] cb_eps            - CB c_7 ID
//   [4+] TensorAccessorArgs(src)
//
// Runtime args:
//   [0] src_addr          - Input buffer DRAM base address
//   [1] num_sticks        - Total sticks to read (Ht_total * 32)
//   [2] Wt               - Tiles per tile-row
//   [3] start_stick_id    - First stick index (0 for single-core)
//   [4] packed_reduce_scaler - 1/W as (bf16 << 16 | bf16)
//   [5] packed_eps        - epsilon as (bf16 << 16 | bf16)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement reader kernel
    // Kernel writer agent will implement:
    //   1. Generate scaler tile in CB c_8 (cb_scaler) at startup
    //   2. Generate epsilon tile in CB c_7 (cb_eps) at startup
    //   3. For each tile-row: read 32 RM sticks from DRAM into CB c_0
}
