// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Writer Kernel
//
// Runs on RISCV_1 (NCRISC), writes RM sticks to DRAM via NOC1.
//
// Per tile-row (Ht_total iterations):
//   1. Wait for Wt tiles in CB c_16 (one tile-row of RM output)
//   2. For each of 32 rows within the tile-row:
//      - Compute L1 read address within CB c_16
//      - Compute output page ID (stick ID) = tile_row_idx * TILE_H + row_in_tile_row
//      - Write output_stick_size bytes to DRAM via noc_async_write with TensorAccessor
//   3. Barrier after all 32 rows written
//   4. Pop Wt tiles from CB c_16
//
// Compile-time args:
//   [0] cb_rm_out         - CB c_16 ID
//   [1] output_stick_size - W * 2 (bytes per output RM stick)
//   [2] tile_height       - 32 (TILE_H)
//   [3] Wt               - tiles per tile-row
//   [4+] TensorAccessorArgs(dst)
//
// Runtime args:
//   [0] dst_addr          - Output buffer DRAM base address
//   [1] num_tile_rows     - Total tile-rows to write (Ht_total)
//   [2] start_tile_row    - First tile-row index (0 for single-core)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement writer kernel
    // Kernel writer agent will implement:
    //   For each tile-row: read 32 RM sticks from CB c_16 and write to DRAM
    //   using TensorAccessor for address computation
}
