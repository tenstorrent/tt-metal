// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Runs on RISCV_0 (BRISC), reads data from DRAM via NOC0
//
// This kernel:
// 1. Generates reduce scaler tile (1/W) -> c_1
// 2. Generates epsilon scalar tile -> c_7
// 3. Reads gamma sticks and replicates to 32 rows -> c_3
// 4. Reads beta sticks and replicates to 32 rows -> c_4
// 5. For each tile-row: reads 32 input sticks -> c_0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement reader kernel
    // This is a stub implementation that will be replaced by the kernel-writer agent
}
