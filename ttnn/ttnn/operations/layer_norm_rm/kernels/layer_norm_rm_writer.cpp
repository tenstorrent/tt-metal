// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel (Stub)
// Runs on NCRISC (RISCV_1), writes RM sticks to DRAM via NOC1
//
// Waits for Wt tiles in c_16 per block, extracts 32 RM sticks,
// writes to interleaved RM output via TensorAccessor.

#include "api/dataflow/dataflow_api.h"
// TensorAccessor is provided by dataflow_api.h

void kernel_main() {
    // Stub: writer kernel - will be implemented in TDD stages
}
