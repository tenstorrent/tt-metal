// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel (Stub)
// Runs on BRISC (RISCV_0), reads RM sticks from DRAM via NOC0
//
// Reads 32 RM sticks per block into c_0.
// On first invocation generates constant tiles:
//   c_8:  reduce scaler (1/W) via prepare_reduce_scaler
//   c_10: epsilon scalar tile
// If gamma/beta present: reads their RM sticks into c_1/c_2 once at start.

#include "api/dataflow/dataflow_api.h"
// TensorAccessor is provided by dataflow_api.h
// reduce_helpers_dataflow.hpp will be added when implementing actual logic

void kernel_main() {
    // Stub: reader kernel - will be implemented in TDD stages
}
