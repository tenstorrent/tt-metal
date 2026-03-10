// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel (STUB)
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Responsibilities:
//   - Read RM sticks from DRAM into c_0 (Wt tile-pages per block)
//   - Fill c_8 with reduce scaler (1/W) via prepare_reduce_scaler
//   - Fill c_9 with epsilon constant via prepare_reduce_scaler
//   - Optional: read gamma/beta sticks, replicate 32x, signal compute to tilize

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Stub: reader kernel - will be implemented in TDD stages
    // Real implementation reads RM sticks from DRAM into c_0,
    // fills scaler/epsilon CBs, and optionally reads gamma/beta.
}
