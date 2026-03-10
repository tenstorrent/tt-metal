// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel (STUB)
// Runs on RISCV_1 (NCRISC), writes untilized RM sticks to DRAM via NOC1.
//
// Responsibilities:
//   - Wait for untilized data in c_17 (Wt tile-pages per block)
//   - Extract 32 RM sticks from tile-pages
//   - Write each stick to output DRAM via TensorAccessor

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // Stub: writer kernel - will be implemented in TDD stages
    // Real implementation writes untilized RM sticks from c_17 to DRAM.
}
