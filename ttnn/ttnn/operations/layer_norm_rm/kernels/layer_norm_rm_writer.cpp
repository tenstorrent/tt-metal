// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel (STUB)
// Runs on RISCV_1 (NCRISC), writes RM output sticks to DRAM via NOC1
//
// When implemented, this kernel will:
// 1. Wait for Wt pages in c_16
// 2. Extract 32 RM sticks per block
// 3. Write each stick to DRAM via TensorAccessor
// 4. Pop c_16 and repeat for each block

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // Stub: no data movement, kernels will be implemented in TDD stages
}
