// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel (STUB)
// Runs on RISCV_0 (BRISC), reads RM input sticks from DRAM via NOC0
//
// When implemented, this kernel will:
// 1. Read 32 RM input sticks per block into c_0 using TensorAccessor
// 2. Generate reduce scaler tile (1/W) into c_2
// 3. Generate epsilon tile into c_7
// 4. Optionally read gamma/beta sticks into c_27/c_28

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Stub: no data movement, kernels will be implemented in TDD stages
}
