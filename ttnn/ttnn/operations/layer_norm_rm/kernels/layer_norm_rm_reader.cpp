// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel (stub)
// Reads RM sticks from DRAM into c_0 using TensorAccessor.
// Fills c_8 (reduce scaler) and c_9 (epsilon) at program start.
// Optionally reads gamma/beta sticks into c_2/c_3.

#include "api/dataflow/dataflow_api.h"
// Full implementation will also need:
// #include "api/tensor/tensor_accessor.h"
// #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Stub: reader kernel
    // Real implementation will:
    // 1. Fill c_8 with prepare_reduce_scaler(1/W)
    // 2. Fill c_9 with epsilon value
    // 3. Optionally read gamma/beta into c_2/c_3
    // 4. Main loop: read 32 RM sticks per tile-row into c_0
}
