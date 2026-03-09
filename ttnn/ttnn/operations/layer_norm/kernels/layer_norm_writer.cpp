// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Writer Kernel (STUB)
// Runs on NCRISC (RISCV_1), writes data from L1 circular buffers to DRAM via NOC1.
//
// Responsibilities:
//   Per tile-row (Ht iterations): drain Wt output tiles from cb_output to DRAM
//
// Full implementation includes:
//   #include "api/tensor/tensor_accessor.h"

#include "api/dataflow/dataflow_api.h"

// Compile-time args (set by program descriptor):
//   Index 0: Ht          -- height in tiles
//   Index 1: Wt          -- width in tiles
//   Index 2+: TensorAccessorArgs for output tensor

constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

// CB index
constexpr uint32_t cb_output = 16;

void kernel_main() {
    // Stub: Real implementation will:
    //   1. Read runtime arg: output_addr
    //   2. Build TensorAccessor for output tensor using compile-time args[2+]
    //   3. Main loop (Ht times):
    //        cb_wait_front(cb_output, Wt)
    //        for each tile in Wt: write tile from cb_output to DRAM via TensorAccessor
    //        cb_pop_front(cb_output, Wt)
}
