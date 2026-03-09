// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Reader Kernel (STUB)
// Runs on BRISC (RISCV_0), reads data from DRAM to L1 circular buffers via NOC0.
//
// Responsibilities:
//   1. Generate cb_scaler (1/W) and cb_eps tiles once at program start
//   2. Optionally load cb_gamma and cb_beta tiles once at program start
//   3. Per tile-row (Ht iterations): load Wt input tiles into cb_input
//
// Full implementation includes:
//   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
//   #include "api/tensor/tensor_accessor.h"

#include "api/dataflow/dataflow_api.h"

// Compile-time args (set by program descriptor):
//   Index 0: Ht          -- height in tiles
//   Index 1: Wt          -- width in tiles
//   Index 2: has_gamma   -- 1 if gamma is provided
//   Index 3: has_beta    -- 1 if beta is provided
//   Index 4+: TensorAccessorArgs for input tensor

constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

// CB indices
constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_eps = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_beta = 4;

void kernel_main() {
    // Stub: Real implementation will:
    //   1. Read runtime args: input_addr, gamma_addr, beta_addr, eps_u32
    //   2. generate_reduce_scaler(cb_scaler, 1.0f/W)       generates 1/W scaler tile
    //   3. generate_bcast_scalar_bfloat16(cb_eps, eps)     generates eps constant tile
    //   4. If has_gamma: read Wt tiles from DRAM -> cb_gamma using TensorAccessor
    //   5. If has_beta:  read Wt tiles from DRAM -> cb_beta using TensorAccessor
    //   6. Main loop (Ht times):
    //        read Wt input tiles from DRAM -> cb_input using TensorAccessor
}
