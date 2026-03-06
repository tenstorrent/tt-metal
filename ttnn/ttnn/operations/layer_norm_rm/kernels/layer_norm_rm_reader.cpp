// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM to L1 CBs via NOC0
//
// Stage 1 stub: minimal implementation that signals completion.
// Full implementation will:
//   1. Read Wt RM sticks per tile-row from input into cb_in (Wt tile-sized pages)
//   2. On first tile-row: generate reduce scaler (1/W) into cb_reduce_scaler
//   3. On first tile-row: generate epsilon scalar tile into cb_eps
//   4. On first tile-row: read gamma Wt tiles into cb_gamma (once, never popped)
//   5. On first tile-row: read beta Wt tiles into cb_beta (once, never popped)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
// TensorAccessor is available via dataflow_api.h (includes api/tensor/tensor_accessor.h)

void kernel_main() {
    // Compile-time args:
    //   [0] stick_size       - Width of one RM stick in bytes
    //   [1] Wt               - Tiles per row
    //   [2] has_gamma        - 1 if gamma tensor provided
    //   [3] has_beta         - 1 if beta tensor provided
    //   [4+] input TensorAccessor compile-time args

    // Runtime args:
    //   [0] src_addr         - Input buffer base address
    //   [1] N                - Tile-rows for this core
    //   [2] start_stick_id   - First stick ID for this core
    //   [3] scaler_packed    - 1/W as packed bfloat16 u32
    //   [4] eps_packed       - epsilon as packed bfloat16 u32
    //   [5] gamma_addr       - Gamma buffer base address (0 if no gamma)
    //   [6] beta_addr        - Beta buffer base address (0 if no beta)

    // Stub: no-op
    // Real implementation will populate cb_in with Wt pages per tile-row,
    // and constant CBs (reduce_scaler, eps, gamma, beta) on first iteration.
}
