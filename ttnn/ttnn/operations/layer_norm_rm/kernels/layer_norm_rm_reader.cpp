// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Compile-time arguments:
// 0: stick_size (W * element_size)
// 1: gamma_beta_stick_size (same as stick_size)
// 2+: TensorAccessorArgs for input
// N+: TensorAccessorArgs for gamma
// M+: TensorAccessorArgs for beta

// Runtime arguments:
// 0: src_addr (input buffer DRAM base address)
// 1: gamma_addr (gamma buffer DRAM base address, 0 if no gamma)
// 2: beta_addr (beta buffer DRAM base address, 0 if no beta)
// 3: num_sticks (total number of input sticks)
// 4: num_tile_rows (number of tile-rows to process)
// 5: Wt (tiles along W)
// 6: reduce_scaler (packed bfloat16 value of 1/W)
// 7: eps_scalar (packed scalar epsilon value)

void kernel_main() {
    // This is a stub reader kernel.
    // Real implementation will:
    // 1. Generate reduce scaler tile into CB 6
    // 2. Generate epsilon scalar tile into CB 7
    // 3. Read gamma sticks and push to CB 2 (if gamma provided)
    // 4. Read beta sticks and push to CB 4 (if beta provided)
    // 5. Loop over tile-rows:
    //    - Read 32 row-major sticks (one tile-row) from DRAM
    //    - Push to CB 0
    //
    // For now, this stub does nothing to allow compilation.
}
