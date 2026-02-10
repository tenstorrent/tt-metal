// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Compile-time arguments:
// 0: output_stick_size (W * element_size)
// 1: tile_height (32)
// 2: num_tile_rows (total number of tile-rows to process)
// 3: Wt (tiles along W dimension)
// 4+: TensorAccessorArgs for output

// Runtime arguments:
// 0: dst_addr (output buffer DRAM base address)

void kernel_main() {
    // This is a stub writer kernel.
    // Real implementation will:
    // 1. Loop over tile-rows:
    //    - Wait for untilized output sticks in CB 16 (Wt tiles worth)
    //    - Write 32 row-major sticks to DRAM
    //    - Pop CB 16
    //
    // For now, this stub does nothing to allow compilation.
}
