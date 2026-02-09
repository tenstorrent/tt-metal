// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Standardize - Reader Kernel (STUB)
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0
//
// Responsibilities:
// 1. Generate reduce scaler tile (1/W) once at start
// 2. Generate epsilon scalar tile once at start
// 3. For each block: read 32 RM sticks from DRAM into cb_rm_in
//
// Compile-time args:
//   0: stick_size_bytes - Size of one RM stick (W * datum_size)
//   1: is_float32 - 1 if float32, 0 if bfloat16
//   2+: TensorAccessorArgs (src)
//
// Runtime args:
//   0: src_addr - Source buffer base address in DRAM
//   1: num_sticks - Total number of sticks to read (nblocks * 32)
//   2: start_stick_id - First stick ID for this core (0 for single-core)
//   3: Wt - Tiles per row
//   4: scaler - Reduce scaler (1/W) as packed uint32
//   5: epsilon - Epsilon as packed uint32

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // STUB: Real implementation will:
    // 1. Generate scaler and epsilon tiles using generate_reduce_scaler() and generate_bcast_scalar()
    // 2. Loop over blocks, reading 32 sticks per block using TensorAccessor
    // 3. Push sticks to cb_rm_in for tilize phase
}
