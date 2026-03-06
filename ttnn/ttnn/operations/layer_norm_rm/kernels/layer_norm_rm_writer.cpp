// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes RM sticks from L1 to DRAM via NOC1
//
// Stage 1 stub: minimal implementation.
// Full implementation will:
//   Per tile-row: wait for Wt pages in cb_out (from untilize),
//   extract 32 RM sticks via get_read_ptr(cb_out) + row*stick_size offsets,
//   write each stick to DRAM output via TensorAccessor + noc_async_write,
//   barrier per block, pop Wt pages from cb_out.

#include "api/dataflow/dataflow_api.h"
// TensorAccessor is available via dataflow_api.h (includes api/tensor/tensor_accessor.h)

void kernel_main() {
    // Compile-time args:
    //   [0] cb_id_out          - Output CB index (CB_OUT = 16)
    //   [1] output_stick_size  - Bytes per output stick
    //   [2] tile_height        - 32
    //   [3] num_tiles_per_row  - Wt
    //   [4+] output TensorAccessor compile-time args

    // Runtime args:
    //   [0] dst_addr           - Output buffer base address
    //   [1] N                  - Tile-rows for this core
    //   [2] start_stick_id     - First output stick ID

    // Stub: no-op
    // Real implementation will write untilized output sticks to DRAM.
}
