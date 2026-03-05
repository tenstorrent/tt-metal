// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Reads untilized RM sticks from c_16 and writes them to DRAM output buffer.
//
// Runtime args (per core):
//   [0] dst_addr       - Output buffer base address
//   [1] num_blocks     - Number of tile-rows for this core
//   [2] start_stick_id - First output stick index
//
// Compile-time args:
//   [0] stick_size     - W * element_size bytes
//   [1] Wt             - Tiles per row
//   [2+] TensorAccessorArgs(output)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stub: Real implementation extracts 32 RM sticks per tile-row and writes to DRAM
    // TODO: implement in kernel-writer stage
}
