// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Writer Kernel
//
// Streams Wt output tiles from c_16 to DRAM for each tile-row.
//
// Compile-time args (via TensorAccessorArgs):
//   [0+] : TensorAccessorArgs for output tensor
//
// Runtime args:
//   [0] output_addr       : uint32  Output buffer base address
//   [1] num_rows_per_core : uint32  Number of tile-rows for this core
//   [2] Wt               : uint32  Width in tiles
//   [3] tile_offset      : uint32  Starting tile index in the output buffer

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement writer kernel
    // Stub: do nothing.
}
