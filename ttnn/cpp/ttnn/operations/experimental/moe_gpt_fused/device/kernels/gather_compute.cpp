// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"

//=============================================================================
// Gather Compute Kernel
//
// In the initial TILE-format DRAM implementation, this is a passthrough.
// The input is already in TILE format, so no tilization is needed.
//
// When ROW_MAJOR DRAM support is added, this kernel will perform:
//   fast_tilize_block() to convert ROW_MAJOR -> TILE format
//=============================================================================

void kernel_main() {
    // Passthrough: input is already in TILE format in DRAM.
    // No compute operation needed.
    //
    // Future: When ROW_MAJOR input is supported:
    // 1. Wait for reader to push ROW_MAJOR data into input CB
    // 2. Tilize using fast_tilize_block()
    // 3. Push tilized data to output CB for multicast
}
