// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

//=============================================================================
// Gather Reader - No-op placeholder
//
// In the initial bring-up, dm0 reads input tiles directly from DRAM.
// This kernel will be implemented to handle sparse token gathering
// and multicast to matmul cores in a future iteration.
//=============================================================================

void kernel_main() {
    // No-op: dm0 handles input reading directly from DRAM
}
