// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
// Reads RM sticks from DRAM into cb_in (c_0).
// On first tile-row, also reads gamma/beta into cb_gamma/cb_beta.
// Generates reduce scaler tile and epsilon tile.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"

void kernel_main() {
    // Stub: reader kernel
    // Real implementation will:
    // 1. Read 32 RM sticks per tile-row into cb_in
    // 2. On first iteration, read gamma/beta as RM sticks into cb_gamma/cb_beta
    // 3. Generate reduce scaler tile (1/W) into cb_reduce_scaler
    // 4. Generate epsilon tile into cb_eps
}
