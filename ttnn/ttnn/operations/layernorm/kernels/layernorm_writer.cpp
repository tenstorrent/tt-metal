// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Writer Kernel
// Writes RM output sticks from cb_out (c_17) to DRAM via TensorAccessor.
// Per tile-row: waits for Wt tiles in cb_out, extracts 32 sticks,
// writes each stick via noc_async_write, then pops cb_out.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"

void kernel_main() {
    // Stub: writer kernel
    // Real implementation will:
    // 1. For each tile-row block:
    //    a. cb_wait_front(cb_out, Wt)
    //    b. Extract 32 sticks from L1 read pointer
    //    c. Write each stick via TensorAccessor + noc_async_write
    //    d. noc_async_write_barrier()
    //    e. cb_pop_front(cb_out, Wt)
}
