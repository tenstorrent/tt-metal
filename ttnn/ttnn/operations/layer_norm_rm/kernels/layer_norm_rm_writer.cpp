// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel (stub)
// Runs on RISCV_1 (NCRISC), writes RM sticks to DRAM via NOC1.
//
// Compile-time args:
//   [0]    stick_size        - bytes per output RM stick (W * sizeof(bfloat16))
//   [1..]  TensorAccessorArgs(output)
//
// Runtime args:
//   [0] dst_addr        - output buffer DRAM address
//   [1] num_rows        - total tile-rows to process
//   [2] Wt              - tiles per row (W / 32)
//   [3] start_stick_id  - first output stick id for this core
//
// CB usage:
//   c_16 (cb_out_rm) - consume Wt pages per tile-row (32 RM sticks from compute)
//
// Pattern per tile-row (block_idx):
//   cb_wait_front(c_16, Wt)
//   for stick in 0..31:
//       page_id = start_stick_id + block_idx * 32 + stick
//       noc_async_write(l1_ptr, dst_noc_addr(page_id), stick_size)
//       l1_ptr += stick_size
//   noc_async_write_barrier()
//   cb_pop_front(c_16, Wt)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stub: no-op
    // Real implementation drains c_16 and writes 32 sticks per block to DRAM.
}
