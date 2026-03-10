// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Writer Kernel (STUB)
//
// Extracts RM sticks from tile-sized CB pages (cb_out_rm) and writes them
// to DRAM. Per tile-row block:
//   1. cb_wait_front(cb_out_rm, Wt)
//   2. For each of 32 rows: extract stick and write to DRAM
//   3. noc_async_write_barrier()
//   4. cb_pop_front(cb_out_rm, Wt)
//
// Compile-time args:
//   [0]  output_stick_size  — W * sizeof(bfloat16) bytes
//   [1]  Wt                 — tiles per row
//   [2+] TensorAccessorArgs(output) — interleaved output accessor
//
// Runtime args:
//   [0] dst_addr       — output buffer address
//   [1] start_stick_id — first output RM stick for this core
//   [2] num_blocks     — number of tile-row blocks to write

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stub: no-op. The real implementation will extract RM sticks from
    // cb_out_rm tile-sized pages and write them back to DRAM.
}
