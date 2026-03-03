// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel (STUB)
//
// Runs on RISCV_1 (NCRISC), writes data to DRAM via NOC1.
//
// Responsibilities:
//   Per-tile-row loop:
//     - Wait for Wt pages in CB_OUT (c_16)
//     - Extract 32 RM sticks (each stick_size_bytes wide)
//     - Write each stick to DRAM via TensorAccessor and noc_async_write
//     - Pop Wt pages from CB_OUT
//
// Compile-time args:
//   [0] stick_size_bytes   : W * element_size (bytes per output RM stick)
//   [1+] TensorAccessorArgs: for output tensor
//
// Runtime args:
//   [0] dst_addr           : output buffer base address
//   [1] num_sticks         : total sticks this core writes
//   [2] start_stick_id     : first output stick index for this core

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // STUB: No-op implementation.
    // The kernel-writer TDD agent will implement the full writer logic:
    //   - TensorAccessor for output
    //   - per-tile-row loop: wait Wt pages, write 32 sticks, pop Wt pages
}
