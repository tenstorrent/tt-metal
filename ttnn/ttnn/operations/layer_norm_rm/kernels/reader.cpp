// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel (STUB)
//
// Runs on RISCV_0 (BRISC), reads data from DRAM via NOC0.
//
// Responsibilities:
//   1. One-time setup:
//      - Generate reduce scaler tile (1/W) into CB_SCALER (c_2)
//      - Generate epsilon tile into CB_EPS (c_3)
//      - Read gamma stick 32 times into CB_GAMMA_RM (c_6)  [if has_gamma]
//      - Read beta stick 32 times into CB_BETA_RM (c_7)    [if has_beta]
//   2. Per-tile-row loop (num_sticks / 32 iterations):
//      - Read 32 RM sticks into CB_IN_RM (c_0) as Wt tile-sized pages
//
// Compile-time args:
//   [0] stick_size_bytes     : W * element_size (bytes per RM stick)
//   [1] gamma_stick_size     : same as stick_size_bytes
//   [2] has_gamma            : 1 if gamma tensor present
//   [3] has_beta             : 1 if beta tensor present
//   [4+] TensorAccessorArgs  : for input tensor
//   [N+] TensorAccessorArgs  : for gamma tensor (if has_gamma)
//   [M+] TensorAccessorArgs  : for beta tensor (if has_beta)
//
// Runtime args:
//   [0] src_addr             : input buffer base address
//   [1] gamma_addr           : gamma buffer address (0 if no gamma)
//   [2] beta_addr            : beta buffer address (0 if no beta)
//   [3] num_sticks           : total RM sticks this core processes
//   [4] start_stick_id       : first stick index for this core
//   [5] scaler_value         : 1/W as bfloat16-packed uint32
//   [6] eps_value            : epsilon as bfloat16-packed uint32

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // STUB: No-op implementation.
    // The kernel-writer TDD agent will implement the full reader logic:
    //   - prepare_reduce_scaler for c_2 and c_3
    //   - gamma/beta RM stick reads (32 repetitions each)
    //   - per-tile-row RM stick reads into c_0
}
