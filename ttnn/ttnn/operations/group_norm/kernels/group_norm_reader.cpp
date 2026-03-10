// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Reader Kernel (Stub)
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0
//
// Reads input RM sticks into cb_input_rm (CB 0) for tilize.
// Also reads gamma/beta tiles, fills eps and scaler CBs.
//
// Based on tilize reference reader pattern with TensorAccessor.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stub: kernel body will be implemented by kernel-writer agent.
    // The reader will:
    //   1. Read gamma tiles into cb_gamma (CB 2)
    //   2. Read beta tiles into cb_beta (CB 3)
    //   3. Fill cb_eps (CB 4) with epsilon scalar
    //   4. Fill cb_scaler (CB 5) with 1/K scalar
    //   5. For each sample, for each tile-row:
    //      - Read 32 RM sticks into cb_input_rm (CB 0), Ct pages
}
