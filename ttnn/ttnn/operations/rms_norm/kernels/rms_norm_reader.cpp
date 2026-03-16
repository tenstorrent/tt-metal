// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Reader Kernel (STUB)
// Runs on RISCV_0 (BRISC), reads data from DRAM via NOC0
//
// Two-pass reader per tile-row:
//   Pass 1: push Wt input tiles to cb_in (RM: read sticks; TILE: read tiles)
//   Pass 2: re-push same Wt tiles for normalization
//   Startup: fill cb_scaler with 1/W, fill cb_eps with epsilon
//   Optional: push gamma tiles per row

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {}
