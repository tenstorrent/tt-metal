// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Writer Kernel (STUB)
// Runs on RISCV_1 (NCRISC), writes data to DRAM via NOC1
//
// TILE output: wait for tiles from cb_out, write to DRAM via TensorAccessor
// RM output: wait for sticks from cb_out_rm, write sticks to DRAM

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {}
