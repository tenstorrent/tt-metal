// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel
// Waits for Wt tile-sized pages in c_17 per tile-row.
// Extracts 32 RM sticks and writes to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {}
