// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Reader Kernel
// Reads input tiles from DRAM via NOC0, generates scaler tiles.
// 3-pass read per row/col for softmax pipeline.
// Helpers needed: TensorAccessor (via dataflow_api), reduce_helpers_dataflow (prepare_reduce_scaler)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {}
