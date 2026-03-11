// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Reads RM sticks from DRAM via TensorAccessor.
// Per tile-row: reads same 32 sticks 3 times (3 passes).
// At program start: fills scaler CB with 1/W, epsilon CB with eps.
// If gamma/beta: reads single stick 32 times to fill one tile-row.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {}
