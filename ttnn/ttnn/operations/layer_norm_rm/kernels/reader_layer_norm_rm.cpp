// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Reader Kernel
// Reads RM sticks from DRAM via TensorAccessor, generates reduce scaler and
// epsilon tile, optionally reads gamma/beta tiles. Sends each tile-row 3 times
// (3-pass streaming) into CB c_0.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {}
