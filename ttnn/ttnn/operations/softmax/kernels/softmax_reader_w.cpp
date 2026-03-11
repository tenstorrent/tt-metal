// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Reader Kernel (dim=-1, width reduction)
// Sequential tile reads: Wt tiles per row, prepare reduce scaler in c_2

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {}
