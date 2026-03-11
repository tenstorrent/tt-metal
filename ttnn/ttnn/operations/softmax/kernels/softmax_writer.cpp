// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Writer Kernel (shared for dim=-1 and dim=-2)
// Generic tile writer: drains c_16 to DRAM using TensorAccessor

#include "api/dataflow/dataflow_api.h"

void kernel_main() {}
