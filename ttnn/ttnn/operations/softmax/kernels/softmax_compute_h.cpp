// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Compute Kernel (dim=-2, height reduction)
// 4-phase per chunk: max(REDUCE_COL), sub+exp(ROW broadcast), sum(REDUCE_COL)+recip, mul(ROW broadcast)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

void kernel_main() {}
