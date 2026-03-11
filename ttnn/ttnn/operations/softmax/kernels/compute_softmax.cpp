// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Compute Kernel
// 6-phase pipeline: max, reduce_max, sub+exp+sum_accum, reduce_sum+recip, final_mul
// Branches on DIM_W / DIM_H compile-time defines.
// Helpers needed: reduce_helpers_compute, binary_op_helpers, exp.h, recip.h

#include "api/compute/common.h"

void kernel_main() {}
