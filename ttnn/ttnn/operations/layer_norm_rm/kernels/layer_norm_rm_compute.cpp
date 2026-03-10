// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel (STUB)
// Runs on math RISC-V core, performs layer normalization in tile domain
//
// When implemented, this kernel will perform these phases per tile-row block:
// Phase 1: tilize input (c_0 -> c_1)
// Phase 2: reduce mean (c_1 -> c_3)
// Phase 3: subtract mean (c_1, c_3 -> c_4)
// Phase 4: square centered (c_4 -> c_5)
// Phase 5: reduce variance (c_5 -> c_6)
// Phase 6: add epsilon + rsqrt (c_6, c_7 -> c_6)
// Phase 7: multiply by inv_std (c_4, c_6 -> c_24)
// Phase 8: multiply by gamma (c_24, c_25 -> c_29) [optional]
// Phase 9: add beta (c_29, c_26 -> c_29) [optional]
// Phase 10: untilize (c_24/c_29 -> c_16)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
// binary_op_helpers.hpp will be included when kernel is implemented
// #include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // Stub: no computation, kernels will be implemented in TDD stages
}
