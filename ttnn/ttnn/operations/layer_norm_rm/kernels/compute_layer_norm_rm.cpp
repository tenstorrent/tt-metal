// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Phases per tile-row:
//   1. Tilize c_0 -> c_1
//   2. Reduce row (c_1, c_2) -> c_24 (mean)
//   3. Tilize c_0 -> c_1 (pass 2)
//   4. Sub<COL>(c_1, c_24) -> c_25 (centered)
//   5. Square(c_25) -> c_1
//   6. Reduce row (c_1, c_2) -> c_26 (variance)
//   7. Add<SCALAR>(c_26, c_3) + rsqrt -> c_27 (inv_std)
//   8. Tilize c_0 -> c_1 (pass 3)
//   9. Sub<COL>(c_1, c_24) -> c_25 (centered)
//  10. Mul<COL>(c_25, c_27) -> c_16 (normalized)
//  11. Mul<NONE>(c_16, c_4) -> c_25 (if gamma)
//  12. Add<NONE>(c_25, c_5) -> c_16 (if beta)
//  13. Untilize c_16 -> c_17

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {}
