// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel (STUB)
// Runs on math RISC-V core, performs FPU/SFPU operations
//
// Phases:
//   1. Tilize (RM input only)
//   2. Square (x^2)
//   3. Reduce row (mean)
//   4. Add epsilon + rsqrt
//   5. Re-tilize (RM input only, pass 2)
//   6. Normalize (broadcast multiply)
//   7. Gamma tilize + multiply (optional)
//   8. Untilize (RM output only)

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {}
