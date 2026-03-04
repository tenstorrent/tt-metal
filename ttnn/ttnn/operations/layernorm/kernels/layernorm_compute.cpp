// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel
// 10-phase pipeline per tile-row:
//   1. Tilize (cb_in -> cb_tilized)
//   2. Mean reduce (cb_tilized -> cb_mean)
//   3. Subtract mean (cb_tilized - cb_mean -> cb_centered)
//   4. Square (cb_centered -> cb_normalized as temp)
//   5. Variance reduce (cb_normalized temp -> cb_var)
//   6. Add eps + rsqrt (cb_var -> cb_var in-place)
//   7. Normalize (cb_centered * cb_var -> cb_normalized)
//   8. Scale by gamma (cb_normalized * cb_gamma -> cb_normalized, conditional)
//   9. Shift by beta (cb_normalized + cb_beta -> cb_normalized, conditional)
//  10. Untilize (cb_normalized -> cb_out)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // Stub: compute kernel
    // Real implementation will execute the 10-phase normalization pipeline
    // per tile-row for nblocks_per_core iterations.
}
