// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel (stub)
// Per tile-row: tilize -> reduce_mean -> sub_mean -> square ->
//               reduce_var -> add_eps+rsqrt -> mul_rsqrt ->
//               (optional: mul_gamma, add_beta) -> untilize

#include "api/compute/compute_kernel_hw_startup.h"
// Full implementation will also need:
// #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
// #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
// #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
// #include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
// #include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // Stub: compute kernel
    // Real implementation will:
    // 1. compute_kernel_hw_startup(c_0, c_8, c_16)
    // 2. Wait on c_9 (epsilon) once
    // 3. Per tile-row loop:
    //    Phase 1: tilize c_0 -> c_1
    //    Phase 2: reduce_mean c_1 -> c_24
    //    Phase 3: sub_mean c_1, c_24 -> c_25 (manual pop c_1)
    //    Phase 4: square c_25 -> c_26
    //    Phase 5: reduce_var c_26 -> c_27
    //    Phase 6: add_eps c_27, c_9 -> c_28 + rsqrt post-op
    //    Phase 7: mul_rsqrt c_25, c_28 -> c_31
    //    (Phase 8: mul_gamma c_31, c_29 -> c_25)
    //    (Phase 9: add_beta cb_in, c_30 -> cb_out)
    //    Phase 10: untilize cb_final -> c_16
    // 4. Pop c_9, c_8
}
