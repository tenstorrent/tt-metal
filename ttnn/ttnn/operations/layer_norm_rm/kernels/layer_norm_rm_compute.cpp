// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel (stub)
//
// Per-block pipeline:
//   Phase 1:  Tilize (cb_rm_in -> cb_tilized)
//   Phase 2:  Reduce row for mean (cb_tilized, cb_reduce_scaler -> cb_mean)
//   Phase 3:  Subtract mean, broadcast COL (cb_tilized, cb_mean -> cb_centered)
//   Phase 4:  Square centered (cb_centered -> cb_centered_sq)
//   Phase 5:  Reduce row for variance (cb_centered_sq, cb_reduce_scaler -> cb_var)
//   Phase 6:  Add eps + rsqrt (cb_var, cb_eps -> cb_inv_std)
//   Phase 7:  Multiply inv_std, broadcast SCALAR (cb_centered, cb_inv_std -> cb_out_pre_untilize)
//   Phase 8:  Optional: multiply gamma, broadcast ROW
//   Phase 9:  Optional: add beta, broadcast ROW
//   Phase 10: Untilize (cb_out_pre_untilize -> cb_rm_out)
//
// Includes needed for full implementation:
//   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
//   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
//   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
//   #include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
//   #include "api/compute/eltwise_unary/rsqrt.h"

#include <cstdint>
#include "api/compute/common.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // Runtime args
    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Stub: do nothing -- real implementation will execute all 10 phases
    // per block in the main loop.
}
