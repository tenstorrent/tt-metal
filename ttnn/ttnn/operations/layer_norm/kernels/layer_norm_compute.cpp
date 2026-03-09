// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Compute Kernel (STUB)
// Runs on math RISC-V core (TRISC), performs FPU/SFPU operations.
//
// Per tile-row (Ht iterations), 6 phases:
//   P1: Mean reduction         reduce<SUM, REDUCE_ROW>(cb_input, cb_scaler) -> cb_mean_rstd
//   P2: Subtract mean          sub<COL>(cb_input, cb_mean_rstd) -> cb_x_minus_mean
//       [manual cb_pop_front(cb_input, Wt)]
//   P3: Square differences     square(cb_x_minus_mean) -> cb_diff_sq_temp (NoPop, persists for P5)
//   P4a: Variance reduction    reduce<SUM, REDUCE_ROW>(cb_diff_sq_temp, cb_scaler) -> cb_variance
//   P4b: Add eps + rsqrt       add<SCALAR>(cb_variance, cb_eps) -> cb_mean_rstd, then rsqrt
//   P5:  Multiply by rstd      mul<COL>(cb_x_minus_mean, cb_mean_rstd) -> cb_output or cb_diff_sq_temp
//   P5b (if has_gamma): mul<ROW>(cb_diff_sq_temp, cb_gamma) -> cb_x_minus_mean or cb_output
//   P5c (if has_beta):  add<ROW>(src, cb_beta) -> cb_output
//
// Full implementation includes:
//   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
//   #include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
//   #include "api/compute/eltwise_unary/rsqrt.h"

#include "api/compute/compute_kernel_hw_startup.h"

// Compile-time args (set by program descriptor):
//   Index 0: Ht          -- height in tiles
//   Index 1: Wt          -- width in tiles
//   Index 2: has_gamma   -- 1 if gamma is provided
//   Index 3: has_beta    -- 1 if beta is provided

constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

// CB indices
constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_eps = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_beta = 4;
constexpr uint32_t cb_output = 16;
constexpr uint32_t cb_mean_rstd = 24;  // dual-use: mean (P1-P2), rstd (P4b-P5)
constexpr uint32_t cb_x_minus_mean = 25;
constexpr uint32_t cb_variance = 26;
constexpr uint32_t cb_diff_sq_temp = 27;  // dual-use: (x-mean)^2 (P3-P4a), temp_norm (P5)

void kernel_main() {
    // Stub: Kernel startup initializes hardware subsystems.
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

    // Stub: Real implementation will run Ht tile-row iterations executing
    // all 6 normalization phases (P1-P5c) listed in the header comment above.
    // Currently a no-op stub that allows infrastructure to be tested.
}
