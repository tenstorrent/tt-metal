// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Compute Kernel (STUB)
// Runs on math RISC-V core, performs layer normalization computation.
//
// Compile-time args:
//   [0] Wt              : uint32_t - Width in tiles
//   [1] num_rows        : uint32_t - Number of tile-rows to process
//   [2] gamma_has_value : uint32_t - 1 if gamma is provided
//   [3] beta_has_value  : uint32_t - 1 if beta is provided
//
// Per-row computation:
//   Phase 1: REDUCE_ROW(input) -> mean (1 tile, Col0 valid)
//   Phase 2: SUB(input, mean) [COL broadcast] -> centered (Wt tiles)
//   Phase 3: SQUARE(centered) -> squared (Wt tiles)
//   Phase 4: REDUCE_ROW(squared) -> variance (1 tile, Col0 valid)
//   Phase 5: ADD(variance, eps) [SCALAR broadcast] -> var_eps; RSQRT(var_eps) -> rstd
//   Phase 6: MUL(centered, rstd) [COL broadcast] -> normalized/output
//   Phase 7 (if gamma): MUL(normalized, gamma) [ROW broadcast] -> gamma_out/output
//   Phase 8 (if beta):  ADD(gamma_out, beta) [ROW broadcast] -> output

#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // Stub: compute kernel for layer_norm
    // Real implementation will perform the 6-8 phase normalization per row
    // using compute_kernel_lib helpers:
    //   - reduce<SUM, REDUCE_ROW> for mean and variance
    //   - sub<COL> for centering
    //   - square for squaring centered values
    //   - add<SCALAR> for adding epsilon
    //   - manual rsqrt_tile for computing reciprocal std
    //   - mul<COL> for normalizing
    //   - mul<ROW> and add<ROW> for optional affine transform
}
