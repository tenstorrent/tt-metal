// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Compute Kernel (STUB)
//
// Full computation pipeline (incrementally filled via TDD stages):
//
// Stage 1 (data_pipeline):   tilize(cb_in_rm -> cb_tilized) + untilize(cb_tilized -> cb_out_rm)
// Stage 2 (reduce_mean):     + reduce(SUM_ROW) -> cb_mean + sub<COL>(cb_tilized, cb_mean -> cb_centered)
// Stage 3 (variance_normalize): + square, reduce variance, add+rsqrt -> cb_inv_std, mul<COL>
// Stage 4 (affine_transform): + mul(gamma), add(beta)
//
// Compile-time args:
//   [0] Wt               — tiles per row (W / 32)
//   [1] nblocks_per_core — tile-rows this core processes
//   [2] has_gamma        — 1 if gamma present, 0 otherwise
//   [3] has_beta         — 1 if beta present, 0 otherwise
//   [4] epsilon_packed   — epsilon as uint32 IEEE-754 bits

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // Stub: no-op. The real implementation will execute 8 phases per tile-row.
}
