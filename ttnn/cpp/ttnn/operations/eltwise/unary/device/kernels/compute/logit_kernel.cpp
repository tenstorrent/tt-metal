// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/logit.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

// logit(x) numerics summary
// =========================
// logit(x) = log(x / (1 - x)) is computed in a single fused SFPU pass
// (logit_tile -> calculate_logit, see ckernel_sfpu_logit.h) as
//
//     logit(x) = log_body(2x) - log1p(1 - 2x)
//
// for both fp32 and bf16 dest. This replaces the original formulation, which
// composed log(x / (1 - x)) out of ~16 tile ops (copy, rsub, div, log, ...).
//
// Why this is accurate -- in particular near x = 0.5, where logit'(0.5) = 4 is
// minimal so a constant ~1e-7 absolute error costs thousands of ULPs:
//
//   The original divider form's accuracy was bounded by div_binary_tile, an
//   SFPU Newton-Raphson reciprocal with ~1e-7 relative error, which after log
//   becomes a ~constant ~1e-7 absolute error -- thousands of ULPs near 0.5.
//
//   The 2x / (1 - 2x) form is reciprocal-free and cancellation-free near 0.5:
//   log_body(2x) range-reduces with exponent k = 0 (2x is near 1) and returns
//   the small poly(2x - 1) directly, and log1p(1 - 2x) returns the small
//   poly(1 - 2x) directly, so subtracting two small values gives
//   4(x - 0.5) + O((x-0.5)^3) with no catastrophic cancellation. See
//   ckernel_sfpu_logit.h for the full rationale.
//
// The kernel computes in fp32 and only rounds the result to bf16 on the final
// pack, so the bf16 dest path gets the same fp32-accurate logit (correctly
// rounded), which is why both dtypes share the one fused op.

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    CircularBuffer cb_in(cb_input);
    CircularBuffer cb_out(cb_output);

    init_sfpu(cb_input, cb_output);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_in.wait_front(1);
        cb_out.reserve_back(1);

        // Single DEST pass: copy x in, optionally clamp to [eps, 1 - eps], then
        // logit(x) -- all chained on the same tile in one acquire so x never
        // round-trips through L1. logit is one fused SFPU pass (fp32 and bf16
        // alike; it computes in fp32 and rounds to the dest dtype on pack).
        tile_regs_acquire();

        copy_tile_init(cb_input);
        copy_tile(cb_input, 0, 0);
#ifdef CLAMP
        clamp_tile_init();
        clamp_tile(0, packed_scalar1, packed_scalar2);
#endif
        logit_tile_init();
        logit_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);
        tile_regs_release();

        cb_in.pop_front(1);
        cb_out.push_back(1);
    }
}
