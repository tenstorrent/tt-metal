// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/tanh_derivative.h"
#include "api/compute/logsigmoid.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_activations.hpp
 * @brief Activation op structs (and their derivatives) for the V2 eltwise
 *        helper family. Backward-pass kernels (gelu_bw, tanh_bw, etc.) use the
 *        derivative variants chained with a CopyTile + binary mul.
 *
 * NOTE: this file does NOT include `sfpu_helpers.hpp`. Calls go directly into
 * `compute_kernel_api/eltwise_unary/...h`.
 */

namespace compute_kernel_lib {

// =============================================================================
// TanhDerivative — used by tanh_bw. Programs the SFPU type register for
// tanh_derivative; init is HW-light so hoist-safe per init_hoist_survey.md.
// =============================================================================

template <Dst Slot = Dst::D0>
struct TanhDerivative : UnaryOp<TanhDerivative<Slot>, Slot> {
    // Per init_hoist_survey: derivative inits touch only the SFPU type
    // register (not the polynomial LUT) — safe to coexist with another LUT op.
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::tanh_derivative_tile_init(); }
    ALWI static void call(uint32_t dst) { ckernel::tanh_derivative_tile(dst); }
};

// =============================================================================
// GeluDerivative — used by gelu_bw. Same shape as TanhDerivative.
// =============================================================================

template <Dst Slot = Dst::D0>
struct GeluDerivative : UnaryOp<GeluDerivative<Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::gelu_derivative_tile_init(); }
    ALWI static void call(uint32_t dst) { ckernel::gelu_derivative_tile(dst); }
};

// =============================================================================
// LogSigmoid — fused 2-DEST kernel: out = log(sigmoid(in0)) = -log(1 + exp(-in0))
// LLK signature: `logsigmoid_tile(idst_in0, idst_in1, idst_out)` where In1 is
// expected to hold exp(-x). Caller composes the chain:
//   CopyTile<cb_x, D0>  CopyTile<cb_x, D1, NoWaitNoPop>  Negative<D1>
//   Exp<.., D1>  LogSigmoid<D0, D1, D0>
// =============================================================================

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct LogSigmoid : BinaryOp<LogSigmoid<In0, In1, Out>, In0, In1, Out> {
    // Programs SFPU LUT for the log path — same chain trait as Log/Exp.
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() { ckernel::logsigmoid_tile_init(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t out_idx) { ckernel::logsigmoid_tile(i0, i1, out_idx); }
};

}  // namespace compute_kernel_lib
