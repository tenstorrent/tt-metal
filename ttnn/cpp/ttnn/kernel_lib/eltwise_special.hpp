// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/erf_erfc.h"
#include "api/compute/eltwise_unary/erfinv.h"
#include "api/compute/eltwise_unary/i0.h"
#include "api/compute/eltwise_unary/i1.h"
#include "api/compute/eltwise_unary/lgamma.h"
#include "api/compute/eltwise_unary/digamma.h"
#include "api/compute/logsigmoid.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_special.hpp
 * @brief Tier 2 special functions: Erf, Erfc, Erfinv, I0, I1, Lgamma, Digamma.
 *
 * `Lgamma` uses the Stirling approximation (`lgamma_stirling_*`). The
 * adjusted-argument variant is left for a follow-up alias once a kernel
 * needs it.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

template <Approx A = Approx::Fast, Dst Slot = Dst::D0>
struct Erf : UnaryOp<Erf<A, Slot>, Slot> {
    ALWI void init() const { erf_tile_init<static_cast<bool>(A)>(); }
    ALWI void call(uint32_t d) const { erf_tile<static_cast<bool>(A)>(d); }
};

template <Approx A = Approx::Fast, Dst Slot = Dst::D0>
struct Erfc : UnaryOp<Erfc<A, Slot>, Slot> {
    ALWI void init() const { erfc_tile_init(); }
    ALWI void call(uint32_t d) const { erfc_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Erfinv : UnaryOp<Erfinv<Slot>, Slot> {
    ALWI void init() const { erfinv_tile_init(); }
    ALWI void call(uint32_t d) const { erfinv_tile(d); }
};

template <Dst Slot = Dst::D0>
struct I0 : UnaryOp<I0<Slot>, Slot> {
    ALWI void init() const { i0_tile_init(); }
    ALWI void call(uint32_t d) const { i0_tile(d); }
};

template <Dst Slot = Dst::D0>
struct I1 : UnaryOp<I1<Slot>, Slot> {
    ALWI void init() const { i1_tile_init(); }
    ALWI void call(uint32_t d) const { i1_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Lgamma : UnaryOp<Lgamma<Slot>, Slot> {
    ALWI void init() const { lgamma_stirling_tile_init(); }
    ALWI void call(uint32_t d) const { lgamma_stirling_tile(d); }
};

template <Dst Slot = Dst::D0>
struct Digamma : UnaryOp<Digamma<Slot>, Slot> {
    ALWI void init() const { digamma_tile_init(); }
    ALWI void call(uint32_t d) const { digamma_tile(d); }
};

/**
 * Logsigmoid — fused 3-DEST binary op (lessons §9):
 *   In0 holds x, In1 holds exp(-x). The LLK does
 *   `-log(1 + exp(-x))` numerically stable, writing the result to Out.
 *
 * Caller is responsible for filling In1 with `exp(-x)` (i.e. via
 * Neg + Exp on the chain element preceding this one).
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct Logsigmoid : BinaryOp<Logsigmoid<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const { logsigmoid_tile_init(); }
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const { logsigmoid_tile(a, b, c); }
};

}  // namespace compute_kernel_lib::eltwise
