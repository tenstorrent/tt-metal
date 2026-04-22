// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/erf_erfc.h"
#include "api/compute/eltwise_unary/erfinv.h"
#include "api/compute/eltwise_unary/i0.h"
#include "api/compute/eltwise_unary/i1.h"
#include "api/compute/eltwise_unary/lgamma.h"
#include "api/compute/logsigmoid.h"

namespace compute_kernel_lib {

// --- Error / Special Functions ---

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erf : UnaryOp<Erf<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erfc : UnaryOp<Erfc<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Erfinv : UnaryOp<Erfinv<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct I0 : UnaryOp<I0<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct I1 : UnaryOp<I1<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Lgamma : UnaryOp<Lgamma<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

/**
 * @brief logsigmoid(x) = -softplus(-x) using two pre-loaded DEST slots.
 *
 * Requires In1 to hold exp(-x) before exec() is called. Typical chain usage:
 *
 *   sfpu_chain(
 *       Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},    // x → D0
 *       Neg<Dst::D0>{},                                       // D0 = -x
 *       Exp<Approx::Fast, Approx::Fast, Dst::D0>{},           // D0 = exp(-x)
 *       Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},     // D1 = x (original)
 *       Logsigmoid<Dst::D0, Dst::D1, Dst::D0>{})             // D0 = logsigmoid(x)
 *       // Note: Logsigmoid(in0=exp(-x), in1=x, out=result)
 *
 * @tparam In0  DEST slot holding exp(-x)
 * @tparam In1  DEST slot holding x
 * @tparam Out  DEST slot for result
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct Logsigmoid : BinaryOp<Logsigmoid<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

// --- Special function aliases ---
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_erf(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_erfc(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_erfinv(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_i0(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_i1(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_lgamma(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_special.inl"
