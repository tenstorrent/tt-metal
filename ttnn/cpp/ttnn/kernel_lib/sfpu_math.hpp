// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/cbrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/log1p.h"

namespace compute_kernel_lib {

// --- Simple Math ---

template <Approx approx = Approx::Exact, Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Exp : UnaryOp<Exp<approx, fast, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log : UnaryOp<Log<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct LogWithBase : UnaryOp<LogWithBase<Slot>, Slot> {
    uint32_t base_scale;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log1p : UnaryOp<Log1p<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt : UnaryOp<Sqrt<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Legacy legacy = Legacy::Off, Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Rsqrt : UnaryOp<Rsqrt<legacy, approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Cbrt : UnaryOp<Cbrt<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Legacy legacy = Legacy::On, Dst Slot = Dst::D0>
struct Recip : UnaryOp<Recip<legacy, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Abs : UnaryOp<Abs<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Neg : UnaryOp<Neg<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Square : UnaryOp<Square<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Sign : UnaryOp<Sign<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Signbit : UnaryOp<Signbit<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Exp2 : UnaryOp<Exp2<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Expm1 : UnaryOp<Expm1<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct PowerIterative : UnaryOp<PowerIterative<Slot>, Slot> {
    uint32_t int_exponent;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Rpow : UnaryOp<Rpow<Slot>, Slot> {
    uint32_t base_val;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Math aliases ---
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE,
    SfpuBatching B = SfpuBatching::Disabled>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_math.inl"
