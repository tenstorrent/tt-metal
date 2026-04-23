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
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/tanh_derivative.h"
#include "api/compute/copy_dest_values.h"

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

/**
 * @brief Copy values from one DEST slot to another within the same acquire window.
 *
 * Wraps `copy_dest_values<DF>(src, dst)`. No CB access — purely DEST-to-DEST.
 * Useful for saving an intermediate result before overwriting the source slot.
 *
 *   sfpu_chain(
 *       ...,
 *       Tanh<Approx::Exact, Dst::D0>{},     // D0 = tanh(x)
 *       CopyDest<Dst::D0, Dst::D1>{},        // D1 = tanh(x)  (save before squaring)
 *       Square<Dst::D0>{},                   // D0 = tanh²(x)
 *       ...)
 *
 * @tparam Src  DEST slot to copy from
 * @tparam Dst_ DEST slot to copy to
 * @tparam DF   Data format (default Float16_b)
 */
template <Dst Src, Dst Dst_, DataFormat DF = DataFormat::Float16_b>
struct CopyDest {
    static constexpr uint32_t src_idx = static_cast<uint32_t>(Src);
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Dst_);
    static constexpr uint32_t max_dst = (src_idx > dst_idx) ? src_idx : dst_idx;
    static_assert(src_idx < 8 && dst_idx < 8, "DEST slot exceeds maximum capacity (8)");

    ALWI void init() const;
    ALWI void exec(uint32_t offset = 0) const;
    ALWI void apply(uint32_t offset = 0) const {
        init();
        exec(offset);
    }
};

/**
 * @brief Fill DEST[Slot] with a runtime float constant.
 *
 * Stores the value in the struct at construction. Use this when the fill value
 * is only known at runtime (e.g. read from a kernel arg). For compile-time
 * constants prefer FillConst<bits> which avoids the runtime field.
 *
 *   sfpu_chain(FillScalar<Dst::D0>{1.0f}, Load<cb_x, Dst::D1>{}, SfpuSub<D0, D1, D0>{})
 *   // D0 = 1.0f - x  (rsub pattern)
 */
template <Dst Slot = Dst::D0>
struct FillScalar : UnaryOp<FillScalar<Slot>, Slot> {
    float value;
    constexpr explicit FillScalar(float v) : value(v) {}
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

/**
 * @brief Fill DEST[Slot] with a compile-time bit-pattern constant.
 *
 * Bits is the IEEE-754 uint32 representation of the desired float value.
 * Zero runtime overhead — constant is embedded at compile time.
 *
 *   sfpu_chain(FillConst<0x3F800000u, Dst::D0>{}, ...)  // fills 1.0f
 */
template <uint32_t Bits, Dst Slot = Dst::D0>
struct FillConst : UnaryOp<FillConst<Bits, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

/**
 * @brief Compute tanh derivative (sech²) in place on DEST[Slot].
 *
 * Uses the numerically stable sech²(x) = 4·exp(-2|x|) / (1 + exp(-2|x|))²
 * formula to avoid catastrophic cancellation in 1 − tanh²(x).
 *
 * Typical chain usage (tanh backward — needs GAP-11 indexed Load to migrate fully):
 *   sfpu_chain(Load<cb_input, Dst::D1>{}, TanhDerivative<Approx::Exact, Dst::D1>{},
 *              Load<cb_grad, Dst::D0>{}, SfpuMul<Dst::D0, Dst::D1, Dst::D0>{})
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct TanhDerivative : UnaryOp<TanhDerivative<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Math aliases ---
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_math.inl"
