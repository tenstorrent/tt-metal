// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/trigonometry.h"

namespace compute_kernel_lib {

// --- Trigonometry ---

template <Dst Slot = Dst::D0>
struct Sin : UnaryOp<Sin<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Cos : UnaryOp<Cos<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Tan : UnaryOp<Tan<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Asin : UnaryOp<Asin<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Acos : UnaryOp<Acos<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Atan : UnaryOp<Atan<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Sinh : UnaryOp<Sinh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Cosh : UnaryOp<Cosh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Asinh : UnaryOp<Asinh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Acosh : UnaryOp<Acosh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Atanh : UnaryOp<Atanh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Trig aliases ---
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_sin(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_cos(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_tan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_asin(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_acos(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_atan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_sinh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_cosh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_asinh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_acosh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::NONE>
ALWI void sfpu_atanh(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_trig.inl"
