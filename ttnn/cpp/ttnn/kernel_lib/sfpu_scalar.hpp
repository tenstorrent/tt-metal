// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fmod.h"
#include "api/compute/eltwise_unary/remainder.h"
#include "api/compute/eltwise_unary/dropout.h"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

namespace compute_kernel_lib {

// --- Scalar Arithmetic ---

template <Dst Slot = Dst::D0>
struct AddScalar : UnaryOp<AddScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct SubScalar : UnaryOp<SubScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct MulScalar : UnaryOp<MulScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct DivScalar : UnaryOp<DivScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct RsubScalar : UnaryOp<RsubScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Rsub : UnaryOp<Rsub<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <RoundingMode rounding_mode = RoundingMode::None, Dst Slot = Dst::D0>
struct Rdiv : UnaryOp<Rdiv<rounding_mode, Slot>, Slot> {
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Fmod : UnaryOp<Fmod<Slot>, Slot> {
    uint32_t param0;
    uint32_t param1;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Remainder : UnaryOp<Remainder<Slot>, Slot> {
    uint32_t param0;
    uint32_t param1;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Dropout : UnaryOp<Dropout<Slot>, Slot> {
    uint32_t probability;
    uint32_t scale_factor;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

}  // namespace compute_kernel_lib

#include "sfpu_scalar.inl"
