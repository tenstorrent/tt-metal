// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rand.h"

// NOTE: BitwiseNot, BitwiseAnd, BitwiseOr, BitwiseXor, LeftShift, RightShift
// are excluded — their ckernel headers pollute the global namespace with operators
// that conflict with reduce.h.

namespace compute_kernel_lib {

// --- Type / Identity / Bitwise ---

template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot = Dst::D0>
struct Typecast : UnaryOp<Typecast<in_dtype, out_dtype, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Identity : UnaryOp<Identity<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Fill and Random ---

template <Dst Slot = Dst::D0>
struct FillTile : UnaryOp<FillTile<Slot>, Slot> {
    float fill_val;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct FillTileBitcast : UnaryOp<FillTileBitcast<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct RandTile : UnaryOp<RandTile<Slot>, Slot> {
    uint32_t from;
    uint32_t scale;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

}  // namespace compute_kernel_lib

#include "sfpu_misc.inl"
