// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Opt-in extension for sfpu_helpers: bitwise and shift operations.
//
// These are split out because the underlying ckernel headers
// (ckernel_sfpu_bitwise_and.h, etc.) contain `using namespace sfpi;`
// at file scope, which introduces operator overloads (e.g. operator&)
// that create ambiguous-overload errors when combined with reduce.h
// or other LLK headers.
//
// Include this header *only* in kernels that need BitwiseAnd / BitwiseOr /
// BitwiseXor / BitwiseNot / LeftShift / RightShift op structs.

#pragma once

#include "sfpu_helpers.hpp"

#include "api/compute/eltwise_unary/bitwise_and.h"
#include "api/compute/eltwise_unary/bitwise_or.h"
#include "api/compute/eltwise_unary/bitwise_xor.h"
#include "api/compute/eltwise_unary/bitwise_not.h"
#include "api/compute/eltwise_unary/left_shift.h"
#include "api/compute/eltwise_unary/right_shift.h"

namespace compute_kernel_lib {

// --- Bitwise Ops ---

template <Dst Slot = Dst::D0>
struct BitwiseNot : UnaryOp<BitwiseNot<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct BitwiseAnd : UnaryOp<BitwiseAnd<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct BitwiseOr : UnaryOp<BitwiseOr<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct BitwiseXor : UnaryOp<BitwiseXor<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Shift Ops ---

template <Dst Slot = Dst::D0>
struct LeftShift : UnaryOp<LeftShift<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct RightShift : UnaryOp<RightShift<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Implementations ---

template <Dst Slot>
ALWI void BitwiseNot<Slot>::init() const {
    bitwise_not_tile_init();
}
template <Dst Slot>
ALWI void BitwiseNot<Slot>::call(uint32_t d0) const {
    bitwise_not_tile(d0);
}
template <Dst Slot>
ALWI void BitwiseAnd<Slot>::init() const {
    bitwise_and_tile_init();
}
template <Dst Slot>
ALWI void BitwiseAnd<Slot>::call(uint32_t d0) const {
    bitwise_and_tile(d0, param0);
}
template <Dst Slot>
ALWI void BitwiseOr<Slot>::init() const {
    bitwise_or_tile_init();
}
template <Dst Slot>
ALWI void BitwiseOr<Slot>::call(uint32_t d0) const {
    bitwise_or_tile(d0, param0);
}
template <Dst Slot>
ALWI void BitwiseXor<Slot>::init() const {
    bitwise_xor_tile_init();
}
template <Dst Slot>
ALWI void BitwiseXor<Slot>::call(uint32_t d0) const {
    bitwise_xor_tile(d0, param0);
}
template <Dst Slot>
ALWI void LeftShift<Slot>::init() const {
    left_shift_tile_init();
}
template <Dst Slot>
ALWI void LeftShift<Slot>::call(uint32_t d0) const {
    left_shift_tile(d0, param0);
}
template <Dst Slot>
ALWI void RightShift<Slot>::init() const {
    right_shift_tile_init();
}
template <Dst Slot>
ALWI void RightShift<Slot>::call(uint32_t d0) const {
    right_shift_tile(d0, param0);
}

}  // namespace compute_kernel_lib
