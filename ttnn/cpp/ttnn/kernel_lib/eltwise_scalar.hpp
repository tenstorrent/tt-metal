// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fmod.h"
#include "api/compute/eltwise_unary/remainder.h"
#include "api/compute/eltwise_unary/dropout.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_scalar.hpp
 * @brief Tier 2 scalar / runtime-arithmetic ops:
 *        AddScalar / SubScalar / MulScalar / DivScalar / RsubScalar (the
 *        binop_with_scalar family), Rsub, Rdiv, Fmod, Remainder, Dropout.
 *
 * "Scalar" here means the second operand is a runtime uint32_t (bit pattern
 * of the float / int constant), not an enum / DEST slot. Each struct holds
 * the runtime value as a public field.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

#define _ELTWISE_SCALAR_BINOP(NAME, FN)                           \
    template <Dst Slot = Dst::D0>                                 \
    struct NAME : UnaryOp<NAME<Slot>, Slot> {                     \
        uint32_t scalar;                                          \
        ALWI void init() const { binop_with_scalar_tile_init(); } \
        ALWI void call(uint32_t d) const { FN(d, scalar); }       \
    }

_ELTWISE_SCALAR_BINOP(AddScalar, add_unary_tile);
_ELTWISE_SCALAR_BINOP(SubScalar, sub_unary_tile);
_ELTWISE_SCALAR_BINOP(MulScalar, mul_unary_tile);
_ELTWISE_SCALAR_BINOP(DivScalar, div_unary_tile);
_ELTWISE_SCALAR_BINOP(RsubScalar, rsub_unary_tile);

#undef _ELTWISE_SCALAR_BINOP

template <Dst Slot = Dst::D0>
struct Rsub : UnaryOp<Rsub<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const { rsub_tile_init(); }
    ALWI void call(uint32_t d) const { rsub_tile(d, param0); }
};

template <RoundingMode RM = RoundingMode::None, Dst Slot = Dst::D0>
struct Rdiv : UnaryOp<Rdiv<RM, Slot>, Slot> {
    uint32_t value;
    ALWI void init() const { rdiv_tile_init(); }
    ALWI void call(uint32_t d) const { rdiv_tile<RM>(d, value); }
};

template <Dst Slot = Dst::D0>
struct Fmod : UnaryOp<Fmod<Slot>, Slot> {
    uint32_t param0;
    uint32_t param1;
    ALWI void init() const { fmod_tile_init(param0, param1); }
    ALWI void call(uint32_t d) const { fmod_tile(d, param0, param1); }
};

template <Dst Slot = Dst::D0>
struct Remainder : UnaryOp<Remainder<Slot>, Slot> {
    uint32_t param0;
    uint32_t param1;
    ALWI void init() const { remainder_tile_init(param0, param1); }
    ALWI void call(uint32_t d) const { remainder_tile(d, param0, param1); }
};

template <Dst Slot = Dst::D0>
struct Dropout : UnaryOp<Dropout<Slot>, Slot> {
    uint32_t probability;
    uint32_t scale_factor;
    ALWI void init() const { /* dropout has no per-tile init */ }
    ALWI void call(uint32_t d) const { dropout_tile(d, probability, scale_factor); }
};

}  // namespace compute_kernel_lib::eltwise
