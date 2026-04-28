// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/trigonometry.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_trig.hpp
 * @brief Tier 2 trig: Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh,
 *        Asinh, Acosh, Atanh.
 *
 * Each is a 4-line CRTP struct calling its tile / tile_init pair.
 */

namespace compute_kernel_lib::eltwise {

using namespace ckernel;

#define _ELTWISE_TRIG_OP(NAME, FN)                         \
    template <Dst Slot = Dst::D0>                          \
    struct NAME : UnaryOp<NAME<Slot>, Slot> {              \
        ALWI void init() const { FN##_tile_init(); }       \
        ALWI void call(uint32_t d) const { FN##_tile(d); } \
    }

_ELTWISE_TRIG_OP(Sin, sin);
_ELTWISE_TRIG_OP(Cos, cos);
_ELTWISE_TRIG_OP(Tan, tan);
_ELTWISE_TRIG_OP(Asin, asin);
_ELTWISE_TRIG_OP(Acos, acos);
_ELTWISE_TRIG_OP(Atan, atan);
_ELTWISE_TRIG_OP(Sinh, sinh);
_ELTWISE_TRIG_OP(Cosh, cosh);
_ELTWISE_TRIG_OP(Asinh, asinh);
_ELTWISE_TRIG_OP(Acosh, acosh);
_ELTWISE_TRIG_OP(Atanh, atanh);

#undef _ELTWISE_TRIG_OP

}  // namespace compute_kernel_lib::eltwise
