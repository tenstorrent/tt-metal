// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_trig.hpp
 * @brief Trigonometric SFPU op structs.
 *
 * Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/compute/eltwise_unary/trigonometry.h"

namespace compute_kernel_lib {

#define ELTWISE_DECLARE_UNARY(Name, fn)                                                             \
    template <Dst Slot = Dst::D0>                                                                   \
    struct Name : UnaryOp<Name<Slot>, Slot> {                                                       \
        static ALWI void init() { fn##_tile_init(); }                                               \
        static ALWI void exec_impl(uint32_t slot_offset) { fn##_tile(to_u32(Slot) + slot_offset); } \
    };

ELTWISE_DECLARE_UNARY(Sin, sin)
ELTWISE_DECLARE_UNARY(Cos, cos)
ELTWISE_DECLARE_UNARY(Tan, tan)
ELTWISE_DECLARE_UNARY(Asin, asin)
ELTWISE_DECLARE_UNARY(Acos, acos)
ELTWISE_DECLARE_UNARY(Atan, atan)
ELTWISE_DECLARE_UNARY(Sinh, sinh)
ELTWISE_DECLARE_UNARY(Cosh, cosh)
ELTWISE_DECLARE_UNARY(Asinh, asinh)
ELTWISE_DECLARE_UNARY(Acosh, acosh)
ELTWISE_DECLARE_UNARY(Atanh, atanh)

#undef ELTWISE_DECLARE_UNARY

}  // namespace compute_kernel_lib
