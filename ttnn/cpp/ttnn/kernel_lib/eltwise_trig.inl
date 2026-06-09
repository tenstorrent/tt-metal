// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_trig.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/trigonometry.h"

namespace compute_kernel_lib {

#define ELTWISE_DECLARE_UNARY(Name, fn)                                                             \
    template <Dst Slot>                                                                             \
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
