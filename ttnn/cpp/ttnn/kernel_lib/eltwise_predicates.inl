// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_predicates.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/isinf_isnan.h"
#include "api/compute/eltwise_unary/logical_not.h"

namespace compute_kernel_lib {

#define ELTWISE_DECLARE_UNARY(Name, fn)                                                             \
    template <Dst Slot>                                                                             \
    struct Name : UnaryOp<Name<Slot>, Slot> {                                                       \
        static ALWI void init() { fn##_tile_init(); }                                               \
        static ALWI void exec_impl(uint32_t slot_offset) { fn##_tile(to_u32(Slot) + slot_offset); } \
    };

ELTWISE_DECLARE_UNARY(Eqz, eqz)
ELTWISE_DECLARE_UNARY(Nez, nez)
ELTWISE_DECLARE_UNARY(Ltz, ltz)
ELTWISE_DECLARE_UNARY(Lez, lez)
ELTWISE_DECLARE_UNARY(Gtz, gtz)
ELTWISE_DECLARE_UNARY(Gez, gez)
ELTWISE_DECLARE_UNARY(Isinf, isinf)
ELTWISE_DECLARE_UNARY(Isnan, isnan)
ELTWISE_DECLARE_UNARY(Isfinite, isfinite)
ELTWISE_DECLARE_UNARY(Isposinf, isposinf)
ELTWISE_DECLARE_UNARY(Isneginf, isneginf)
// LogicalNot — needs a DataFormat template arg; declared with explicit template.
template <DataFormat DF, Dst Slot>
struct LogicalNot : UnaryOp<LogicalNot<DF, Slot>, Slot> {
    static ALWI void init() { logical_not_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { logical_not_tile<DF>(to_u32(Slot) + slot_offset); }
};

#undef ELTWISE_DECLARE_UNARY

// Runtime-param scalar comparisons. Override exec(uint32_t) directly to capture param0.
#define ELTWISE_DECLARE_UNARY_PARAM(Name, fn)                                                                         \
    template <Dst Slot>                                                                                               \
    struct Name : UnaryOp<Name<Slot>, Slot> {                                                                         \
        uint32_t param0;                                                                                              \
        constexpr explicit Name(uint32_t p) noexcept : param0(p) {}                                                   \
        constexpr Name() noexcept : param0(0) {}                                                                      \
        static ALWI void init() { fn##_tile_init(); }                                                                 \
        ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { fn##_tile(to_u32(Slot) + slot_offset, param0); } \
    };

ELTWISE_DECLARE_UNARY_PARAM(UnaryEq, unary_eq)
ELTWISE_DECLARE_UNARY_PARAM(UnaryNe, unary_ne)
ELTWISE_DECLARE_UNARY_PARAM(UnaryGt, unary_gt)
ELTWISE_DECLARE_UNARY_PARAM(UnaryGe, unary_ge)
ELTWISE_DECLARE_UNARY_PARAM(UnaryLt, unary_lt)
ELTWISE_DECLARE_UNARY_PARAM(UnaryLe, unary_le)

#undef ELTWISE_DECLARE_UNARY_PARAM

}  // namespace compute_kernel_lib
