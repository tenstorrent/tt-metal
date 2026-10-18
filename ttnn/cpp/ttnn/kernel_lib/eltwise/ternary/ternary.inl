// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of ternary.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"

namespace compute_kernel_lib {

// Lerp — y = start + weight * (end - start). lerp_tile<DF>(start, end, weight, out).
template <DataFormat DF, Dst In0, Dst In1, Dst In2, Dst Out>
struct Lerp : TernaryOp<Lerp<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    static ALWI void init() { lerp_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        lerp_tile<DF>(
            to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(In2) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// SnakeBeta — snake_beta_tile<DF>(x, alpha, beta, out).
template <DataFormat DF, Dst In0, Dst In1, Dst In2, Dst Out>
struct SnakeBeta : TernaryOp<SnakeBeta<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    static ALWI void init() { snake_beta_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        snake_beta_tile<DF>(
            to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(In2) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// Addcmul — out = in0 + value * (in1 * in2). addcmul_tile<DF>(in0, in1, in2, out, value).
// Runtime `value` (uint32 bits) => instance exec, like FillScalar.
template <DataFormat DF, Dst In0, Dst In1, Dst In2, Dst Out>
struct Addcmul : TernaryOp<Addcmul<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    constexpr explicit Addcmul(uint32_t v) noexcept : value(v) {}
    constexpr Addcmul() noexcept : value(0) {}
    static ALWI void init() { addcmul_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        addcmul_tile<DF>(
            to_u32(In0) + slot_offset,
            to_u32(In1) + slot_offset,
            to_u32(In2) + slot_offset,
            to_u32(Out) + slot_offset,
            value);
    }
};

// Addcdiv — out = in0 + value * (in1 / in2). addcdiv_tile<DF>(in0, in1, in2, out, value).
template <DataFormat DF, Dst In0, Dst In1, Dst In2, Dst Out>
struct Addcdiv : TernaryOp<Addcdiv<DF, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    constexpr explicit Addcdiv(uint32_t v) noexcept : value(v) {}
    constexpr Addcdiv() noexcept : value(0) {}
    static ALWI void init() { addcdiv_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        addcdiv_tile<DF>(
            to_u32(In0) + slot_offset,
            to_u32(In1) + slot_offset,
            to_u32(In2) + slot_offset,
            to_u32(Out) + slot_offset,
            value);
    }
};

}  // namespace compute_kernel_lib
