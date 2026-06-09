// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_ternary.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"

namespace compute_kernel_lib {

namespace detail {
// Max DEST slot index + 1 across the four ternary slots (per-lane footprint).
template <Dst In0, Dst In1, Dst In2, Dst Out>
inline constexpr uint32_t ternary_lane_width() {
    uint32_t m = to_u32(In0);
    if (to_u32(In1) > m) {
        m = to_u32(In1);
    }
    if (to_u32(In2) > m) {
        m = to_u32(In2);
    }
    if (to_u32(Out) > m) {
        m = to_u32(Out);
    }
    return m + 1;
}
}  // namespace detail

// Lerp — y = start + weight * (end - start). lerp_tile<DF>(start, end, weight, out).
template <DataFormat DF, Dst In0, Dst In1, Dst In2, Dst Out>
struct Lerp : DestOnlyTag {
    static constexpr uint32_t lane_width = detail::ternary_lane_width<In0, In1, In2, Out>();
    static ALWI void init() { lerp_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        lerp_tile<DF>(
            to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(In2) + slot_offset, to_u32(Out) + slot_offset);
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { exec_impl(slot_offset); }
};

// SnakeBeta — snake_beta_tile<DF>(x, alpha, beta, out).
template <DataFormat DF, Dst In0, Dst In1, Dst In2, Dst Out>
struct SnakeBeta : DestOnlyTag {
    static constexpr uint32_t lane_width = detail::ternary_lane_width<In0, In1, In2, Out>();
    static ALWI void init() { snake_beta_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        snake_beta_tile<DF>(
            to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(In2) + slot_offset, to_u32(Out) + slot_offset);
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { exec_impl(slot_offset); }
};

// Addcmul — out = in0 + value * (in1 * in2). addcmul_tile<DF>(in0, in1, in2, out, value).
// Runtime `value` (uint32 bits) => instance exec, like FillScalar.
template <DataFormat DF, Dst In0, Dst In1, Dst In2, Dst Out>
struct Addcmul : DestOnlyTag {
    uint32_t value;
    constexpr explicit Addcmul(uint32_t v) noexcept : value(v) {}
    constexpr Addcmul() noexcept : value(0) {}
    static constexpr uint32_t lane_width = detail::ternary_lane_width<In0, In1, In2, Out>();
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
struct Addcdiv : DestOnlyTag {
    uint32_t value;
    constexpr explicit Addcdiv(uint32_t v) noexcept : value(v) {}
    constexpr Addcdiv() noexcept : value(0) {}
    static constexpr uint32_t lane_width = detail::ternary_lane_width<In0, In1, In2, Out>();
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
