// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation detail of eltwise_rounding.hpp — full op-struct definitions live here. The public
// header forward-declares these structs and includes this file at its tail.

#include "api/compute/eltwise_unary/rounding.h"

namespace compute_kernel_lib {

template <Dst Slot>
struct Floor : UnaryOp<Floor<Slot>, Slot> {
    static ALWI void init() { rounding_op_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { floor_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Ceil : UnaryOp<Ceil<Slot>, Slot> {
    static ALWI void init() { rounding_op_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { ceil_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Trunc : UnaryOp<Trunc<Slot>, Slot> {
    static ALWI void init() { rounding_op_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { trunc_tile(to_u32(Slot) + slot_offset); }
};

template <Dst Slot>
struct Frac : UnaryOp<Frac<Slot>, Slot> {
    static ALWI void init() { rounding_op_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { frac_tile(to_u32(Slot) + slot_offset); }
};

// Round — runtime decimals.
template <Dst Slot>
struct Round : UnaryOp<Round<Slot>, Slot> {
    int32_t decimals;
    constexpr explicit Round(int32_t d) noexcept : decimals(d) {}
    constexpr Round() noexcept : decimals(0) {}
    static ALWI void init() { rounding_op_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const { round_tile(to_u32(Slot) + slot_offset, decimals); }
};

}  // namespace compute_kernel_lib
