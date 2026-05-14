// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_binary_sfpu.hpp
 * @brief DEST-DEST SFPU binary op structs.
 *
 * Wraps `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`, `div_binary_tile` —
 * these operate on two DEST slots, write back to a (possibly third) DEST slot. No CB
 * input/output (DestOnly via BinaryOp CRTP base).
 *
 * Used by mid-loop fused patterns (e.g. tanhshrink: `x - tanh(x)`, hardswish: `x * hardsigmoid(x)`)
 * where both operands are already in DEST.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"

namespace compute_kernel_lib {

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct AddBinary : BinaryOp<AddBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { add_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        add_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SubBinary : BinaryOp<SubBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { sub_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        sub_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct MulBinary : BinaryOp<MulBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { mul_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        mul_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct DivBinary : BinaryOp<DivBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { div_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        div_binary_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

// binary_max_tile / binary_min_tile — SFPU two-DEST max/min into third DEST slot.
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct BinaryMax : BinaryOp<BinaryMax<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_max_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_max_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct BinaryMin : BinaryOp<BinaryMin<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { binary_min_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        binary_min_tile(to_u32(In0) + slot_offset, to_u32(In1) + slot_offset, to_u32(Out) + slot_offset);
    }
};

}  // namespace compute_kernel_lib
