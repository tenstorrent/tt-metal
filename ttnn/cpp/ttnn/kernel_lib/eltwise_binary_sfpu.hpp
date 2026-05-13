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

namespace compute_kernel_lib {

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct AddBinary : BinaryOp<AddBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { add_binary_tile_init(); }
    static ALWI void exec_impl() { add_binary_tile(to_u32(In0), to_u32(In1), to_u32(Out)); }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SubBinary : BinaryOp<SubBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { sub_binary_tile_init(); }
    static ALWI void exec_impl() { sub_binary_tile(to_u32(In0), to_u32(In1), to_u32(Out)); }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct MulBinary : BinaryOp<MulBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { mul_binary_tile_init(); }
    static ALWI void exec_impl() { mul_binary_tile(to_u32(In0), to_u32(In1), to_u32(Out)); }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct DivBinary : BinaryOp<DivBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { div_binary_tile_init(); }
    static ALWI void exec_impl() { div_binary_tile(to_u32(In0), to_u32(In1), to_u32(Out)); }
};

}  // namespace compute_kernel_lib
