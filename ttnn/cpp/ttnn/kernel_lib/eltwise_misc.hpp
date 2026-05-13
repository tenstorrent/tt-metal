// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_misc.hpp
 * @brief Misc / utility SFPU op structs — Identity, Negative, Typecast, Sign, Abs, Square.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/compute_kernel_api.h"  // sign_tile, abs_tile, square_tile fallbacks

namespace compute_kernel_lib {

template <Dst Slot = Dst::D0>
struct Identity : UnaryOp<Identity<Slot>, Slot> {
    static ALWI void init() { identity_tile_init(); }
    static ALWI void exec_impl() { identity_tile(to_u32(Slot)); }
};

template <Dst Slot = Dst::D0>
struct Negative : UnaryOp<Negative<Slot>, Slot> {
    static ALWI void init() { negative_tile_init(); }
    static ALWI void exec_impl() { negative_tile(to_u32(Slot)); }
};

template <Dst Slot = Dst::D0>
struct Abs : UnaryOp<Abs<Slot>, Slot> {
    static ALWI void init() { abs_tile_init(); }
    static ALWI void exec_impl() { abs_tile(to_u32(Slot)); }
};

template <Dst Slot = Dst::D0>
struct Sign : UnaryOp<Sign<Slot>, Slot> {
    static ALWI void init() { sign_tile_init(); }
    static ALWI void exec_impl() { sign_tile(to_u32(Slot)); }
};

template <Dst Slot = Dst::D0>
struct Square : UnaryOp<Square<Slot>, Slot> {
    static ALWI void init() { square_tile_init(); }
    static ALWI void exec_impl() { square_tile(to_u32(Slot)); }
};

// Typecast — compile-time in/out dtype encoded as numeric IDs (uint32_t form expected by LLK).
template <uint32_t InDF, uint32_t OutDF, Dst Slot = Dst::D0>
struct Typecast : UnaryOp<Typecast<InDF, OutDF, Slot>, Slot> {
    static ALWI void init() { typecast_tile_init<InDF, OutDF>(); }
    static ALWI void exec_impl() { typecast_tile<InDF, OutDF>(to_u32(Slot)); }
};

}  // namespace compute_kernel_lib
