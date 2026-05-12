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
    static ALWI void call(uint32_t idst) { identity_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Negative : UnaryOp<Negative<Slot>, Slot> {
    static ALWI void init() { negative_tile_init(); }
    static ALWI void call(uint32_t idst) { negative_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Abs : UnaryOp<Abs<Slot>, Slot> {
    static ALWI void init() { abs_tile_init(); }
    static ALWI void call(uint32_t idst) { abs_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Sign : UnaryOp<Sign<Slot>, Slot> {
    static ALWI void init() { sign_tile_init(); }
    static ALWI void call(uint32_t idst) { sign_tile(idst); }
};

template <Dst Slot = Dst::D0>
struct Square : UnaryOp<Square<Slot>, Slot> {
    static ALWI void init() { square_tile_init(); }
    static ALWI void call(uint32_t idst) { square_tile(idst); }
};

// Typecast — compile-time in/out dtype encoded as numeric IDs (uint32_t form expected by LLK).
template <uint32_t InDF, uint32_t OutDF, Dst Slot = Dst::D0>
struct Typecast : UnaryOp<Typecast<InDF, OutDF, Slot>, Slot> {
    static ALWI void init() { typecast_tile_init<InDF, OutDF>(); }
    static ALWI void call(uint32_t idst) { typecast_tile<InDF, OutDF>(idst); }
};

}  // namespace compute_kernel_lib
