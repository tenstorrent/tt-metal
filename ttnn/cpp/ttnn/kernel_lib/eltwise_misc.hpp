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
#include "api/compute/mask.h"
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

// Mask — bakes the hardcoded `mask_tile` LLK contract (mask lives at DataSlot+1) into
// the type per §1.4. Caller pre-loads data into DataSlot and mask into DataSlot+1; the
// op writes the masked result back into DataSlot. Compile-time `static_assert` rejects
// `DataSlot == D_LAST` (no slot for the mask tile).
//
// LLK supports `DataFormat::Float16`, `DataFormat::Float16_b`, and `DataFormat::Int32`.
template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>
struct Mask : BinaryOp<Mask<DF, DataSlot>, DataSlot, static_cast<Dst>(to_u32(DataSlot) + 1), DataSlot> {
    static_assert(
        to_u32(DataSlot) + 1 < DEST_AUTO_LIMIT,
        "Mask: DataSlot + 1 exceeds DEST_AUTO_LIMIT (mask tile lives at DataSlot + 1).");
    static ALWI void init() { mask_tile_init(); }
    static ALWI void exec_impl() { mask_tile(to_u32(DataSlot), to_u32(DataSlot) + 1, DF); }
};

}  // namespace compute_kernel_lib
